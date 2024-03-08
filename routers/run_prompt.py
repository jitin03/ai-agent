from fastapi import APIRouter, Depends, HTTPException, Path
import logging
import asyncio
import os
import shutil
import subprocess
import argparse
from langchain_groq import ChatGroq
from pydantic import BaseModel
import torch
from semantic_router.encoders import HuggingFaceEncoder
from .semantic_routers import routes
from .semanic_router_response import appointment_inquiry,politics,chitchat,greetings,done_task
from .utils import log_to_csv
from .utils import get_embeddings
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.callbacks.manager import CallbackManager
# from langchain.embeddings import HuggingFaceEmbeddings
from semantic_router import RouteLayer
from .prompt_template_utils import get_prompt_template
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler  # for streaming response
from langchain.callbacks import AsyncIteratorCallbackHandler
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.vectorstores import Chroma
from werkzeug.utils import secure_filename
from typing import AsyncIterable
from fastapi.middleware.cors import CORSMiddleware
from .constants import (
    # CHROMA_SETTINGS,
    EMBEDDING_MODEL_NAME, PERSIST_DIRECTORY, MODEL_ID, MODEL_BASENAME,
    MAX_NEW_TOKENS)
from fastapi.responses import StreamingResponse
# API queue addition
from threading import Lock
from starlette import status
import sys
import textwrap
from langchain.vectorstores.redis import Redis
from redisvl.extensions.llmcache import SemanticCache
encoder = HuggingFaceEncoder()
from .rag_redis.config import (
    INDEX_NAME, INDEX_SCHEMA, REDIS_URL
)
from transformers import (
    GenerationConfig,
    pipeline,
)
from langchain.llms import HuggingFacePipeline
from .load_models import (
    load_quantized_model_awq,
    load_quantized_model_gguf_ggml,
    load_quantized_model_qptq,
    load_full_model,
)

request_lock = Lock()

if torch.backends.mps.is_available():
    DEVICE_TYPE = "mps"
elif torch.cuda.is_available():
    DEVICE_TYPE = "cuda"
else:
    DEVICE_TYPE = "cpu"
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
SHOW_SOURCES = True
logging.info(f"Running on: {DEVICE_TYPE}")
logging.info(f"Display Source Documents set to: {SHOW_SOURCES}")

EMBEDDINGS = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": DEVICE_TYPE})

# uncomment the following line if you used HuggingFaceEmbeddings in the ingest.py
# EMBEDDINGS = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
# if os.path.exists(PERSIST_DIRECTORY):
#     try:
#         shutil.rmtree(PERSIST_DIRECTORY)
#     except OSError as e:
#         print(f"Error: {e.filename} - {e.strerror}.")
# else:
#     print("The directory does not exist")

# run_langest_commands = ["python", "ingest.py"]
# if DEVICE_TYPE == "cpu":
#     run_langest_commands.append("--device_type")
#     run_langest_commands.append(DEVICE_TYPE)

# result = subprocess.run(run_langest_commands, capture_output=True)
# if result.returncode != 0:
#     raise FileNotFoundError(
#         "No files were found inside SOURCE_DOCUMENTS, please put a starter file inside before starting the API!"
#     )

# load the vectorstore
# DB = Chroma(
#     persist_directory=PERSIST_DIRECTORY,
#     embedding_function=EMBEDDINGS,
#     client_settings=CHROMA_SETTINGS,
# )

router = APIRouter(
    prefix='/api/run-agent',
    tags=['agent']
)

res = None
answer = None


## Cite sources
def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text


def process_llm_response(llm_response):
    print(wrap_text_preserve_newlines(llm_response['result']))
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])


class Prompt(BaseModel):
    prompt: str


def load_model(device_type, model_id, model_basename=None, LOGGING=logging):
    """
    Select a model for text generation using the HuggingFace library.
    If you are running this for the first time, it will download a model for you.
    subsequent runs will use the model from the disk.

    Args:
        device_type (str): Type of device to use, e.g., "cuda" for GPU or "cpu" for CPU.
        model_id (str): Identifier of the model to load from HuggingFace's model hub.
        model_basename (str, optional): Basename of the model if using quantized models.
            Defaults to None.

    Returns:
        HuggingFacePipeline: A pipeline object for text generation using the loaded model.

    Raises:
        ValueError: If an unsupported model or device type is provided.
    """
    logging.info(f"Loading Model: {model_id}, on: {device_type}")
    logging.info("This action can take a few minutes!")

    if model_basename is not None:
        if ".gguf" in model_basename.lower():
            llm = load_quantized_model_gguf_ggml(model_id, model_basename, device_type, LOGGING)
            return llm
        elif ".ggml" in model_basename.lower():
            model, tokenizer = load_quantized_model_gguf_ggml(model_id, model_basename, device_type, LOGGING)
        elif ".awq" in model_basename.lower():
            model, tokenizer = load_quantized_model_awq(model_id, LOGGING)
        else:
            model, tokenizer = load_quantized_model_qptq(model_id, model_basename, device_type, LOGGING)
    else:
        model, tokenizer = load_full_model(model_id, model_basename, device_type, LOGGING)

    # Load configuration from the model to avoid warnings
    generation_config = GenerationConfig.from_pretrained(model_id)
    # see here for details:
    # https://huggingface.co/docs/transformers/
    # main_classes/text_generation#transformers.GenerationConfig.from_pretrained.returns

    # Create a pipeline for text generation
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=MAX_NEW_TOKENS,
        temperature=0.2,
        # top_p=0.95,
        repetition_penalty=1.15,
        generation_config=generation_config,
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)
    logging.info("Local LLM Loaded")

    return local_llm


def retrieval_qa_pipline(device_type, use_history, promptTemplate_type="llama"):
    """
    Initializes and returns a retrieval-based Question Answering (QA) pipeline.

    This function sets up a QA system that retrieves relevant information using embeddings
    from the HuggingFace library. It then answers questions based on the retrieved information.

    Parameters:
    - device_type (str): Specifies the type of device where the model will run, e.g., 'cpu', 'cuda', etc.
    - use_history (bool): Flag to determine whether to use chat history or not.

    Returns:
    - RetrievalQA: An initialized retrieval-based QA system.

    Notes:
    - The function uses embeddings from the HuggingFace library, either instruction-based or regular.
    - The Chroma class is used to load a vector store containing pre-computed embeddings.
    - The retriever fetches relevant documents or data based on a query.
    - The prompt and memory, obtained from the `get_prompt_template` function, might be used in the QA system.
    - The model is loaded onto the specified device using its ID and basename.
    - The QA system retrieves relevant documents using the retriever and then answers questions based on those documents.
    """

    """
    (1) Chooses an appropriate langchain library based on the enbedding model name.  Matching code is contained within ingest.py.
    
    (2) Provides additional arguments for instructor and BGE models to improve results, pursuant to the instructions contained on
    their respective huggingface repository, project page or github repository.
    """

    embeddings = get_embeddings(device_type)

    logging.info(f"Loaded embeddings from {EMBEDDING_MODEL_NAME}")

    # load the vectorstore
    # db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
     # init cache with TTL (expiration) policy and semantic distance threshhold

   # store user queries and LLM responses in the semantic cache
    print(REDIS_URL)
    rds = Redis.from_existing_index(
    embedding=embeddings,
    index_name=INDEX_NAME,
    schema=INDEX_SCHEMA,
    redis_url=REDIS_URL,
)
    # basic "top 4" vector search on a given query
    rds.similarity_search_with_score(query="Profit margins", k=4)
    # retriever=rds.as_retriever(search_type="similarity_distance_threshold",search_kwargs={"distance_threshold":0.5}),
    retriever = rds.as_retriever(search_type="mmr")
    # retriever = db.as_retriever()

    # get the prompt template and memory if set by the user.
    prompt, memory = get_prompt_template(promptTemplate_type=promptTemplate_type, history=use_history)

    print(memory)
    # load the llm pipeline
    # llm = load_model(device_type, model_id=MODEL_ID, model_basename=MODEL_BASENAME, LOGGING=logging)
    llm = ChatGroq(temperature=0, groq_api_key="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", model_name="llama2-70b-4096")
    if use_history:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # try other chains types as well. refine, map_reduce, map_rerank
            retriever=retriever,
            return_source_documents=True,  # verbose=True,
            callbacks=callback_manager,
            chain_type_kwargs={"prompt": prompt, "memory": memory},
        )
        
        print(qa.combine_documents_chain.memory)
        # qa = ConversationalRetrievalChain.from_llm(
        #     llm,
        #     chain_type="stuff",
        #     retriever=retriever,
        #     memory=memory,
        #     combine_docs_chain_kwargs={"prompt": prompt},
        #     return_source_documents=True,
        #     verbose=True,
        # )
        
        
    else:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # try other chains types as well. refine, map_reduce, map_rerank
            retriever=retriever,
            return_source_documents=True,  # verbose=True,
            callbacks=callback_manager,
            chain_type_kwargs={
                "prompt": prompt,
            },
        )

    return qa,llm



@router.get("/run_ingest", status_code=status.HTTP_200_OK)
async def run_ingest_route():
    global RETRIEVER
    global DB
    global QA
    try:
        if os.path.exists(PERSIST_DIRECTORY):
            try:
                shutil.rmtree(PERSIST_DIRECTORY)
            except OSError as e:
                print(f"Error: {e.filename} - {e.strerror}.")
        else:
            print("The directory does not exist")

        run_langest_commands = ["python", "routers/ingest.py"]
        if DEVICE_TYPE == "cpu":
            run_langest_commands.append("--device_type")
            run_langest_commands.append(DEVICE_TYPE)

        result = subprocess.run(run_langest_commands, capture_output=True)
        if result.returncode != 0:
            return "Script execution failed: {}".format(result.stderr.decode("utf-8")), 500
        # load the vectorstore
        # DB = Chroma(
        #     persist_directory=PERSIST_DIRECTORY,
        #     embedding_function=EMBEDDINGS,
        #     client_settings=CHROMA_SETTINGS,
        # )
        # RETRIEVER = DB.as_retriever()
        # prompt, memory = get_prompt_template(promptTemplate_type="llama", history=False)

        # QA = RetrievalQA.from_chain_type(
        #     llm=LLM,
        #     chain_type="stuff",
        #     retriever=RETRIEVER,
        #     return_source_documents=SHOW_SOURCES,
        #     chain_type_kwargs={
        #         "prompt": prompt,
        #     },
        # )
        return "Script executed successfully: {}".format(result.stdout.decode("utf-8")), 200
    except Exception as e:
        # return f"Error occurred: {str(e)}", 500
        raise HTTPException(status_code=500, detail='Something went wrong!.')


@router.post("/prompt_agent", status_code=status.HTTP_200_OK)
async def prompt_agent(request: Prompt):
    device_type ="mps"
    show_sources= False
    use_history= True
    model_type="llama"
    
    logging.info(f"Running on: {device_type}")
    logging.info(f"Display Source Documents set to: {show_sources}")
    logging.info(f"Use history set to: {use_history}")
    qa,llm = retrieval_qa_pipline(device_type, use_history, promptTemplate_type=model_type)
    
    rl = RouteLayer(encoder=encoder, routes=routes, llm=llm)
    user_prompt = request.prompt
    global answer
    
    
    if user_prompt:
        llmcache = SemanticCache(
        name="llmcache",
         ttl=360,
        redis_url=REDIS_URL
        )
        route = rl(user_prompt)
        print("route.name")
        print(route.name)
        if route.name == "appointment_inquiry" or  route.name== None:
            
            if response := llmcache.check(prompt=user_prompt,return_fields=["prompt", "response"]):
        # if False:
                print(response)
                res =response[0]["response"]
                answer= res
                type(res)
            else:
                print("Empty cache")
                res = qa(user_prompt)
                # res = qa.predict({"query": query})
                print("called llm and replying")
                process_llm_response(res)
                answer, docs = res["result"], res["source_documents"]

            # res = qa(query)
            # print("called llm and replying")
            # process_llm_response(res)
            # answer, docs = res["result"], res["source_documents"]

            print("from redis cache")
            llmcache.store(
            prompt=user_prompt,
            response=answer,
            metadata={}
                )
            
        elif route.name == "politics":
            print("inside politics now")
            answer = politics()
        elif route.name == "chitchat":
            answer = chitchat()
        elif route.name == "greetings":
            print("inside greetings now")
            answer = greetings()
        elif route.name == "done_task":
            answer = done_task()

# quickly check the cache with a slightly different prompt (before invoiking an LLM)


        # Print the result
        print("\n\n> Question:")
        print(user_prompt)
        print("\n> Answer:")
        print(answer)
        prompt_response_dict = {
                "Prompt": user_prompt,
                "Answer": answer,
            }
        # log_to_csv(user_prompt,answer)
        return prompt_response_dict, 200
    else:
        raise HTTPException(status_code=404, detail='No user prompt received')

  
    # global QA
    # global request_lock
    # rds = Redis.from_existing_index(
    #     embedding=EMBEDDINGS,
    #     index_name=INDEX_NAME,
    #     schema=INDEX_SCHEMA,
    #     redis_url=REDIS_URL,
    # )
    # rds.similarity_search_with_score(query="Profit margins", k=4)
    # RETRIEVER = rds.as_retriever(search_type="mmr")
    # # RETRIEVER = DB.as_retriever()

    # # LLM = load_model(device_type=DEVICE_TYPE, model_id=MODEL_ID, model_basename=MODEL_BASENAME, LOGGING=logging)
    # LLM = ChatGroq(temperature=0, groq_api_key="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", model_name="llama2-70b-4096")
    # prompt, memory = get_prompt_template(promptTemplate_type="llama", history=True)
    
    # print(memory)
    # # QA = retrieval_qa_pipline(device_type, use_history, promptTemplate_type=model_type)
    # QA = RetrievalQA.from_chain_type(
    #     llm=LLM,
    #     chain_type="stuff",
    #     retriever=RETRIEVER,
    #     return_source_documents=SHOW_SOURCES,
    #     callbacks=callback_manager,
    #     chain_type_kwargs={"prompt": prompt, "memory": memory},
    # )
    # print("memory")
    # print(QA.combine_documents_chain.memory)

    # user_prompt = request.prompt
    # if user_prompt:
    #     logging.info(f"Running on: {DEVICE_TYPE}")
    #     logging.info(f"Display Source Documents set to: {SHOW_SOURCES}")
    #     llmcache = SemanticCache(
    #         name="llmcache",
    #         ttl=360,
    #         redis_url=REDIS_URL
    #     )
    #     # Acquire the lock before processing the prompt
    #     with request_lock:
    #         # print(f'User Prompt: {user_prompt}')              
    #         # Get the answer from the chain
    #         test = llmcache.check(prompt=user_prompt, return_fields=["prompt", "response"])
    #         print("test: ")
    #         print(test)
    #         print("test: ")
    #         if response := llmcache.check(prompt=user_prompt, return_fields=["prompt", "response"]):
    #         # if False:
    #             print("from cace only")
    #             res = response[0]["response"]
    #             answer = res
    #         else:
    #             print("fresh from llm")
    #             res = QA(user_prompt)
    #             process_llm_response(res)
    #             answer, docs = res["result"], res["source_documents"]
    #             # for document in docs:
    #             #     prompt_response_dict["Sources"].append(
    #             #         (os.path.basename(str(document.metadata["source"])), str(document.page_content))
    #             #     )
    #             answer = wrap_text_preserve_newlines(res['result'])
    #             print(answer)
               

    #         prompt_response_dict = {
    #             "Prompt": user_prompt,
    #             "Answer": answer,
    #         }
    #         llmcache.store(
    #                 prompt=user_prompt,
    #                 response=answer,
    #                 metadata={}
    #             )



    #     # Uncomment for Debugging only
    #     # prompt_response_dict["Sources"] = []
       
    #     # return StreamingResponse(prompt_response_dict["Answer"], media_type="text/event-stream")
    #     logging.info("Save chat history to csv")
    
    #     return prompt_response_dict, 200
    # else:
    #     raise HTTPException(status_code=404, detail='No user prompt received')
