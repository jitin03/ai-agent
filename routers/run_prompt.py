from fastapi import APIRouter, Depends, HTTPException, Path
import logging
import asyncio
import pandas as pd
import os
import json
import redis
import shutil
import subprocess
import argparse
from langchain_groq import ChatGroq
from pydantic import BaseModel
import torch
from semantic_router.encoders import HuggingFaceEncoder
from .semantic_routers import routes
from .semanic_router_response import appointment_inquiry,politics,chitchat,greetings,done_task,appointment_form,end_conversation
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
from typing import List
from .constants import (
    # CHROMA_SETTINGS,
    EMBEDDING_MODEL_NAME, PERSIST_DIRECTORY, MODEL_ID, MODEL_BASENAME,
    MAX_NEW_TOKENS,PERSONAL_INFO_SCHEMA)
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
from kor import extract_from_documents, from_pydantic, create_extraction_chain
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
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

ROLE_CLASS_MAP = {
    "AI": AIMessage,
    "Human": HumanMessage,
    "system": SystemMessage
    
}

request_lock = Lock()
r = redis.Redis(host='localhost', port=6379, db=0)
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

router = APIRouter(
    prefix='/api/run-agent',
    tags=['agent']
)

res = None
answer = None
index =0
# conversation_id="123"
conversation_history=[]
conversation_index = {}
appointment_form_index=0
INTIAL_CONVERSTATION = {"conversation": {"role": "system", "content": "You are a helpful assistant."}}

class Message(BaseModel):
    role: str
    content: str

class Conversation(BaseModel):
    conversation: List[Message]
    
def create_messages(conversation):
    messages = []
    for message in conversation:
        print(message)
        # Get the role of the message
        role = message["role"]
        
        # Get the content of the message
        content = message["content"]
        
        # Get the corresponding message class from the ROLE_CLASS_MAP based on the role
        message_class = ROLE_CLASS_MAP[role]
        
        # Instantiate the message class with the content and append it to the list of messages
        message_object = message_class(content=content)
        messages.append(message_object)
    
    return messages
    # return [ROLE_CLASS_MAP[message.role](content=message.content) for message in conversation]

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
    print("prompt")
    print(prompt)
    logging.info(prompt)
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


@router.get("/get_personal_info/{conversation_id}",status_code=status.HTTP_200_OK)
async def get_personal_info(conversation_id: str):
    device_type ="mps"
    show_sources= False
    use_history= True
    model_type="llama"
    reviews=pd.read_csv('./local_chat_history/qwerty_qa_log.csv')
    reviews["question"]=reviews["question"].drop_duplicates()
    reviews["question"].dropna()
    concatenated_string = ' '.join(reviews['question'].dropna())
    qa,llm = retrieval_qa_pipline(device_type, use_history=conversation_history, promptTemplate_type=model_type)
    chain = create_extraction_chain(llm, PERSONAL_INFO_SCHEMA,input_formatter="text_prefix")
    response = chain.run(concatenated_string)["data"]
    return {"prompts":response}
    

@router.post("/prompt_agent/{conversation_id}", status_code=status.HTTP_200_OK)
async def prompt_agent(conversation_id:str,request: Prompt):
    global conversation_history
    global conversation_index
    global answer
    global appointment_form_index
    device_type ="mps"
    show_sources= False
    use_history= True
    model_type="llama"
    
    logging.info(f"Running on: {device_type}")
    logging.info(f"Display Source Documents set to: {show_sources}")
    logging.info(f"Use history set to: {use_history}")
   
    user_prompt = request.prompt
    if user_prompt:
    
        existing_conversation_json = r.get(conversation_id)
        if existing_conversation_json:
            existing_conversation = json.loads(existing_conversation_json)
            print("inside redis")
            print(existing_conversation)
            print("inside redis")
        else:
            print("no index for history")
            existing_conversation={}
            existing_conversation.setdefault("conversation_history", [])
    
    
        existing_conversation["conversation_history"].append({"role": "system", "content": "You are a helpful assistant."})
        conversation_history.append({'role':'Human','content':user_prompt})
        qa,llm = retrieval_qa_pipline(device_type, use_history=existing_conversation["conversation_history"], promptTemplate_type=model_type)
    
        rl = RouteLayer(encoder=encoder, routes=routes, llm=llm)
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
        elif route.name=="end_conversation":
            answer = end_conversation()
        elif route.name =="appointment_form":
            
            qs = appointment_form()
            # Extracting the content of the last element in the history
            

            # Check if the last content contains the questions sequentially
            if existing_conversation_json:
                print("heroes")
                print(existing_conversation["conversation_history"][-2]['content'])
                print(existing_conversation["conversation_history"][-1]['content'])
                print("heroes")
                # last_content = existing_conversation["conversation_history"][-1]['content']
                last_content = [msg['content'] for msg in reversed(existing_conversation["conversation_history"]) if msg['role'] == "AI"][0]
                if existing_conversation["appointment_form_index"]==2:
                    answer = done_task()
                else:
                    # Find the index of the last question found in last_content
                    last_index = -1
                    for i, question in enumerate(qs):
                        if question in last_content:
                            last_index = i

                    # Print the next question if found
                    print(last_index + 1)
                    print(last_index)
                    if last_index + 1 < len(qs):
                        print("qs[last_index + 1]")
                        print(qs[last_index + 1])
                        answer=qs[last_index + 1]
                        existing_conversation["appointment_form_index"] +=1
                    elif last_index+1 == len(qs):
                        answer = done_task()
                    
            else:   
                    last_content = conversation_history[-1]['content']
                    index = 0
                    for question in qs:
                        if question in last_content:
                            index += 1
                            appointment_form_index=index
                        else:
                            break
                    # If the last content does not contain the questions sequentially, print the corresponding question
                    if index < len(qs):
                        print(qs[index])
                        answer = qs[index]
        elif route.name == "done_task":
            answer = done_task()
        # Print the result
        print("\n\n> Question:")
        print(user_prompt)
        print("\n> Answer:")
        print(answer)
        print("\n> history:")
        conversation_history.append({'role':'AI','content':answer})
        # conversation_index["conversation_history"]=conversation_history
        # conversation_index["appointment_form_index"]=appointment_form_index
        existing_conversation["conversation_history"]=conversation_history
        print(json.dumps(conversation_history))
        r.set(conversation_id, json.dumps(existing_conversation))
        print(json.dumps(conversation_index))
        prompt_response_dict = {
                "Prompt": user_prompt,
                "Answer": answer,
                "history":existing_conversation
                
            }
        log_to_csv(user_prompt,answer,conversation_id)
        return prompt_response_dict, 200
    else:
        raise HTTPException(status_code=404, detail='No user prompt received')

  