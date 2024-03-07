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

from .utils import log_to_csv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.callbacks.manager import CallbackManager
# from langchain.embeddings import HuggingFaceEmbeddings
from .run_localGPT import load_model, retrieval_qa_pipline
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
    EMBEDDING_MODEL_NAME, PERSIST_DIRECTORY, MODEL_ID, MODEL_BASENAME)
from fastapi.responses import StreamingResponse
# API queue addition
from threading import Lock
from starlette import status
import sys
import textwrap
from langchain.vectorstores.redis import Redis
from redisvl.extensions.llmcache import SemanticCache
from .rag_redis.config import (
    INDEX_NAME, INDEX_SCHEMA, REDIS_URL
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


async def send_message(content: str) -> AsyncIterable[str]:
    callback = AsyncIteratorCallbackHandler()
    model = ChatOpenAI(
        streaming=True,
        verbose=True,
        callbacks=[callback],
    )

    task = asyncio.create_task(
        model.agenerate(messages=[[HumanMessage(content=content)]])
    )

    try:
        async for token in callback.aiter():
            yield token
    except Exception as e:
        print(f"Caught exception: {e}")
    finally:
        callback.done.set()

    await task


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
    print(type(request))
    global QA
    global request_lock
    rds = Redis.from_existing_index(
        embedding=EMBEDDINGS,
        index_name=INDEX_NAME,
        schema=INDEX_SCHEMA,
        redis_url=REDIS_URL,
    )
    rds.similarity_search_with_score(query="Profit margins", k=4)
    RETRIEVER = rds.as_retriever(search_type="mmr")
    # RETRIEVER = DB.as_retriever()

    LLM = load_model(device_type=DEVICE_TYPE, model_id=MODEL_ID, model_basename=MODEL_BASENAME, LOGGING=logging)
    # LLM = ChatGroq(temperature=0, groq_api_key="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", model_name="llama2-70b-4096")
    prompt, memory = get_prompt_template(promptTemplate_type="llama", history=True)
    print("memory")
    print(memory)
    # QA = retrieval_qa_pipline(device_type, use_history, promptTemplate_type=model_type)
    QA = RetrievalQA.from_chain_type(
        llm=LLM,
        chain_type="stuff",
        retriever=RETRIEVER,
        return_source_documents=SHOW_SOURCES,
        callbacks=callback_manager,
        chain_type_kwargs={"prompt": prompt, "memory": memory},
    )

    user_prompt = request.prompt
    if user_prompt:
        logging.info(f"Running on: {DEVICE_TYPE}")
        logging.info(f"Display Source Documents set to: {SHOW_SOURCES}")
        llmcache = SemanticCache(
            name="llmcache",
            ttl=360,
            redis_url=REDIS_URL
        )
        # Acquire the lock before processing the prompt
        with request_lock:
            # print(f'User Prompt: {user_prompt}')              
            # Get the answer from the chain
            test = llmcache.check(prompt=user_prompt, return_fields=["prompt", "response"])
            print("test: ")
            print(test)
            print("test: ")
            if response := llmcache.check(prompt=user_prompt, return_fields=["prompt", "response"]):
                print("from cace only")
                res = response[0]["response"]
                answer = res
            else:
                print("fresh from llm")
                res = QA(user_prompt)
                process_llm_response(res)
                answer, docs = res["result"], res["source_documents"]
                answer = wrap_text_preserve_newlines(res['result'])
                print(answer)
                llmcache.store(
                    prompt=user_prompt,
                    response=answer,
                    metadata={}
                )

            prompt_response_dict = {
                "Prompt": user_prompt,
                "Answer": answer,
            }




        # Uncomment for Debugging only
        # prompt_response_dict["Sources"] = []
        # for document in docs:
        #     prompt_response_dict["Sources"].append(
        #         (os.path.basename(str(document.metadata["source"])), str(document.page_content))
        #     )
        # return StreamingResponse(prompt_response_dict["Answer"], media_type="text/event-stream")
        logging.info("Save chat history to csv")
        log_to_csv(user_prompt,answer)
        return prompt_response_dict, 200
    else:
        raise HTTPException(status_code=404, detail='No user prompt received')
