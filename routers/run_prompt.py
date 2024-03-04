from fastapi import APIRouter, Depends, HTTPException, Path
import logging
import os
import shutil
import subprocess
import argparse
from pydantic import BaseModel
import torch
from flask import Flask, jsonify, request
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.callbacks.manager import CallbackManager
# from langchain.embeddings import HuggingFaceEmbeddings
from .run_localGPT import load_model, retrieval_qa_pipline
from .prompt_template_utils import get_prompt_template
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler  # for streaming response
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.vectorstores import Chroma
from werkzeug.utils import secure_filename

from .constants import CHROMA_SETTINGS, EMBEDDING_MODEL_NAME, PERSIST_DIRECTORY, MODEL_ID, MODEL_BASENAME

# API queue addition
from threading import Lock
from starlette import status
import sys
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
DB = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=EMBEDDINGS,
    client_settings=CHROMA_SETTINGS,
)

RETRIEVER = DB.as_retriever()

LLM = load_model(device_type=DEVICE_TYPE, model_id=MODEL_ID, model_basename=MODEL_BASENAME,LOGGING=logging)
prompt, memory = get_prompt_template(promptTemplate_type="llama", history=True)

# QA = retrieval_qa_pipline(device_type, use_history, promptTemplate_type=model_type)
QA = RetrievalQA.from_chain_type(
    llm=LLM,
    chain_type="stuff",
    retriever=RETRIEVER,
    return_source_documents=SHOW_SOURCES,
    callbacks=callback_manager,
    chain_type_kwargs={"prompt": prompt, "memory": memory},
)

router = APIRouter(
    prefix='/api/run-agent',
    tags=['agent']
)

class Prompt(BaseModel):
    prompt: str


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
        DB = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=EMBEDDINGS,
            client_settings=CHROMA_SETTINGS,
        )
        RETRIEVER = DB.as_retriever()
        prompt, memory = get_prompt_template(promptTemplate_type="llama", history=False)

        QA = RetrievalQA.from_chain_type(
            llm=LLM,
            chain_type="stuff",
            retriever=RETRIEVER,
            return_source_documents=SHOW_SOURCES,
            chain_type_kwargs={
                "prompt": prompt,
            },
        )
        return "Script executed successfully: {}".format(result.stdout.decode("utf-8")), 200
    except Exception as e:
        # return f"Error occurred: {str(e)}", 500
        raise HTTPException(status_code=500, detail='Something went wrong!.')



@router.post("/prompt_agent",status_code=status.HTTP_200_OK)
async def prompt_agent(request: Prompt):
    print(type(request))
    global QA
    global request_lock
    user_prompt = request.prompt
    if user_prompt:
        logging.info(f"Running on: {DEVICE_TYPE}")
        logging.info(f"Display Source Documents set to: {SHOW_SOURCES}")
        
        # Acquire the lock before processing the prompt
        with request_lock:
            # print(f'User Prompt: {user_prompt}')              
            # Get the answer from the chain
            res = QA(user_prompt)
            answer, docs = res["result"], res["source_documents"]
            print(docs)
            prompt_response_dict = {
                "Prompt": user_prompt,
                "Answer": answer,
            }

            prompt_response_dict["Sources"] = []
            for document in docs:
                prompt_response_dict["Sources"].append(
                    (os.path.basename(str(document.metadata["source"])), str(document.page_content))
                )

        return prompt_response_dict, 200
    else:
        raise HTTPException(status_code=404, detail='No user prompt received')