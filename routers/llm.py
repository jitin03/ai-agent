
from fastapi import APIRouter, Depends, HTTPException, Path
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import CTransformers
from langchain.vectorstores import Chroma
from starlette import status
import sys
# from run_localGPT import load_model
router = APIRouter(
    prefix='/run-agent',
    tags=['agent']
)


@router.get("", status_code=status.HTTP_200_OK)
async def run_agent():
    loader=DirectoryLoader('./data/',
                       glob="*.txt",
                       loader_cls=TextLoader)

    documents=loader.load() 
    print(documents)
    #***Step 2: Split Text into Chunks***

    text_splitter=RecursiveCharacterTextSplitter(
                                                chunk_size=500,
                                                chunk_overlap=50)


    text_chunks=text_splitter.split_documents(documents)

    print(len(text_chunks))
    #**Step 3: Load the Embedding Model***


    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device':'cpu'})


    #**Step 4: Convert the Text Chunks into Embeddings and Create a FAISS Vector Store***
    # vector_store=FAISS.from_documents(text_chunks, embeddings)


    # Embed and store the texts
    # Supplying a persist_directory will store the embeddings on disk
    persist_directory = 'db'

    ## here we are using OpenAI embeddings but in future we will swap out to local embeddings
    # embedding = OpenAIEmbeddings()
    texts = text_splitter.split_documents(documents)
    vectordb = Chroma.from_documents(documents=texts, 
                                    embedding=embeddings,
                                    persist_directory=persist_directory)


    ##**Step 5: Find the Top 3 Answers for the Query***

    query="YOLOv7 outperforms which models"
    docs = vectordb.similarity_search(query)

    #print(docs)
    llm=CTransformers(model="./models/llama-2-7b-chat.ggmlv3.q4_0.bin",
                    model_type="llama",
                    config={'max_new_tokens':128,
                            'temperature':0.01})


    # template="""Use the following pieces of information to answer the user's question.
    # If you dont know the answer just say you know, don't try to make up an answer.

    # Context:{context}
    # Question:{question}

    # Only return the helpful answer below and nothing else
    # Helpful answer
    # """
    # template= """You are a helpful, respectful and Doctor assistant. And doctor assistant fix the appointment and always answer as helpfully as possible using the context text provided. Your answers should only answer the question once and not have any text after the answer is done.

    # If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.

    # Speaker:/n/n {context}/n

    # Doctor assistant: {question}"""

    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    DEFAULT_SYSTEM_PROMPT="""\
    Your name is AI Agent. You are a virtual assistant for a business.
            ## Information on the business.
            ## Personality and Job Duties
            Here you can describe personality traits and job duties in plain language.
            ## Greeting Rules
            Greet the user and thank them for showing interest in the business. 
            Prefix the greeting with a 'good morning', 'good afternoon', or a 'good evening' depending on the time of day."""

    instruction = "Doctor assistant: \n\n {text}"

    SYSTEM_PROMPT = B_SYS + DEFAULT_SYSTEM_PROMPT + E_SYS

    template = B_INST + SYSTEM_PROMPT + instruction + E_INST

    print(template)
    prompt = PromptTemplate(template=template, input_variables=["text"])
    # qa_prompt=PromptTemplate(template=template, input_variables=[ 'text'])

    #start=timeit.default_timer()

    chain = RetrievalQA.from_chain_type(llm=llm,
                                    chain_type='stuff',
                                    retriever=vectordb.as_retriever(search_kwargs={'k': 2}),
                                    return_source_documents=True,
                                    # chain_type_kwargs={'prompt': qa_prompt}
                                    )
    response=chain( "Hello, Good morning?")

    #end=timeit.default_timer()
    #print(f"Here is the complete Response: {response}")
    return {"data":response['result']}



