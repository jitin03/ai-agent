import os

# from dotenv import load_dotenv
# from chromadb.config import Settings
import enum
# https://python.langchain.com/en/latest/modules/indexes/document_loaders/examples/excel.html?highlight=xlsx#microsoft-excel
from langchain.document_loaders import CSVLoader, PDFMinerLoader, TextLoader, UnstructuredExcelLoader, Docx2txtLoader
from langchain.document_loaders import UnstructuredFileLoader, UnstructuredMarkdownLoader
from langchain.document_loaders import UnstructuredHTMLLoader
from pydantic import BaseModel,Field
from typing import Optional
from typing import List
from kor.nodes import Object, Text, Number
from kor import extract_from_documents, from_pydantic, create_extraction_chain
# load_dotenv()
ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

print(ROOT_DIRECTORY)
# Define the folder for storing database
SOURCE_DIRECTORY = f"{ROOT_DIRECTORY}/SOURCE_DOCUMENTS"
# SOURCE_DIRECTORY = "SOURCE_DOCUMENTS"

PERSIST_DIRECTORY = f"{ROOT_DIRECTORY}/DB"

MODELS_PATH =f"{ROOT_DIRECTORY}/models"

# Can be changed to a specific number
INGEST_THREADS = os.cpu_count() or 8

# Define the Chroma settings
# CHROMA_SETTINGS = Settings(
#     anonymized_telemetry=False,
#     is_persistent=True,
# )

# Context Window and Max New Tokens
CONTEXT_WINDOW_SIZE = 2098
MAX_NEW_TOKENS = CONTEXT_WINDOW_SIZE  # int(CONTEXT_WINDOW_SIZE/4)

#### If you get a "not enough space in the buffer" error, you should reduce the values below, start with half of the original values and keep halving the value until the error stops appearing

N_GPU_LAYERS = 100  # Llama-2-70B has 83 layers
N_BATCH = 512

### From experimenting with the Llama-2-7B-Chat-GGML model on 8GB VRAM, these values work:
# N_GPU_LAYERS = 20
# N_BATCH = 512


# https://python.langchain.com/en/latest/_modules/langchain/document_loaders/excel.html#UnstructuredExcelLoader
DOCUMENT_MAP = {
    ".html": UnstructuredHTMLLoader,
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
    ".py": TextLoader,
    # ".pdf": PDFMinerLoader,
    ".pdf": UnstructuredFileLoader,
    ".csv": CSVLoader,
    ".xls": UnstructuredExcelLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
}

# Default Instructor Model
EMBEDDING_MODEL_NAME = "hkunlp/instructor-large"  # Uses 1.5 GB of VRAM (High Accuracy with lower VRAM usage)
# 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
####
#### OTHER EMBEDDING MODEL OPTIONS
####

# EMBEDDING_MODEL_NAME = "hkunlp/instructor-xl" # Uses 5 GB of VRAM (Most Accurate of all models)
# EMBEDDING_MODEL_NAME = "intfloat/e5-large-v2" # Uses 1.5 GB of VRAM (A little less accurate than instructor-large)
# EMBEDDING_MODEL_NAME = "intfloat/e5-base-v2" # Uses 0.5 GB of VRAM (A good model for lower VRAM GPUs)
# EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" # Uses 0.2 GB of VRAM (Less accurate but fastest - only requires 150mb of vram)

####
#### MULTILINGUAL EMBEDDING MODELS
####

# EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large" # Uses 2.5 GB of VRAM
# EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base" # Uses 1.2 GB of VRAM


#### SELECT AN OPEN SOURCE LLM (LARGE LANGUAGE MODEL)
# Select the Model ID and model_basename
# load the LLM for generating Natural Language responses

#### GPU VRAM Memory required for LLM Models (ONLY) by Billion Parameter value (B Model)
#### Does not include VRAM used by Embedding Models - which use an additional 2GB-7GB of VRAM depending on the model.
####
#### (B Model)   (float32)    (float16)    (GPTQ 8bit)         (GPTQ 4bit)
####    7b         28 GB        14 GB       7 GB - 9 GB        3.5 GB - 5 GB
####    13b        52 GB        26 GB       13 GB - 15 GB      6.5 GB - 8 GB
####    32b        130 GB       65 GB       32.5 GB - 35 GB    16.25 GB - 19 GB
####    65b        260.8 GB     130.4 GB    65.2 GB - 67 GB    32.6 GB -  - 35 GB

# MODEL_ID = "TheBloke/Llama-2-7B-Chat-GGML"
# MODEL_BASENAME = "llama-2-7b-chat.ggmlv3.q4_0.bin"

####
#### (FOR GGUF MODELS)
####

# MODEL_ID = "TheBloke/Llama-2-13b-Chat-GGUF"
# MODEL_BASENAME = "llama-2-13b-chat.Q4_K_M.gguf"

# MODEL_ID = "TheBloke/Llama-2-7b-Chat-GGUF"
MODEL_ID = "tiiuae/falcon-7b-instruct"
# MODEL_BASENAME = "llama-2-7b-chat.Q4_K_M.gguf"
MODEL_BASENAME = None



# MODEL_ID = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
# MODEL_BASENAME = "mistral-7b-instruct-v0.1.Q8_0.gguf"

# MODEL_ID = "TheBloke/Llama-2-70b-Chat-GGUF"
# MODEL_BASENAME = "llama-2-70b-chat.Q4_K_M.gguf"

####
#### (FOR HF MODELS)
####

# MODEL_ID = "NousResearch/Llama-2-7b-chat-hf"
# MODEL_BASENAME = None
# MODEL_ID = "TheBloke/vicuna-7B-1.1-HF"
# MODEL_BASENAME = None
# MODEL_ID = "TheBloke/Wizard-Vicuna-7B-Uncensored-HF"
# MODEL_ID = "TheBloke/guanaco-7B-HF"
# MODEL_ID = 'NousResearch/Nous-Hermes-13b' # Requires ~ 23GB VRAM. Using STransformers
# alongside will 100% create OOM on 24GB cards.
# llm = load_model(device_type, model_id=model_id)

####
#### (FOR GPTQ QUANTIZED) Select a llm model based on your GPU and VRAM GB. Does not include Embedding Models VRAM usage.
####

##### 48GB VRAM Graphics Cards (RTX 6000, RTX A6000 and other 48GB VRAM GPUs) #####

### 65b GPTQ LLM Models for 48GB GPUs (*** With best embedding model: hkunlp/instructor-xl ***)
# MODEL_ID = "TheBloke/guanaco-65B-GPTQ"
# MODEL_BASENAME = "model.safetensors"
# MODEL_ID = "TheBloke/Airoboros-65B-GPT4-2.0-GPTQ"
# MODEL_BASENAME = "model.safetensors"
# MODEL_ID = "TheBloke/gpt4-alpaca-lora_mlp-65B-GPTQ"
# MODEL_BASENAME = "model.safetensors"
# MODEL_ID = "TheBloke/Upstage-Llama1-65B-Instruct-GPTQ"
# MODEL_BASENAME = "model.safetensors"

##### 24GB VRAM Graphics Cards (RTX 3090 - RTX 4090 (35% Faster) - RTX A5000 - RTX A5500) #####

### 13b GPTQ Models for 24GB GPUs (*** With best embedding model: hkunlp/instructor-xl ***)
# MODEL_ID = "TheBloke/Wizard-Vicuna-13B-Uncensored-GPTQ"
# MODEL_BASENAME = "Wizard-Vicuna-13B-Uncensored-GPTQ-4bit-128g.compat.no-act-order.safetensors"
# MODEL_ID = "TheBloke/vicuna-13B-v1.5-GPTQ"
# MODEL_BASENAME = "model.safetensors"
# MODEL_ID = "TheBloke/Nous-Hermes-13B-GPTQ"
# MODEL_BASENAME = "nous-hermes-13b-GPTQ-4bit-128g.no-act.order"
# MODEL_ID = "TheBloke/WizardLM-13B-V1.2-GPTQ"
# MODEL_BASENAME = "gptq_model-4bit-128g.safetensors

### 30b GPTQ Models for 24GB GPUs (*** Requires using intfloat/e5-base-v2 instead of hkunlp/instructor-large as embedding model ***)
# MODEL_ID = "TheBloke/Wizard-Vicuna-30B-Uncensored-GPTQ"
# MODEL_BASENAME = "Wizard-Vicuna-30B-Uncensored-GPTQ-4bit--1g.act.order.safetensors"
# MODEL_ID = "TheBloke/WizardLM-30B-Uncensored-GPTQ"
# MODEL_BASENAME = "WizardLM-30B-Uncensored-GPTQ-4bit.act-order.safetensors"

##### 8-10GB VRAM Graphics Cards (RTX 3080 - RTX 3080 Ti - RTX 3070 Ti - 3060 Ti - RTX 2000 Series, Quadro RTX 4000, 5000, 6000) #####
### (*** Requires using intfloat/e5-small-v2 instead of hkunlp/instructor-large as embedding model ***)

### 7b GPTQ Models for 8GB GPUs
# MODEL_ID = "TheBloke/Wizard-Vicuna-7B-Uncensored-GPTQ"
# MODEL_BASENAME = "Wizard-Vicuna-7B-Uncensored-GPTQ-4bit-128g.no-act.order.safetensors"
# MODEL_ID = "TheBloke/WizardLM-7B-uncensored-GPTQ"
# MODEL_BASENAME = "WizardLM-7B-uncensored-GPTQ-4bit-128g.compat.no-act-order.safetensors"
# MODEL_ID = "TheBloke/wizardLM-7B-GPTQ"
# MODEL_BASENAME = "wizardLM-7B-GPTQ-4bit.compat.no-act-order.safetensors"

####
#### (FOR GGML) (Quantized cpu+gpu+mps) models - check if they support llama.cpp
####

# MODEL_ID = "TheBloke/wizard-vicuna-13B-GGML"
# MODEL_BASENAME = "wizard-vicuna-13B.ggmlv3.q4_0.bin"
# MODEL_BASENAME = "wizard-vicuna-13B.ggmlv3.q6_K.bin"
# MODEL_BASENAME = "wizard-vicuna-13B.ggmlv3.q2_K.bin"
# MODEL_ID = "TheBloke/orca_mini_3B-GGML"
# MODEL_BASENAME = "orca-mini-3b.ggmlv3.q4_0.bin"

####
#### (FOR AWQ QUANTIZED) Select a llm model based on your GPU and VRAM GB. Does not include Embedding Models VRAM usage.
### (*** MODEL_BASENAME is not actually used but have to contain .awq so the correct model loading is used ***)
### (*** Compute capability 7.5 (sm75) and CUDA Toolkit 11.8+ are required ***)
####
# MODEL_ID = "TheBloke/Llama-2-7B-Chat-AWQ"
# MODEL_BASENAME = "model.safetensors.awq"
PERSONAL_INFO_SCHEMA = Object(
    id="personal_info",
    description="Personal information about a given person.",
    attributes=[
        Text(
            id="name",
            description="The  name of the person",
            examples=[("My name is Jitin Doriya", "Jitin Doriya"),("Myself is Jitin Doriya", "Jitin Doriya"),("I am Jitin Doriya", "Jitin Doriya")],
        ),
        Text(
            id="contact_no",
            description="The contact number of the person",
            examples=[("My phone number is 9780032269", "9780032269")],
        ),
        Text(
            id="location",
            description="The location of the person.",
            examples=[("I am from Jagatpura", "Jagatpura"), ("calling from Indira Gandhi nagar", "Indira Gandhi nagar")],
        ),
    ],
    examples=[
        (
            "I am jitin doriya calling from jagatpura  and phone no is 9780032269",
            [
                {"name": "John", "contact_no": "9780032269", "location": "jagatpura"},
                
            ],
        )
    ],
    many=True,
)


class requirements(enum.Enum):
    name = "name"
    day = "day"
    time = "time"
    phone_number = "phone_number"

class extraction(BaseModel):
    name : Optional[List[str]] = Field(
        default=None, description=" The name that the patient or user will provide",
        examples=[("My name is Om Tarkunde","Om Tarkunde"),
        (" Om Tarkunde","Om Tarkunde"),
        (" name Aryaman Sinha","Aryaman Sinha"),
        ]
    )

    day : Optional[List[str]] = Field(
        default=None, description=" The day that the patient or user will provide",
        examples=[("sunday and 8.30pm","sunday"),
                  ("Sunday is the day","Sunday"),
                  ("I choose Monday","Monday")]
    )

    time : Optional[List[str]] = Field(
        default=None, description=" The time that the patient or user will provide",
        examples=[("sunday and 8.30pm","8:30pm"),
                  ("time I prefer is 5:00 am","5:00 am")]
    )
    phone_number: Optional[List[int]] = Field(
    default=None, description=" The phone number that the patient or user will provide",
    examples=[
        ("My contact is 4561237890", "4561237890"),
        ("4561237890", "4561237890"),
        ("number 1234567888", "1234567888"),
        ("my phone number is 7020588856", "7020588856"),
        ("Great, I'd be happy to assist you with booking an appointment with Dr. Jitin. Can you please tell me your contact information? yeah sure 9075898463.Great, I'd be happy to assist you with booking an appointment with Dr. Jitin. Can you please tell me your preferred appointment day and time? Keep in mind that Dr. Jitin's available days and times are Monday, Wednesday from 7pm to 9pm, and Saturday, Sunday from 1pm to 8pm.","9075898463")]
    )