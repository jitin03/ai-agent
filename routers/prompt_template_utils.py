"""
This file implements prompt template for llama based models. 
Modify the prompt template based on the model you select. 
This seems to have significant impact on the output of the LLM.
"""

from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# this is specific to Llama-2.

# system_prompt = """You are a AI assistant at a Clinic, you will use the provided context to answer user questions.
# Read the given context before answering questions and think step by step. If you can not answer a user question based on 
# the provided context, inform the user. Do not use any other information for answering user."""   
# Assist a patient in booking an appointment for a doctor by asking mandaroty details their name, contact information, and preferred appointment day and time before booking the appointment and once you have name, contact and data and time complete the appointment for user.

system_prompt="""I want you to act as a Assistant at clinic appointment scheduler. 
You must ask patient details i.e. name, phone and date and time before booking the appointment and once you have name, contact and data and time complete the appointment for patient.
Ask each question sequentially and keep your responses concise. For Example :
Question 1: Can you please tell me your preffered day and time for appointment?.
Question 2: Can you please tell your name?
Question 3: Can you please tell me your contact number?.

Read the given context before answering questions and think step by step.
You must confirm the appointment details with patient. If you can not answer a user question based on 
the provided context, inform the user. Do not use any other information for answering user."""

hotel_system_prompt="""
I want you to act as a Virtual agent at a cafe managing customer calls who talks very professional responsd very concise, answering queries about booking tables for fine dining experiences.
Read the given context before answering questions and think step by step.
Ask each question one by one only and keep your responses short. For Example :
Question 1: Whose name should I reserve a table ?
Question 2: if you don't mind , Please can we know is there any special occasion so that we can help you more ?.
Question 3: Can you please tell me your contact number?.
You must confirm the booking details with customer. If you can not answer a user question based on 
the provided context, inform the user. Do not use any other information for answering user."""
# At the end of table booking be knowledgeable about the restaurant's offerings and provide excellent customer service.
def get_prompt_template(system_prompt=system_prompt, promptTemplate_type=None, history=[]):
 
    if promptTemplate_type == "llama":
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        SYSTEM_PROMPT = B_SYS + hotel_system_prompt + E_SYS
        if history:

            chat_history_str = "\n".join([f"{message['role']}: {message['content']}" for message in history])
            instruction = """
            Context: {history} \n {context}
            ### Input: {question}
            ### Response:"""
            final_template = instruction.replace("{history}",chat_history_str)
            prompt_template = B_INST + SYSTEM_PROMPT + final_template + E_INST
            prompt = PromptTemplate(input_variables=["history", "context", "question"], template=prompt_template)
        else:
            instruction = """
            Context: {context}
            User: {question}"""

            prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
            prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)
    elif promptTemplate_type == "mistral":
        B_INST, E_INST = "<s>[INST] ", " [/INST]"
        if history:
            prompt_template = (
                B_INST
                + system_prompt
                + """
    
            Context: {history} \n {context}
            User: {question}"""
                + E_INST
            )
            prompt = PromptTemplate(input_variables=["history", "context", "question"], template=prompt_template)
        else:
            prompt_template = (
                B_INST
                + system_prompt
                + """
            
            Context: {context}
            User: {question}"""
                + E_INST
            )
            prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)
    else:
        # change this based on the model you have selected.
        if history:
            prompt_template = (
                system_prompt
                + """
    
            Context: {history} \n {context}
            User: {question}
            Answer:"""
            )
            prompt = PromptTemplate(input_variables=["history", "context", "question"], template=prompt_template)
        else:
            prompt_template = (
                system_prompt
                + """
            
            Context: {context}
            User: {question}
            Answer:"""
            )
            prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)
    memory = ConversationBufferMemory(
        memory_key="history",
        human_prefix="### Input",
        ai_prefix="### Response",
        input_key="question",
        output_key="output_text",
        return_messages=False,
    )
    # memory = ConversationBufferMemory(input_key="question", memory_key="history")

    return (
        prompt,
        memory,
    )
