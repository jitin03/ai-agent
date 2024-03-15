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

 You must ask patient details i.e. name, phone and date and time before booking the appointment and once you have name, contact and data and time complete the appointment for user.
Ask each question sequentially and keep your responses concise. For Example :
Question 1: Can you please tell your name?
Question 2: Can you please tell me your contact number?.
Question 2: Can you please tell me your preffered day and time for appointment?.
Read the given context before answering questions and think step by step. If you can not answer a user question based on 
the provided context, inform the user. Do not use any other information for answering user."""

def get_prompt_template(system_prompt=system_prompt, promptTemplate_type=None, history=[]):
 
    if promptTemplate_type == "llama":
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        SYSTEM_PROMPT = B_SYS + system_prompt + E_SYS
        if history:
            # prompt_template = """
            #     ### Instruction: You're a virtual assistant for a Clinic. You must ask patient details i.e. name, phone and date and time before booking the appointment and once you have name, contact and data and time complete the appointment for user.
            #     Question 1: Can you please tell your name?
            #     Question 2: Can you please tell me your contact number?
                
            #     Always keep your reply short only.
                
            #     Use the chat history \n
            #     Chat History:\n\n{history} \n
            #     and the following information 
            #     \n\n {context}
            #     to answer in a helpful manner to the question. If you don't know the answer -
            #     say that you don't know. 
                
            #     ### Input: {question}
            #     ### Response:
            #     """.strip()
            # # """
            #             # prompt_template = '''
            #             # Your name is AI Agent. You are a virtual assistant for a Clinic.
            #             #             Here you can describe personality traits and job duties in plain language.
            #             #             ## Greeting Rules
            #             #             Greet the user and thank them for calling Clinic
            #             #             Prefix the greeting with a 'good morning', 'good afternoon', or a 'good evening' depending on the time of day.
            #             # ----------------
            #             # {history} \n {context}
            #             #
            #             # Question: {question}
            #             # Helpful Answer:'''
            #             # instruction = """
            # # Context: {history} \n {context}
            # # User: {question}"""
            # chat_history_str = "\n".join([f"{message['role']}: {message['content']}" for message in history])
            # final_template=prompt_template.replace("{history}",chat_history_str)
            # prompt_template = B_INST + final_template + E_INST
            # prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)
            # # prompt = PromptTemplate(input_variables=["context", "question", "history"], template=prompt_template)
            chat_history_str = "\n".join([f"{message['role']}: {message['content']}" for message in history])
            instruction = """
            Context: {history} \n {context}
            User: {question}"""
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
