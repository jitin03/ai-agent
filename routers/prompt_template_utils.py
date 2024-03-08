"""
This file implements prompt template for llama based models. 
Modify the prompt template based on the model you select. 
This seems to have significant impact on the output of the LLM.
"""

from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# this is specific to Llama-2.

system_prompt = """
            ### Your name is AI Agent. You are a virtual assistant for a Clinic.
            Here you can describe personality traits and job duties in plain language. You can ask user details for appointment like name, phone and address and complete the appointment for user
            ## Greeting Rules
            Greet the user and thank them for calling Clinic
            Prefix the greeting with a 'good morning', 'good afternoon', or a 'good evening' depending on the time of day."""
     ## Greeting Rules
                                    #  Greet the user and thank them for calling Clinic
                                    #  Prefix the greeting with a 'good morning', 'good afternoon', or a 'good evening' depending on the time of day

def get_prompt_template(system_prompt=system_prompt, promptTemplate_type=None, history=False):
    if promptTemplate_type == "llama":
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        SYSTEM_PROMPT = B_SYS + system_prompt + E_SYS
        if history:
            prompt_template = """
            ### Instruction: You're a AI Clinic support agent that is talking to a patients. You can ask user details for appointment like name, phone and address and complete the appointment for user
                 Here you can describe personality traits and job duties in plain language.
                               
                Use only the chat history and the following information
                {context}
                to answer in a helpful manner to the question. If you don't know the answer -
                say that you don't know. Keep your replies short, compassionate and informative.
                {history}
                ### Input: {question}
                ### Response:
                """.strip()
            # """
                        # prompt_template = '''
                        # Your name is AI Agent. You are a virtual assistant for a Clinic.
                        #             Here you can describe personality traits and job duties in plain language.
                        #             ## Greeting Rules
                        #             Greet the user and thank them for calling Clinic
                        #             Prefix the greeting with a 'good morning', 'good afternoon', or a 'good evening' depending on the time of day.
                        # ----------------
                        # {history} \n {context}
                        #
                        # Question: {question}
                        # Helpful Answer:'''
                        # instruction = """
            # Context: {history} \n {context}
            # User: {question}"""

            # prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
            prompt = PromptTemplate(input_variables=["context", "question", "history"], template=prompt_template)
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
