from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from .seed import Seed
from .config import Config

class OrganicAI:
    def __init__(self):
        self.config = Config()
        self.seed = Seed(self.config)
        self.llm = Ollama(model="llama2") # Default Ollama model

        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are Organic AI, a helpful AI assistant."),
            ("human", "{question}")
        ])

        self.chain = ({"question": RunnablePassthrough()} | self.prompt_template | self.llm | StrOutputParser())

    def process_prompt(self, prompt: str) -> str:
        # Here we can integrate the seed's learning or other functionalities
        # For now, let's just pass the prompt to the LLM
        response = self.chain.invoke(prompt)
        self.seed.learn({'text': prompt, 'response': response}) # Example of integrating seed learning
        return response
