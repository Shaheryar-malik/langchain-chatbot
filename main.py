import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
llm = ChatOllama(
    model="minimax-m2.5:cloud",
    temperature=0.7

)
prompt = ChatPromptTemplate.from_messages([
    ("system", "you are a helpful AI assistant"),
    ("human", "What is Rag?")
])

chain = prompt | llm | StrOutputParser()

response = chain.invoke({"Question": "What is Rag?"})
print(response)