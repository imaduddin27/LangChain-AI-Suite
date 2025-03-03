from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq 
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama-3.2-90b-vision-preview")

messages = [
    SystemMessage("You are an expert in social media content strategy"),
    HumanMessage("Give a short tip to create engaging posts on iInstagram"), 
    AIMessage("")
] 

result = llm.invoke(messages)

print(result.content)
