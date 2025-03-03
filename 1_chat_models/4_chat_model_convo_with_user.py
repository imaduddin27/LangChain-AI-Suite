from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq 
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama-3.2-90b-vision-preview")

chat_history = []

# set an initial system message 
system_message = SystemMessage(content = "You arean helpful AI Assitant")
chat_history.append(system_message) 

# Chat loop
while True:
    query = input("You: ")
    if query.lower() =="exit":
        break
    chat_history.append(HumanMessage(content = query)) # Add the user message

    result = llm.invoke(chat_history)
    response = result.content
    chat_history.append(AIMessage(content = response))

    print(f"AI: {response}")

print("---- History ----")
print(chat_history)