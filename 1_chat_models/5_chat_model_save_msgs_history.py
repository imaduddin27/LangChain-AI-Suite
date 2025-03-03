from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq 
from dotenv import load_dotenv
from google.cloud import firestore
from langchain_google_firestore import FirestoreChatMessageHistory

load_dotenv()

PROJECT_ID = "langchain-e01b2"
SESSION_ID= "user_session_new"
COLLECTION_NAME = "chat_history"


# Initialize Firestore Client
print("Initializing Firestore Client...")
client = firestore.Client(project=PROJECT_ID)

# initialize Firestore Chat Message History
print("Initializing Firestore Chat Message History...")
chat_history = FirestoreChatMessageHistory(
    session_id = SESSION_ID,
    collection=COLLECTION_NAME,
    client=client,

)
print("Chat History Initialized.")
print("current Chat History:", chat_history.messages)

llm = ChatGroq(model="llama-3.2-90b-vision-preview")

print("Start chatting with AI. Type 'exit' to quit.")

# Chat loop
while True:
    human_input = input("User: ")
    if human_input.lower() =="exit":
        break
    chat_history.add_user_message(human_input)

    ai_response = llm.invoke(chat_history.messages)
    
    chat_history.add_ai_message(ai_response.content)

    print(f"AI: {ai_response.content}")