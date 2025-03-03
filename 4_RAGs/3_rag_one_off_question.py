import os 
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings 
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Define the current dir
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

# Define the embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

#load the existing vector storte with embedding function
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# Actual question from the user
query = "What happened to Alice when she entered the rabbit-hole?"

# Retrive relevant documents based on query
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.2},
)
relevant_docs = retriever.invoke(query)

# Display the relevant results wwith metadata
print("\n---- Relevant Documents ----")
for i, doc in enumerate(relevant_docs, 1):
    print(f"document {i}:\n{doc.page_content}\n")
    print(f"Source: {doc.metadata['source']}\n")

# Combine the query and the relevant chunks content
Combined_input = (
    "Here are some documents that might help answer the question:"
    + query
    + "\n\nRelevant Documents: \n"
    + "\n\n" .join([doc.page_content for doc in relevant_docs])
    + "\n\nPlease provide a rough answer based only on provided documents. if answer is not found in the documents then respond with 'I'm not sure'."
)

load_dotenv()

# Create a llm model
llm = llm = ChatGroq(model="deepseek-r1-distill-llama-70b") 

messages = [
    SystemMessage(content= "You are a helpful assistant"),
    HumanMessage(content = Combined_input)
]

result = llm.invoke(messages)

# Display the content of the result
print("\n--- Generated Response ---")
print(result.content)