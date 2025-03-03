import os 
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings 

# Define the current dir
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

# Define the embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

#load the existing vector storte with embedding function
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# Actual question from the user
query = "Where is the Dracula's Castle located?"

# Retrive relevant documents based on query
retriever = db.as_retriever(
    search_type = "similarity_score_threshold",
    search_kwargs = {"k":3, "score_threshold": 0.2},
)
relevant_docs = retriever.invoke(query)

# Display the relevant results wwith metadata
print("\n---- Relevant Documents ----")
for i, doc in enumerate(relevant_docs, 1):
    print(f"document {i}:\n{doc.page_content}\n")
    print(f"Source: {doc.metadata['source']}\n")

