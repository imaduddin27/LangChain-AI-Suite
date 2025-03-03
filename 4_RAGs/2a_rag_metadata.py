import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings 

# Define the directory containing the text files and the persistent dir
current_dir = os.path.dirname(os.path.abspath(__file__))
books_dir = os.path.join(current_dir, "documents")
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

print(f"Books directory: {books_dir}")
print(f"Persistent directory: {persistent_directory}")

# Check if the chroma vector store already exists
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. initializing vector store...")

    # Ensure the text file exists
    if not os.path.exists(books_dir):
        raise FileNotFoundError(
            f"The file {books_dir} does not exist, please check the path."
        )
    
    # List all the text files in the directory
    book_files= [f for f in os.listdir(books_dir) if f.endswith(".txt")]

    # Read the text content from each file and store it with metadata
    documents = []
    for book_file in book_files:
        file_path = os.path.join(books_dir, book_file)
        loader = TextLoader(file_path, encoding="utf-8")
        books_docs = loader.load()
        for doc in books_docs:
            # Add the metadata to each document indicating its source
            doc.metadata = {"source": book_file}
            documents.append(doc)
        

    # Split the documents into chunks
    text_splitter= CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    # Display information about the split documents
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")

    # Create embeddings
    print("\n--- Creating embeddings ---")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    print("\n---Finished creating embeddings ---")

    # create the vector store and persist it automatically
    print("\n--- Creating vector store ---")
    db = Chroma.from_documents(
        docs, embeddings, persist_directory=persistent_directory
    )
    print("\n--- Finished creating vector store ---")

else:
    print ("Vector store already exists. No need to initialize")