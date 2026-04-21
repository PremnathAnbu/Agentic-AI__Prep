import os
from langchain_community.document_loaders import TextLoader,DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()


def load_documents(docs_path="docs"):
    """Load all text files from the docs directory"""
    print(docs_path)
    
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist.")
    
    loader=DirectoryLoader(path=docs_path,glob="*.txt",loader_cls=TextLoader)
    
    documents=loader.load()
    
    if len(documents) ==0:
        raise FileNotFoundError(f"No .txt file found in {docs_path}.")
    
    for i, doc in enumerate(documents[:3]):
        print(f"\n Document {i+1}:")
        print(f"\n Source:{doc.metadata['source']}")
        print(f"\n Content length:{len(doc.page_content)} charcters")
        print(f" content preview: {doc.page_content[:100]}....")
        print(f"metadata :{doc.metadata}")
        
    return documents





def split_documents(documents,chunk_size=800,chunk_overlap=100):
    
    """Split documents into smaller chunks with overlap"""
    text_splitter=CharacterTextSplitter(separator="\n\n",
                                        chunk_overlap=chunk_overlap,
                                        chunk_size=chunk_size)
    chunks=text_splitter.split_documents(documents)
    
    if chunks:
        for i, chunk in enumerate(chunks[:5]):
            print(f"\n --Chunk {i+1}--")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length:{len(chunk.page_content)} characters")
            print(f"Contents")
            print(chunk.page_content)
            print("-"*50)
            
        if len(chunks)>5:
            print(f"\n... and {len(chunks)-5} more chunks")
            
    return chunks

def create_vector_store(chunks,persist_directory="db/chroma_db"):
    """Create and presist chromaDB vector store"""
    print("Creating embedding and storing in chromadb")
    embedding_model=OllamaEmbeddings(model="nomic-embed-text")
    vectorstore=Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space":"cosine"}
    
    )
    
    print(f"Vector store created and saved to {persist_directory}")
    return vectorstore
    



def main():
    print("This is Main function")
    #loading the files
    documents=load_documents(docs_path="docs")
    
        
        
    #chunking the files
    chunks=split_documents(documents)
    
    #embedding and storing in vector db
    vectorstore=create_vector_store(chunks)
if __name__ == "__main__":
    main()
