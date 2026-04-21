from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv

load_dotenv()

presistent_directory="db/chroma_db"

#load embeddings and vector store
embedding_model=OllamaEmbeddings(model="nomic-embed-text")

db=Chroma(persist_directory=presistent_directory,
          embedding_function=embedding_model,
          collection_metadata={"hnsw:space":"cosine"}
          )

# search for relevant info from the documents
query="Python was conceived in the late?"

retriever=db.as_retriever(search_kwargs={"k":5})

relevant_docs=retriever.invoke(query)


print(f"User Query: {query}")
print("--Content--")
for i, doc in enumerate(relevant_docs,1):
    print(f"Document {i}: \n {doc.page_content}\n")