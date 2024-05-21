from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

query = "Examples of callouts"
embedding_function = OpenAIEmbeddings()

db = Chroma(persist_directory="./chroma.db", embedding_function=embedding_function)

# perform a similarity search on the loaded database
# Note: This is to demonstrate that the loaded database is functioning correctly.
docs = db.similarity_search(query, k=8)
for doc in docs:
    print("-----")
    print(doc.page_content)