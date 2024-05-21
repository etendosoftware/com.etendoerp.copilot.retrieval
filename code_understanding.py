from git import Repo
from langchain_text_splitters import Language
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

embedding_function = OpenAIEmbeddings()

db = Chroma(persist_directory="./chroma.db", embedding_function=embedding_function)
retriever = db.as_retriever(
    search_type="mmr",  # mmr or "similarity"
    search_kwargs={"k": 8},
)

"""### Chat

Test chat, just as we do for [chatbots](/docs/use_cases/chatbots).

#### Go deeper

- Browse the > 55 LLM and chat model integrations [here](https://integrations.langchain.com/).
- See further documentation on LLMs and chat models [here](/docs/modules/model_io/).
- Use local LLMS: The popularity of [PrivateGPT](https://github.com/imartinez/privateGPT) and [GPT4All](https://github.com/nomic-ai/gpt4all) underscore the importance of running LLMs locally.
"""

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

# First we need a prompt that we can pass into an LLM to generate this search query

prompt = ChatPromptTemplate.from_messages(
    [
        ("placeholder", "{chat_history}"),
        ("user", "{input}"),
        (
            "user",
            "Given the above conversation, generate a search query to look up to get information relevant to the conversation",
        ),
    ]
)

retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the user's questions based on the below context:\n\n{context}",
        ),
        ("placeholder", "{chat_history}"),
        ("user", "{input}"),
    ]
)
document_chain = create_stuff_documents_chain(llm, prompt)

qa = create_retrieval_chain(retriever_chain, document_chain)

while True:
    question = input("Por favor, introduce tu pregunta (o 'salir' para terminar): ")
    if question.lower() == 'salir':
        break
    result = qa.invoke({"input": question})
    print(f"-> **Pregunta**: {question} \n")
    print(f"**Respuesta**: {result['answer']} \n")