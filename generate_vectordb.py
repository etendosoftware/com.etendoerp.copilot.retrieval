# -*- coding: utf-8 -*-
from git import Repo
from langchain_text_splitters import Language
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
import os

# Clone
repo_path = "./test_repo"
# if repo path not exists, clone the repo
if not os.path.exists(repo_path):
    repo = Repo.clone_from(os.environ["REPO_URL"], to_path=repo_path)

"""We load the py code using [`LanguageParser`](/docs/integrations/document_loaders/source_code), which will:

* Keep top-level functions and classes together (into a single document)
* Put remaining code into a separate document
* Retains metadata about where each split comes from
"""


# Load
loader = GenericLoader.from_filesystem(
    repo_path ,
    glob="**/*",
    suffixes=[".java"],
    exclude=["**/non-utf8-encoding.py"],
    parser=LanguageParser(language=Language.JAVA, parser_threshold=500),

)
documents = loader.load()
len(documents)

"""### Splitting

Split the `Document` into chunks for embedding and vector storage.

We can use `RecursiveCharacterTextSplitter` w/ `language` specified.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.JAVA, chunk_size=2000, chunk_overlap=200
)
texts = python_splitter.split_documents(documents)
len(texts)

"""### RetrievalQA

We need to store the documents in a way we can semantically search for their content.

The most common approach is to embed the contents of each document then store the embedding and document in a vector store.

When setting up the vectorstore retriever:

* We test [max marginal relevance](/docs/use_cases/question_answering) for retrieval
* And 8 documents returned

#### Go deeper

- Browse the > 40 vectorstores integrations [here](https://integrations.langchain.com/).
- See further documentation on vectorstores [here](/docs/modules/data_connection/vectorstores/).
- Browse the > 30 text embedding integrations [here](https://integrations.langchain.com/).
- See further documentation on embedding models [here](/docs/modules/data_connection/text_embedding/).
"""

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

db = Chroma.from_documents(texts, OpenAIEmbeddings(disallowed_special=(), show_progress_bar=True), persist_directory="./chroma.db")