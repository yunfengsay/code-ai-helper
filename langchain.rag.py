from langchain.llms import Ollama
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA

ollama = Ollama(model="zephyr")

loader = PyPDFLoader('~/Documents/books/matrix vector derivatives for machine learning.pdf')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

embeddings = OllamaEmbeddings(model='zephyr')
vectorstore = Chroma.from_documents(texts, embeddings)

question="矩阵迹是什么?用中文回答"

qachain = RetrievalQA.from_chain_type(ollama, retriever = vectorstore.as_retriever())
response = qachain({"query": question})
print(response)
