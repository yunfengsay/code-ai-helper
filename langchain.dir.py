import sys
import torch
from langchain.llms import Ollama
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Get file path from command-line argument
file_path = sys.argv[1] if len(sys.argv) > 1 else './'  # Default path is './' if no argument is provided

# Modify the loader based on the file_path provided
if file_path.endswith('.pdf'):
    loader = PyPDFLoader(file_path)
else:
    loader = DirectoryLoader(file_path, glob="**/*.ts")

documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

ollama = Ollama(model="zephyr", device=device)
embeddings = OllamaEmbeddings(model='zephyr', device=device)
vectorstore = Chroma.from_documents(texts, embeddings)

# Interactive Q&A loop
while True:
    question = input("Please enter your question (in Chinese): ")
    
    if question.lower() in ['exit', 'quit', 'q']:
        print("Exiting Q&A session.")
        break
    
    qachain = RetrievalQA.from_chain_type(ollama, retriever=vectorstore.as_retriever())
    response = qachain({"query": question})
    print("Answer:", response)
