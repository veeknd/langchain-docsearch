'''
This program supports pdf files
store the documents in in ./data

requirements.text
chromadb==0.3.26
Cython==0.29.35
faiss-cpu==1.7.4
langchain==0.0.194
langchainplus-sdk==0.0.7
openai==0.27.8
openapi-schema-pydantic==1.2.4
pdf2image==1.16.3
pypdf==3.9.1
python-magic==0.4.27
python-magic-bin==0.4.14
tiktoken==0.4.0
tqdm==4.65.0
unstructured==0.7.2

'''
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import OpenAI
from langchain.chains import RetrievalQA
import magic
import os

#Enter Valid OPENAI API KEY
API_KEY= "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"

#Relative Directory of documents folder
DOCUMENT_DIR= "./data"

openai_api_key = os.getenv(API_KEY)

# Get your loader ready
loader = PyPDFDirectoryLoader(DOCUMENT_DIR)
# Load up your text into documents
documents = loader.load_and_split()

# Get your text splitter ready
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

# Split your documents into texts
texts = text_splitter.split_documents(documents)

# Turn your texts into embeddings
embeddings = OpenAIEmbeddings(openai_api_key=API_KEY)

# Get your docsearch ready
docsearch = FAISS.from_documents(texts, embeddings)

# Load up your LLM
llm = OpenAI(openai_api_key=API_KEY)

# Create your Retriever
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever())
qa = RetrievalQA.from_chain_type(llm=llm,
                                chain_type="stuff",
                                retriever=docsearch.as_retriever(),
                                return_source_documents=True)

print("\n*you can ask the demo questions in the second page of each document in dir /data*\n")
print("*or you can replace your own documents in the folder /data and query from them*\n")


#Loop of query and result
while 1:
	query= str(input('query:'))
	result = qa({"query": query})
	def print_result():
		print(f"result:{result['result']}")
		(a ,b) = result['source_documents'][0]
		print(b[1])
		print("\n")
	print_result()



