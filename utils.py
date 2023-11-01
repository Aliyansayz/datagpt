from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import pinecone
import asyncio
from langchain.document_loaders.sitemap import SitemapLoader
from langchain.document_loaders import DirectoryLoader
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain


#Function to fetch data from website
#https://python.langchain.com/docs/modules/data_connection/document_loaders/integrations/sitemap
def get_website_data(sitemap_url):

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loader = SitemapLoader(
    sitemap_url
    )

    docs = loader.load()

    return docs


def load_docs(directory):
  loader = DirectoryLoader(directory)
  documents = loader.load()
  return documents


# number of docs
# documents = load_docs(directory)
# len(documents)


#Function to split data into smaller chunks
def split_data(docs):

    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap  = 200,
    length_function = len,
    )

    docs_chunks = text_splitter.split_documents(docs)
    return docs_chunks

#Function to create embeddings instance
def create_embeddings(model_name="all-MiniLM-L6-v2"):

    embeddings = SentenceTransformerEmbeddings(model_name)
    return embeddings

def create_embeddings_openai():
    embeddings = OpenAIEmbeddings(model_name="ada")
    return embeddings

#Function to push data to Pinecone
def push_to_pinecone(pinecone_apikey,pinecone_environment,pinecone_index_name,embeddings,docs):

    pinecone.init(
    api_key=pinecone_apikey,
    environment=pinecone_environment
    )

    index_name = pinecone_index_name
    index = Pinecone.from_documents(docs, embeddings, index_name=index_name)
    return index

#Function to pull index data from Pinecone
def pull_from_pinecone(pinecone_apikey,pinecone_environment,pinecone_index_name,embeddings):

    pinecone.init(
    api_key=pinecone_apikey,
    environment=pinecone_environment
    )

    index_name = pinecone_index_name

    index = Pinecone.from_existing_index(index_name, embeddings)
    return index

#This function will help us in fetching the top relevent documents from our vector store - Pinecone Index
def get_similar_docs(index,query,k=2):

    similar_docs = index.similarity_search(query, k=k)
    return similar_docs





def get_answer_from_openai(query,  similar_docs, model_name="gpt-3.5-turbo" ):
  model_name = "gpt-3.5-turbo"
  llm = OpenAI(model_name)  
  similar_docs = get_similiar_docs(query)
  chain = load_qa_chain(llm, chain_type="stuff")  
  answer = chain.run(input_documents=similar_docs, question=query)
  return answer
