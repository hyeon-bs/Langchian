import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if __name__ == '__main__':
    print("Ingesting...")
    loader = TextLoader("/Users/baeksohyeon/Desktop/intro-to-vector-dbs/mediumblog1.txt")
    document = loader.load()

    print("splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    # embeddings = OllamaEmbeddings(model="nomic-embed-text")

    print("ingesting...")
    PineconeVectorStore.from_documents(
        texts,
        embeddings,
        index_name=os.environ['INDEX_NAME'])
    print("finish")