import uuid, os
from os import getenv

from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_openai import OpenAIEmbeddings
# from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def ingest_docs():
    loader = ReadTheDocsLoader("langchain-docs/api.python.langchain.com/en/latest")

    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50
    )
    documents = text_splitter.split_documents(raw_documents)
    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url.split("?")[0]})

    print(f"Going to add {len(documents)} to Pinecone")
    # PineconeVectorStore.from_documents(
    #    documents,
    #    embeddings,
    #    index_name=INDEX_NAME,
    # )

    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]

    pc = Pinecone(api_key=API_KEY)
    index = pc.Index(INDEX_NAME)
    # index = pc.list_indexes() # str
    # print(indexes)

    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i: i + batch_size]
        batch_metadatas = metadatas[i: i + batch_size]
        batch_vectors = embeddings.embed_documents(batch_texts)

        # 각 벡터에 UUID 부여 - 중복방지
        vectors_to_upsert = [
            (str(uuid.uuid4()), vector, metadata)
            for vector, metadata in zip(batch_vectors, batch_metadatas)
        ]

        index.upsert(vectors=vectors_to_upsert)
        print(f"✅ Uploaded batch {i // batch_size + 1} ({len(vectors_to_upsert)} vectors)")

    print("🎉 Uploaded to Pinecone successfully")

    # ✅ 업로드 후 벡터 수 확인
    pc = Pinecone()
    index = pc.Index("langchain-doc-index2")
    stats = index.describe_index_stats()
    print(f"📊 Index stats: {stats}")

def ingest_docs2() -> None:
    from langchain_community.document_loaders.firecrawl import FireCrawlLoader

    langchain_documents_base_urls = [
        "https://python.langchain.com/docs/integrations/chat/",
        "https://python.langchain.com/docs/integrations/llms/",
        "https://python.langchain.com/docs/integrations/text_embedding/",
        "https://python.langchain.com/docs/integrations/document_loaders/",
        "https://python.langchain.com/docs/integrations/document_transformers/",
        "https://python.langchain.com/docs/integrations/vectorstores/",
        "https://python.langchain.com/docs/integrations/retrievers/",
        "https://python.langchain.com/docs/integrations/tools/",
        "https://python.langchain.com/docs/integrations/stores/",
        "https://python.langchain.com/docs/integrations/llm_caching/",
        "https://python.langchain.com/docs/integrations/graphs/",
        "https://python.langchain.com/docs/integrations/memory/",
        "https://python.langchain.com/docs/integrations/callbacks/",
        "https://python.langchain.com/docs/integrations/chat_loaders/",
        "https://python.langchain.com/docs/concepts/",
    ]

    langchain_documents_base_urls2 = [langchain_documents_base_urls[0]]
    for url in langchain_documents_base_urls2:
        print(f"FireCrawling {url=}")
        loader = FireCrawlLoader(
            url=url,
            api_key=FIRECRAWL_API_KEY,
            mode="scrape",
        )
        docs = loader.load()

        print(f"Going to add {len(docs)} documents to Pinecone")
        PineconeVectorStore.from_documents(
            docs, embeddings, index_name="firecrawl-index"
        )
        print(f"****Loading {url} to vectorstore done****")


if __name__ == "__main__":
    ingest_docs2()