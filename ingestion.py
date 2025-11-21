import asyncio
import os
import ssl
from typing import Any, Dict, List

import certifi
from dotenv import load_dotenv

from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
#from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
from openai import batches

from logger import (Colors, log_error, log_header, log_info, log_success,
                    log_warning)

load_dotenv()

# Configure SSL context to use certifi's CA bundle certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small", chunk_size=50, show_progress_bar=False, retry_min_seconds=10
)

#chroma = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
vectorstore = PineconeVectorStore(index_name=os.getenv("INDEX_NAME"), embedding=embeddings)
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=10, max_pages=1000)
tavily_crawl = TavilyCrawl()

async def index_documents_async(documents: List[Document], batch_size: int = 50):
    """Asynchronously index documents into the vector store in batches."""
    log_header("VECTOR STORAGE PHASE")
    log_info(
        f"** Vector Store Indexing: Preparing {len(documents)} documents for vector store.",
        Colors.DARKCYAN,
    )
    batches = [
        documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
    ]

    log_info(
        f"** Vector Store Indexing: Split into {len(batches)} batches of {batch_size} documents each."
    )

    async def add_batch(batch: List[Document], batch_num: int):
        try:
            await vectorstore.aadd_documents(batch)
            log_success(
                f"Vector Store Indexing: Successfully added batch {batch_num} / {len(batches)}  ({len(batch)}documents."
            )
        except Exception as e:
            log_error(
                f"Vector Store Indexing: Failed to add batch {batch_num} due to error: {e}"
            )
            return False
        return True


    tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    successful = sum(1 for result in results if result is True)

    if successful == len(batches):
        log_success(
            f"Vector Store Indexing: All batches processed successfully. ({successful} / {len(batches)})"
        )
    else:
        log_warning(
            f"Vector Store Indexing: Processed {successful} out of {len(batches)} batches successfully."
        )


async def main():
    """Main async function to orchestrate the entire processing pipeline."""
    print("Starting the data ingestion pipeline...")
    log_header("DOCUMENTATION INGESTION PIPELINE")

    log_info(
        "** TavilyCrawl: Starting crawl from seed URL 'https://python.langchain.com/",
        Colors.PURPLE,
    )

    #Crawl the documentation site
    res = tavily_crawl.invoke({
        "url": "https://python.langchain.com/",
        "max_depth": 5, #Should start with 1-2 to avoid too many pages and overrun.  Evaluate and increase as needed.
        "extract_depth": "advanced" #setting to advance Will increase latency but crawl more meta data like tables
        #"instructions": "content on ai agents"
        # The instructions field can be used to guide the crawler on what specific content to look for. and focus the results
    })

    #all_docs = res["results"]
    all_docs = [Document(page_content=result['raw_content'], metadata={"source": result['url']}) for result in res['results']]
    log_success(
        f"TavilyCrawl: Successfully crawled {len(all_docs)} URLs from site."
    )

    log_header("DOCUMENT CHUNKING PHASE")
    log_info(
        f"**  Text Splitter:  Processing {len(all_docs)} documents with 4000 chunk size and 200 overlap",
            Colors.YELLOW,
    )

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(
        f"Text Splitter: Successfully split documents into {len(splitted_docs)} chunks."
    )

    #Process documents into vector store asynchronously
    await index_documents_async(splitted_docs, batch_size=500)

    log_header("INGESTION PIPELINE COMPLETE")
    log_success("Data ingestion pipeline completed successfully!")
    log_info("Summary:", Colors.BOLD)
    #log_info(f"- URLs mapped: {len(site_map['results'])}")
    log_info(f"- Documents Extracted: {len(all_docs)}")
    log_info(f"- Document Chunks Created: {len(splitted_docs)}")

if __name__ == "__main__":
    asyncio.run(main())    

