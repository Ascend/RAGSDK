from loguru import logger
from pymilvus import MilvusClient


def main():
    milvus_url: str = "http://127.0.0.1:19530"
    milvus_client = MilvusClient(milvus_url)
    collection_names = milvus_client.list_collections()
    for collection_name in collection_names:
        milvus_client.drop_collection(collection_name)
        logger.info(f"collection '{collection_name}' dropped successfully")


if __name__ == "__main__":
    main()
