import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

index_name = os.environ["PINECONE_INDEX"]
cloud = os.environ["PINECONE_CLOUD"]
region = os.environ["PINECONE_REGION"]

# Create index with integrated embedding (llama-text-embed-v2, default dimension 1024)
existing = [i["name"] for i in pc.list_indexes()]
if index_name not in existing:
    pc.create_index_for_model(
        name=index_name,
        cloud=cloud,
        region=region,
        embed={
            "model": "llama-text-embed-v2",
            "field_map": {"text": "text"},
        },
    )
    print("Created:", index_name)
else:
    print("Already exists:", index_name)
