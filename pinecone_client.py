import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

_pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
_index = _pc.Index(os.environ["PINECONE_INDEX"])

def get_index():
    return _index
