from dotenv import load_dotenv
from langchain_community import vectorstores
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from transformers import LlamaForCausalLM, LlamaTokenizer

model = SentenceTransformer("all-minilm-l6-v2")
