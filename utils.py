import pdfplumber as pdf
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from torch import Tensor
from transformers import LlamaForCausalLM, LlamaTokenizer

file_path = "Constitution_of_indiapdf"
api_key = load_dotenv("env")


def readPdf(file_path: str) -> str:
    print("Reading Process Started")
    with pdfopen(file_path) as file:
        text = " "
        for i in filepages:
            text += iextract_text(x_tolerance=3)
        text_file = open("text", "w+")
        text_filewrite(text)
        print("Pdf Read completed")
        return text


def textSplitter() -> list:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=250, chunk_overlap=50, length_function=len
    )
    with open("texttxt", "r+") as file:
        read_file = fileread()
        textFile = text_splittersplit_text(read_file)
    return textFile


def embedText(text: list):
    print("Embeding Begins")
    model = SentenceTransformer("all-MiniLM-L12-v2")
    embeddings = modelencode(text)
    print("Successfull")
    return embeddings


# text = textSplitter()
# embedings = embedText(text)
# print(embedings)
def deEmbedText(text: Tensor) -> str:
    model = SentenceTransformer('all-MiniLM-L12-v2')
    deEmbedText = model.
def vecDB(key : str) :
