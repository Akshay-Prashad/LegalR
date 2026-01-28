import pdfplumber as pdf
import sentence_transformers
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import LlamaForCausalLM, LlamaTokenizer

file_path = "Constitution_of_india.pdf"


def readPdf(file_path: str) -> str:
    print("Reading Process Started...")
    with pdf.open(file_path) as file:
        text = " "
        for i in file.pages:
            text += i.extract_text(x_tolerance=3)
        text_file = open("text", "w+")
        text_file.write(text)
        print("Pdf Read completed...")
        return text


def textSplitter() -> list:
    with open("text", "r") as file:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=250, chunk_overlap=50, length_function=len
        )
        textFile = text_splitter.split_text(file)
    return textFile


text = textSplitter()
for i in text:
    print(i)


def embedText(text: str) -> list:
    model = SentenceTransformer("all-minilm-l6-v2")
    return model.encode(text)
