"""Ask a question to the notion database."""
import faiss
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
import pickle
import argparse
import pathlib
from langchain import OpenAI

parser = argparse.ArgumentParser(description='Ask a question to the notion DB.')
parser.add_argument('question', type=str, help='The question to ask the notion DB')
args = parser.parse_args()

# Load the LangChain.
index = faiss.read_index("docs.index")

with open("faiss_store.pkl", "rb") as f:
    store = pickle.load(f)

store.index = index
llm = ChatOpenAI(model_name="gpt-4", temperature=0, max_tokens=1000)
chain = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, retriever=store.as_retriever())
result = chain({"question": args.question})
print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
