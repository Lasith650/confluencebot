from flask import Flask, request, jsonify
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import OpenAI
import os

app = Flask(__name__)

openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")
os.environ["OPENAI_API_KEY"] = openai_key

# Load FAISS index with safe deserialization
embedding_model = OpenAIEmbeddings()
vector_store = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)

llm = OpenAI(temperature=0)
qa_chain = load_qa_chain(llm, chain_type="stuff")

@app.route("/api/messages", methods=["POST"])
def chatbot():
    data = request.json
    user_input = data.get("text", "")

    docs = vector_store.similarity_search(user_input, k=3)
    answer = qa_chain.run(input_documents=docs, question=user_input)

    return jsonify({"text": answer})

if __name__ == "__main__":
    app.run(debug=True)