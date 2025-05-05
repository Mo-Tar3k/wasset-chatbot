from flask import Flask, request, jsonify, render_template_string
import pandas as pd
from langchain.text_splitter import TokenTextSplitter
from langchain_community.vectorstores import FAISS  # ✅ التعديل هنا
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_huggingface import HuggingFaceEmbeddings  # ✅ التعديل هنا
from langchain_community.llms import HuggingFacePipeline  # ✅ التعديل هنا
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os
import traceback

# Load and truncate CSV data
df = pd.read_csv("products_dataset.csv").head(100)  # خدت أول 100 بس
documents = df["description"].astype(str).tolist()
metadatas = [
    {"product_id": row["product_id"], "title": row["title"]}
    for _, row in df.iterrows()
]

# Text splitting
splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=20)
chunks = splitter.create_documents(documents, metadatas=metadatas)

# Embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# Vector store
vector_db = FAISS.from_documents(chunks, embedding_model)

# Load lightweight LLM (distilgpt2)
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=256)
llm = HuggingFacePipeline(pipeline=pipe)

# Prompt template
qna_template = "\n".join([
    "You are an E-commerce chatbot assistant. Answer the next question using the provided context.",
    "If the answer is not contained in the context, say 'NO ANSWER IS AVAILABLE'.",
    "### Context:",
    "{context}",
    "",
    "### Question:",
    "{question}",
    "",
    "### Answer:",
])

qna_prompt = PromptTemplate(template=qna_template, input_variables=['context', 'question'], verbose=True)
stuff_chain = load_qa_chain(llm, chain_type="stuff", prompt=qna_prompt)

# Flask app
app = Flask(__name__)

# Simple HTML UI
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head><title>Product Chatbot</title></head>
<body>
    <h2>Ask about a product:</h2>
    <form id="chat-form">
        <input type="text" id="question" name="question" placeholder="Ask a question..." required>
        <button type="submit">Ask</button>
    </form>
    <h3>Answer:</h3>
    <pre id="answer"></pre>
    <script>
        document.getElementById("chat-form").onsubmit = async (e) => {
            e.preventDefault();
            const question = document.getElementById("question").value;
            const res = await fetch("/ask", {
                method: "POST",
                headers: { "Content-Type": "application/x-www-form-urlencoded" },
                body: `question=${encodeURIComponent(question)}`
            });
            const contentType = res.headers.get("content-type") || "";
            if (!res.ok || !contentType.includes("application/json")) {
                const errorText = await res.text();
                console.error("Error:", errorText);
                document.getElementById("answer").innerText = "⚠️ Server error. Check logs.";
                return;
            }
            const data = await res.json();
            document.getElementById("answer").innerText = data.answer;
        };
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/ask', methods=['POST'])
def ask():
    question = request.form.get('question')
    try:
        similar_docs = vector_db.similarity_search(question, k=1)
        response = stuff_chain({"input_documents": similar_docs, "question": question}, return_only_outputs=True)
        output_text = response.get('output_text', 'No answer found')
        answer = output_text.split('### Answer:')[1].strip() if '### Answer:' in output_text else output_text
        return jsonify({
            'answer': answer,
            'context': [doc.page_content for doc in similar_docs]
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'answer': f"Server error: {str(e)}",
            'context': []
        }), 500
