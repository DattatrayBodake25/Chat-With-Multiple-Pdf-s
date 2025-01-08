import streamlit as st
import pdfplumber
from sentence_transformers import SentenceTransformer
import numpy as np
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import faiss
import logging

import warnings
warnings.filterwarnings("ignore", message="Tried to instantiate class '__path__._path'")

# Load environment variables
load_dotenv()

# Initialize Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("Gemini API key is missing. Please set it in the .env file.")
    st.stop()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=GEMINI_API_KEY, temperature=1)

# Initialize SentenceTransformer model for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Configure logging
logging.basicConfig(level=logging.INFO)

# Streamlit app setup
st.set_page_config(page_title="Chat with PDF", layout="wide")
st.title("Chat with PDF Application")

# Sidebar for PDF upload
with st.sidebar:
    st.header("Upload PDF(s)")
    uploaded_files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)

# Helper function to extract text from PDF
def extract_text_from_pdf(file):
    try:
        text = ""
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip()
    except Exception as e:
        logging.error(f"Error reading PDF: {e}")
        return ""

# Helper function to split text into chunks
def chunk_text(text, max_chunk_size=1000):
    sentences = text.split('. ')
    chunks, current_chunk = [], ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# Function to compute embeddings
def get_embeddings(chunks):
    try:
        return np.array([embedding_model.encode(chunk) for chunk in chunks])
    except Exception as e:
        logging.error(f"Error generating embeddings: {e}")
        return np.array([])

# Function to find the most relevant chunk using FAISS
def find_relevant_chunk(question_embedding, chunk_embeddings):
    index = faiss.IndexFlatL2(chunk_embeddings.shape[1])
    index.add(chunk_embeddings)
    _, indices = index.search(np.array([question_embedding]), 1)
    return indices[0][0]

# Function to get answers from Gemini API
def get_answer_from_gemini(question, context):
    prompt = f"""
    You are a helpful assistant. Based on the following context, provide a clear, concise, and detailed answer.

    Question: {question}
    Context: {context}
    Answer:
    """
    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        logging.error(f"Error during answer generation: {e}")
        return "Sorry, there was an error generating the answer."

# Cache embeddings and chunks for performance
if uploaded_files:
    combined_text = ""
    for file in uploaded_files:
        combined_text += extract_text_from_pdf(file) + "\n\n"

    if not combined_text.strip():
        st.error("Failed to extract text from the uploaded PDFs. Please check the files.")
        st.stop()

    st.subheader("Extracted PDF Content")
    st.text_area("PDF Content", combined_text, height=300)

    chunks = chunk_text(combined_text)
    if not chunks:
        st.error("No text chunks generated. Please check the PDF content.")
        st.stop()

    embeddings = get_embeddings(chunks)
    if embeddings.size == 0:
        st.error("Failed to generate embeddings for the text chunks.")
        st.stop()

    question = st.text_input("Ask a question about the document:")

    if question:
        query_embedding = embedding_model.encode(question)
        best_chunk_idx = find_relevant_chunk(query_embedding, embeddings)
        best_chunk = chunks[best_chunk_idx]

        answer = get_answer_from_gemini(question, best_chunk)

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Relevant Section from PDF")
        st.text_area("Context from PDF", best_chunk, height=150)
else:
    st.info("Please upload at least one PDF file to begin.")