# Chat with PDF Application

## Overview
The "Chat with PDF" application is an interactive tool built using Streamlit that allows users to upload PDF documents, extract their content, convert the extracted text into embeddings, and interact with it by asking questions. The application leverages machine learning techniques such as embeddings, FAISS for similarity search, and the Gemini LLM for generating context-aware responses.

## Table of Contents
- [Features](#features)
- [Technical Specifications](#technical-specifications)
- [Installation Guide](#installation-guide)
  - [Dependencies](#dependencies)
  - [Setup Instructions](#setup-instructions)
- [How to Use](#how-to-use)
- [Code Explanation](#code-explanation)
- [Bonus Features](#bonus-features)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

## Features
1. **PDF Upload & Text Extraction**: Upload PDFs and extract text from them using pdfplumber.
2. **Text Embedding & Search**: Convert extracted text into embeddings using the `sentence-transformers` model and search for the most relevant context based on a user query.
3. **LLM-Based Question Answering**: Use the Gemini API (or any other LLM like GPT) to generate answers based on the extracted content.
4. **Multiple PDF Support**: Allows uploading multiple PDFs at once and querying across documents.
5. **Context-Aware Responses**: Respond to user queries with answers backed by the context from the uploaded PDFs.

## Technical Specifications
- **Frontend**: Streamlit
- **Backend**: Python (with dependencies such as PyPDF2, pdfplumber, LangChain, Sentence-Transformers, FAISS)
- **LLM**: Gemini (via ChatGoogleGenerativeAI API) or any compatible LLM like GPT-3.5
- **Embeddings Model**: `sentence-transformers` (e.g., `all-MiniLM-L6-v2`)
- **Similarity Search**: FAISS (for efficient nearest neighbor search)

## Installation Guide

### Dependencies
This project requires Python 3.7+ and several Python libraries, which can be installed using `pip`.

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-repo/chat-with-pdf.git
   cd chat-with-pdf
   ```

### Install the required dependencies:

```bash
pip install -r requirements.txt
```
### The requirements.txt file includes all the necessary libraries:
```bash
streamlit
pdfplumber
sentence-transformers
numpy
langchain-google-genai
faiss-cpu
python-dotenv
logging
Setup Instructions
Set up your environment:
```
Create a .env file at the root of the project.
Add your Gemini API key (or other LLM API keys) in the .env file:
```bash
GEMINI_API_KEY=your_api_key_here
```
Run the app: Once the dependencies are installed and the .env file is set up, start the application by running:

```bash
streamlit run app.py
```
### How to Use
- Upload PDF: Use the sidebar to upload one or more PDF files.
- View Extracted Content: Once the PDF is uploaded, the extracted text will be displayed in a text area.
- Ask Questions: Enter your questions about the content of the PDF in the provided input field.
- Get Answers: The system will fetch the most relevant chunk of text from the PDF and generate an answer using the Gemini LLM.
- View Context: The application will highlight the section of the document that is most relevant to your query.

### Code Explanation
Main Workflow:
- PDF Extraction: The uploaded PDFs are processed using pdfplumber, which extracts the text from each page.
- Text Chunking: The extracted text is split into manageable chunks for embedding and querying purposes. This is done by splitting the text into sentences and grouping them into chunks of a predefined size.
- Embeddings: The text chunks are converted into embeddings using the sentence-transformers model, which creates a high-dimensional vector representation of the text.
- FAISS for Similarity Search: The embeddings are indexed using FAISS, which allows for fast similarity search when a question is asked.
- Question Answering: The user's question is converted into an embedding and compared to the existing text chunks to find the most relevant content. This content is then used to generate a response via the Gemini LLM.

### Key Functions:
- extract_text_from_pdf(file): Extracts text from the uploaded PDF.
- chunk_text(text): Splits the extracted text into chunks for processing.
- get_embeddings(chunks): Generates embeddings for each text chunk.
- find_relevant_chunk(question_embedding, chunk_embeddings): Finds the most relevant chunk for a given question using FAISS.
- get_answer_from_gemini(question, context): Uses the Gemini API to generate an answer based on the relevant context.

### Bonus Features
- Highlight Relevant Sections: The application highlights the section of the uploaded PDF that is most relevant to the answer provided.
- Cross-Document Querying: You can upload multiple PDFs and ask questions that span across documents.
- Deployment: The application can be deployed on platforms like Streamlit Cloud, Hugging Face Spaces, or custom cloud setups.

### Deployment
Once the app is running locally, you can deploy it on Streamlit Cloud or other hosting platforms like Hugging Face Spaces. Follow the respective platform’s guidelines for deployment.
Here is the link of my app: https://chat-with-multiple-pdf-s-db22kbnsjspssb.streamlit.app/

### Example Deployment on Streamlit Cloud:
- Push the code to GitHub.
- Create an account on Streamlit Cloud.
- Link your GitHub repository and deploy the app directly from there.
