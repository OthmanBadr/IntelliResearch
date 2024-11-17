
import os
import time
import torch
import getpass
import requests
import mimetypes
import pytesseract
import numpy as np
import pandas as pd
from PIL import Image
from langchain import hub
from PyPDF2 import PdfReader
from collections import deque
import matplotlib.pyplot as plt
from google.colab import drive
from huggingface_hub import login
from semantic_router import Routetz
from google.colab import userdata
import google.generativeai as genai
from langchain.schema import Document
from pdf2image import convert_from_path
from langchain.vectorstores import Chroma
from semantic_router.layer import RouteLayer
from vertexai.generative_models import Part
from langchain.prompts import PromptTemplate
from semantic_router.encoders import CohereEncoder
from sentence_transformers import SentenceTransformer
from langchain_google_genai import GoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline 

# Configure the Google API Key for genai
def configure_genai(api_key_summary):
    api_key = api_key_summary
    genai.configure(api_key=api_key)

# Function to generate a collective summary and questions for a list of PDFs
def generate_collective_summary_and_questions(pdf_list,api_key_summary):

    parameters = {
    "temperature": 0.0,
    "top_p":0.9,
    "top_k": 20,
    }

    configure_genai("AIzaSyBnUydStaEeHFdz0-Ek0yBZVJ2YZF9iq1c")
    # Initialize the model
    model = genai.GenerativeModel('gemini-1.5-pro',generation_config=parameters)

    # Define the prompt for analyzing and summarizing across all PDFs
    prompt = '''
     Please analyze the following Arabic and multilingual PDF documents to generate a structured survey paper. The paper should be comprehensive and well-organized, capturing all relevant information from each PDF. The structure of the paper should include the following sections:

        Title: A concise and informative title that accurately reflects the scope of the survey.

        Abstract: A high-level overview summarizing the main research topics, domains, and objectives covered in the survey.

        Keywords: List of relevant keywords capturing each unique domain or area discussed, ensuring all covered topics are reflected.

        Introduction: Present the primary research challenges and themes addressed across the papers. Provide a brief introduction to each domain if the scope covers multiple fields.

        Related Work: A thorough review of existing surveys or studies related to the topics in the PDF. Highlight the contributions of each document to its respective field, emphasizing distinctions between domains if multiple fields are involved.

        Methodologies and Approaches: A detailed explanation of the techniques, models, and methodologies used across studies. Organize this section by domain when multiple fields are present, ensuring clarity by explicitly referring to each methodology and its specific research area.

        Results and Findings: Summarize the key findings of each paper, including comparative analyses where relevant. When tables or figures are present, discuss them thoroughly, specifying the paper each result pertains to. Ensure any tables are formatted correctly and presented in table format for clarity.

        Discussion of Trends: An in-depth discussion on notable trends, common insights, and any key distinctions between domains, where applicable.

        Conclusion and Future Directions: Summarize the main conclusions from the survey and propose directions for future research, distinguishing between domains as needed.

        Please ensure the following:
        Language Consistency: Answer in Arabic when discussing Arabic content, and provide clear language tags for sections in other languages where necessary.
        Tables and Figures: Represent all tables and figures in the correct format, ensuring each is referenced within the "Results and Findings" section.
        Clear Citations: Explicitly reference each paper when discussing methodologies, findings, and trends.
        No External Data: Only use content from the provided PDFs for information extraction and analysis.
        Note: Please avoid non-standard characters or LaTeX commands in the output. Maintain structured and clear formatting throughout the paper, and ensure any distinctions between papers or domains are explicitly noted.

        **Important Notes:**
        - **Language Consistency:** Use the dominant language of the pdf, so if the dominant language is English use English or if the the dominant language of the pdf is Arabic use Arabic.
        - **No External Data:** Rely solely on the content of the uploaded PDFs.
        - **Standard Format:** Avoid non-standard characters or LaTeX commands.

        **Question Generation:**

        - After summarizing the PDF, generate 5 open-ended, in-depth questions suitable for academic or technical discussions based on the content.
          Each question should encourage exploration of alternative approaches to specific challenges addressed in the PDF, considering any relevant factors or constraints mentioned.
          Structure each question to:
        - Encourage critical evaluation of the approaches discussed in the PDF.
        - Explore how alternative methods, technologies, or frameworks might address the challenges highlighted.
        - Avoid numbering or bullets before each question and ensure the questions are written in the same language as the PDF content.

        **Important:**
        - Clearly separate the summary and the generated questions in your response.
        - Use headings or formatting to distinguish them (e.g., "## Summary" and "## Generated Questions")
    '''

    # Upload and collect references for each PDF
    file_references = []
    for pdf_path in pdf_list:
        mime_type, _ = mimetypes.guess_type(pdf_path)
        mime_type = mime_type or "application/pdf"
        file_ref = genai.upload_file(pdf_path, mime_type=mime_type)
        file_references.append(file_ref)

    # Pass all file references together with the prompt to the model
    contents = file_references + [prompt]
    response = model.generate_content(contents)

    # Extract summary and questions from the response
    response_text = response.text
    if ("## Generated Questions" in response_text) or ("## الأسئلة المُولّدة" in response_text):
        if ("## Generated Questions" in response_text):
            summary_text, questions_section = response_text.split("## Generated Questions", 1)
        else:
            summary_text, questions_section = response_text.split("## الأسئلة المُولّدة", 1)
        # Extract questions individually
        questions = [q.strip() for q in questions_section.splitlines() if q.strip()]
    else:
        # Handle case where separator is missing
        summary_text = response_text  # Assume the whole response is the summary
        questions = []  # No questions were generated

    # Return the structured result
    return {
        "summary": summary_text.strip(),
        "questions": questions
    }




import streamlit as st
import mimetypes
import os

# Import or define your `generate_collective_summary_and_questions` function here

# Set page configuration
st.set_page_config(page_title="IntelliResearch", layout="wide")

# Apply dark theme and custom CSS
st.markdown(
    """
    <style>
    /* Custom styles here */
    body {
        background-color: #000000;  /* Set background to black */
        color: white;  /* Set text color to white */
    }
    .stTextArea textarea {
        background-color: #1E1E1E;
        color: white;
        border: 1px solid #444;
    }
    .stButton > button {
        background-color: #1A73E8;
        color: white;
        border-radius: 5px;
        border: none;
    }
    .stButton > button:hover {
        background-color: #155DB2;
    }
    .sidebar .sidebar-content {
        background-color: #000000;  /* Sidebar dark theme */
        color: white;
    }
    .css-1r0h57r {
        background-color: #000000;  /* Change color of the header */
        color: white;
    }
    .stApp {
        background-color: #000000;  /* Set the whole app background color */
    }
    .summary-box, .questions-box {
        background-color: #1E1E1E;
        color: white;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        max-width: 800px;
        margin: auto;
    }
    .summary-title, .questions-title {
        font-weight: bold;
        font-size: 18px;
    }
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 10px;
        padding: 15px;
        max-width: 800px;
        margin: auto;
        background-color: #1E1E1E;
        border-radius: 8px;
        height: 500px;  /* Restrict height for scrolling */
        overflow-y: auto;
    }
    .user-message {
        background-color: #1A73E8;
        padding: 12px;
        border-radius: 8px;
        color: white;
        align-self: flex-end;
        max-width: 70%;
        word-wrap: break-word;
    }
    .bot-message {
        background-color: #444;
        padding: 12px;
        border-radius: 8px;
        color: white;
        align-self: flex-start;
        max-width: 70%;
        word-wrap: break-word;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar for file uploads
st.sidebar.title("Sources")
uploaded_files = st.sidebar.file_uploader(
    "Upload files to include as sources",
    type=["pdf"],
    accept_multiple_files=True
)

# Placeholder for summaries and questions
summary_placeholder = st.empty()
questions_placeholder = st.empty()

# Initialize session state for summaries and questions
if "summaries" not in st.session_state:
    st.session_state["summaries"] = ""
if "questions" not in st.session_state:
    st.session_state["questions"] = []

# Process uploaded PDFs
if uploaded_files:
    # Save uploaded files to temporary storage
    pdf_paths = []
    for uploaded_file in uploaded_files:
        pdf_path = os.path.join("temp", uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.read())
        pdf_paths.append(pdf_path)

    # Generate summary and questions
    st.write("Processing uploaded files...")
    api_key_summary = "AIzaSyBnUydStaEeHFdz0-Ek0yBZVJ2YZF9iq1c"  # Replace with your actual API key
    try:
        result = generate_collective_summary_and_questions(pdf_paths, api_key_summary)
        st.session_state["summaries"] = result["summary"]
        st.session_state["questions"] = result["questions"]
    except Exception as e:
        st.error(f"Error generating summary and questions: {e}")

# Display the summary
summary_placeholder.markdown(
    f"<div class='summary-box'><p class='summary-title'>Summary</p><p>{st.session_state['summaries']}</p></div>",
    unsafe_allow_html=True
)

# Display the questions
questions_placeholder.markdown(
    f"<div class='questions-box'><p class='questions-title'>Generated Questions</p><ul>{''.join([f'<li>{q}</li>' for q in st.session_state['questions']])}</ul></div>",
    unsafe_allow_html=True
)

# Chat Section (unchanged)
if "messages" not in st.session_state:
    st.session_state.messages = []

chat_ui = st.empty()
with chat_ui:
    for message in st.session_state.messages:
        if message["type"] == "user":
            st.markdown(f"<div class='user-message'>{message['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bot-message'>{message['content']}</div>", unsafe_allow_html=True)

def handle_input():
    user_input = st.session_state.input_text.strip()
    if user_input:
        st.session_state.messages.append({"type": "user", "content": user_input})
        bot_reply = f"Processing your query: {user_input}"  # Placeholder for model response
        st.session_state.messages.append({"type": "bot", "content": bot_reply})
        st.session_state.input_text = ""

st.text_area("Enter your query:", key="input_text", on_change=handle_input, placeholder="Ask something about the uploaded files...")
