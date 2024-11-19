import streamlit as st
import matplotlib.pyplot as plt
import os
import fitz
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
from huggingface_hub import login
from semantic_router import Route
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
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline # Import the pipeline function

os.environ['GOOGLE_API_KEY'] = "AIzaSyDcbFtWlJXlRy-aAwteRmMY3wV1HHJ4Nfs"
os.environ['GOOGLE_API_KEY_1'] = "AIzaSyBq0ZNZL6n0CZJH6GKTnGBoy8WbZdJSrkA"
os.environ['GOOGLE_API_KEY_plot'] = "AIzaSyCY4wmYHP9bY1vvMCZ5A4aOjwQj2j6rIBU"
os.environ['gemni_key_Sammary'] = "AIzaSyCY4wmYHP9bY1vvMCZ5A4aOjwQj2j6rIBU"
os.environ['COHERE_API_KEY'] = "BmiASvQqYpBBIvhrI0LoJXTQmFxZQGpWasP6SrfL"

# Configure the Google API Key for genai
def configure_genai(api_key_summary):
    api_key = api_key_summary
    genai.configure(api_key=api_key)


# Function to generate a collective summary and questions for a list of PDFs
def generate_collective_summary_and_questions(pdf_list, api_key_summary=os.environ['gemni_key_Sammary']):

    parameters = {
        "temperature": 0.0,
        "top_p": 0.9,
        "top_k": 20,
    }

    configure_genai(api_key_summary)
    # Initialize the model
    model = genai.GenerativeModel('gemini-1.5-pro', generation_config=parameters)

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
        Tables and Figures: Represent all tables and figures in the correct format, ensuring each is referenced within the "Results and Findings" section.
        Clear Citations: Explicitly reference each paper when discussing methodologies, findings, and trends.
        No External Data: Only use content from the provided PDFs for information extraction and analysis.
        Note: Please avoid non-standard characters or LaTeX commands in the output. Maintain structured and clear formatting throughout the paper, and ensure any distinctions between papers or domains are explicitly noted.

        **Important Notes:**
        - **Language Consistency:** Use the dominant language of the pdf, so if the dominant language is English use English or if the the dominant language of the pdf is Arabic use Arabic.
        - **Language Consistency: Answer in Arabic when discussing Arabic content, and provide clear language tags for sections in other languages where necessary.
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
    if "## Generated Questions" in response_text:
        summary_text, questions_section = response_text.split("## Generated Questions", 1)
        questions = [q.strip() for q in questions_section.splitlines() if q.strip()]
    else:
        summary_text = response_text  # Assume the whole response is the summary
        questions = []  # No questions were generated

    # Return the structured result
    return {
        "summary": summary_text.strip(),
        "questions": questions
    }
def load_ieee_papers_ocr(pdf_files):
    """
    Extracts text from PDF files using OCR for pages without text.
    """
    all_text_with_page_numbers = []
    for pdf_file in pdf_files:
        pdf_name = os.path.basename(pdf_file)
        pdf_reader = PdfReader(open(pdf_file, "rb"))
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()

            if not page_text.strip():
                page_images = convert_from_path(pdf_file, first_page=page_num + 1, last_page=page_num + 1, dpi=300)
                for page_image in page_images:
                    page_text = pytesseract.image_to_string(page_image)
                    break

            all_text_with_page_numbers.append({
                "pdf_name": pdf_name,
                "page_number": page_num + 1,
                "text": page_text
            })
    return all_text_with_page_numbers
def get_pages_with_figures(pdf_path, output_dir="screenshots", zoom=2.0):
    """
    Capture screenshots of pages containing figures from a PDF.

    Parameters:
    - pdf_path: Path to the PDF file.
    - output_dir: Directory where screenshots will be saved.
    - zoom: Zoom factor for increasing image resolution (default is 2.0).

    Returns:
    - pages_with_figures: List of page numbers containing figures.
    - total_pages: Total number of pages in the PDF.
    """
    try:
        doc = fitz.open(pdf_path)
        pages_with_figures = []
        os.makedirs(output_dir, exist_ok=True)

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            image_list = page.get_images(full=True)
            print(f"Page {page_num + 1}: Found {len(image_list)} images.")  # Debugging

            if image_list:
                pages_with_figures.append(page_num + 1)
                matrix = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=matrix)

                image_path = os.path.join(output_dir, f"{os.path.basename(pdf_path)}_page_{page_num + 1}.png")
                pix.save(image_path)
                print(f"Saved image for page {page_num + 1}: {image_path}")

        print(f"Total pages with figures: {len(pages_with_figures)}")
        return pages_with_figures, len(doc)
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return [], 0
def analyze_image(image_path, api_key_1):
    """
    Analyzes the provided image to extract data information and generates a plot.

    Parameters:
    - image_path (str): The file path of the image to analyze.
    - api_key_1 (str): The API key for the generative AI model.

    Returns:
    - (response.text, number_of_figures): The generated content based on the analysis of the image and the number of figures found.
    """
    try:
        # Log the image path being processed
        print(f"Analyzing image: {image_path}")

        # Example interaction with the generative model
        genai.configure(api_key=api_key_1)
        model = genai.GenerativeModel('gemini-1.5-pro')

        # Upload the image and get the file reference
        mime_type = "image/png"
        file_ref = genai.upload_file(image_path, mime_type=mime_type)

        # Define the prompt for image analysis
        prompt = """
        Analyze the provided image and extract all information for replotting each figure or table.
        Include details such as data points, axis labels, title, legend, and styling.
        """
        response = model.generate_content([file_ref, prompt])

        print(f"Analysis result for {image_path}: {response.text}")
        return response.text, response.text.count("Figure")  # Adjust as necessary
    except Exception as e:
        print(f"Error analyzing image {image_path}: {e}")
        return None, 0
def analyze_images_in_directory(directory_path, api_key_1=os.environ['GOOGLE_API_KEY_1']):
    """
    Iterates over a directory of images, analyzing each image.

    Returns:
    - extracted_data: A list of dictionaries with filenames and corresponding analyzed data.
    - total_pages: Total number of pages analyzed.
    """
    try:
        if not os.path.exists(directory_path):
            print(f"Directory {directory_path} does not exist.")
            return [], 0

        extracted_data = []
        total_pages = 0
        filenames = sorted(
            [f for f in os.listdir(directory_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))],
            key=lambda x: int(x.split("_page_")[1].split(".")[0]) if "_page_" in x else float('inf')
        )

        if not filenames:
            print("No image files found in the directory.")
            return [], 0

        for filename in filenames:
            image_path = os.path.join(directory_path, filename)
            print(f"Analyzing image: {filename}")

            try:
                data, pages = analyze_image(image_path, api_key_1)
                total_pages += pages
                extracted_data.append({"filename": filename, "image_data": data})
            except Exception as e:
                print(f"Error analyzing {filename}: {e}")

        return extracted_data, total_pages
    except Exception as e:
        print(f"Error in analyze_images_in_directory: {e}")
        return [], 0
def combine_text_and_images(pdf_files, output_dir="screenshots", api_key_1=None, chunk_size=10):
    """
    Combine text and image data from a list of PDF files into a structured format, divide it into chunks,
    and index the combined data using a vector store.

    Parameters:
    - pdf_files: List of PDF file paths.
    - output_dir: Directory for storing output screenshots.
    - api_key_1: API key for image analysis if required.
    - chunk_size: The number of entries in each chunk.

    Returns:
    - retriever: A retriever object for querying the indexed data.
    """
    # Set up embeddings
    embd = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
    combined_data = []

    try:
        # Step 1: Extract text data from PDFs
        text_data = load_ieee_papers_ocr(pdf_files)  # Replace with your text extraction function

        # Step 2: Extract images and analyze them
        for pdf_file in pdf_files:
            pdf_name = os.path.basename(pdf_file)
            pages_with_figures, total_pages = get_pages_with_figures(pdf_file, output_dir)  # Replace with your function
            print(f"{pdf_name}: {len(pages_with_figures)} pages with figures out of {total_pages} total pages.")

            # Analyze images in the output directory
            image_data, _ = analyze_images_in_directory(output_dir, api_key_1)  # Pass `api_key_1` explicitly
            print(f"Image data extracted: {len(image_data)} items.")

            # Step 3: Combine text and image data
            for page_num in range(1, total_pages + 1):
                text_entry = next((entry for entry in text_data if entry['pdf_name'] == pdf_name and entry['page_number'] == page_num), None)
                image_entry = next((item for item in image_data if f"_page_{page_num}" in item['filename']), None)

                combined_entry = {
                    "pdf_name": pdf_name,
                    "page_number": page_num,
                    "text": text_entry['text'] if text_entry else "No text found",
                    "image_data": image_entry['image_data'] if image_entry else "No image data"
                }
                combined_data.append(combined_entry)

        # Step 4: Divide combined data into chunks
        chunks = [
            combined_data[i:i + chunk_size]
            for i in range(0, len(combined_data), chunk_size)
        ]

        print(f"Combined data split into {len(chunks)} chunks.")

        # Step 5: Convert combined data to Document objects
        documents = []
        for entry in combined_data:
            doc_text = (
                f"PDF Name: {entry['pdf_name']}\n"
                f"Page Number: {entry['page_number']}\n"
                f"Text: {entry['text']}\n"
                f"Image Data: {entry['image_data']}"
            )
            # Create Document objects with a page_content attribute
            documents.append(Document(page_content=doc_text, metadata={"pdf_name": entry['pdf_name'], "page_number": entry['page_number']}))

        # Step 6: Split documents into smaller chunks for indexing
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size, chunk_overlap=0
        )
        doc_splits = text_splitter.split_documents(documents)

        print(f"Documents split into {len(doc_splits)} smaller chunks.")
        # Step 7: Add document splits to vectorstore
        vectorstore = Chroma.from_documents(
            documents=doc_splits,
            collection_name="rag-chroma",
            embedding=embd,
        )
        retriever = vectorstore.as_retriever()

        print("Indexing completed successfully.")
        return retriever
    except Exception as e:
        print(f"Error combining and indexing data: {e}")
        return None
import hashlib

def hash_image_data(image_data):
    """
    Compute a hash of the image data to identify unique images.
    """
    return hashlib.sha256(image_data.encode('utf-8') if isinstance(image_data, str) else image_data).hexdigest()
def image_plot(extracted_data, api_key_plot):
    api_key = api_key_plot
    genai.configure(api_key=api_key_plot)

    model = genai.GenerativeModel('gemini-1.5-pro')

    # Define prompt for generating Python code for plotting
    code_prompt = '''Based on the following data, generate Python code using matplotlib to plot a chart.
    Provide the full Python code, including comments explaining each step.
    Note1: Don't write code as a comment. Remove all special characters from the code.
    Note2: Don't write anything else; if there are multiple figures, write plot for each one.
    Note3: in the end of the code write plt.savefig("sine_wave_plot.png", dpi=300, bbox_inches='tight') this line to save the plot.
    '''
    text = extracted_data
    contents = [text, code_prompt]

    # Generate the plotting code
    code_response = model.generate_content(contents)
     # Save the generated response to a text file
    output_file_path = "generated_output.txt"
    with open(output_file_path, "w") as file:
        file.write(code_response.text)

    print(f"Response saved to {output_file_path}")

    # Execute the generated Python code
    with open(output_file_path) as file:
        lines = file.readlines()

    # Skip the first and last lines
    lines_to_execute = lines[1:-1]  # Adjust as necessary

    # Execute each line and handle errors
    for line in lines_to_execute:
        try:
            exec(line.strip())  # Ensure to strip whitespace
        except Exception as e:
            print(f"Error in line '{line.strip()}': {e}")
    st.image("/sine_wave_plot.png")
        
def generate_response(question: str, retriever, model, conversation_context: str = "", api_key_plot=None) -> str:
    """
    Generate a response using the provided model and context.
    """
    # Retrieve relevant documents
    retrieved_docs = retriever.invoke(question)

    # Track unique image hashes
    processed_images = set()

    # Process retrieved image data and plot it
    for doc in retrieved_docs:
        image_data = doc.metadata.get('image_data')
        if image_data:
            # Compute the hash for the current image
            image_hash = hash_image_data(image_data)

            # Skip if the image has already been processed
            if image_hash in processed_images:
                continue

            # Otherwise, process and add to the set of processed images
            processed_images.add(image_hash)
            try:
                image_plot(image_data, api_key_plot)
            except Exception as e:
                print(f"Error displaying image: {e}")

    # Prepare the context from the retrieved documents
    context = "\n".join([doc.page_content for doc in retrieved_docs])

    # Define the prompt template
    prompt_template = f"""
        You are a Deep Learning Expert. Answer the following question based on the context provided:

        {context}

        Question: {question}

        Display tables in proper table format where available.
        Include the page number(s) in the format {{page_num}} of {{pdf_name}} for any relevant information.
        If there is data suitable for plotting, generate a clear Python code for a matplotlib plot.
        Answer in the same language as the question.
    """

    # Generate text using the model
    response = model.generate_content(prompt_template)
    response_text = response.text

    # Collect references
    references = set()
    for doc in retrieved_docs:
        page_number = doc.metadata.get("page_number", "Unknown")
        pdf_name = doc.metadata.get("pdf_name", "Unknown PDF")
        references.add(f"{pdf_name}, Page {page_number}")

    # Append references to the response
    if references:
        response_text += "\nReferences: " + ", ".join(references)
    else:
        response_text += "\nReferences: None"

    return response_text
def chat_with_pdf(pdf_list,question,api_key_plot=os.environ['GOOGLE_API_KEY_plot'], max_history=5):
    """
    Manage a conversation loop with exit condition, storing a limited conversation history.
    """
    # Initialize conversation history
    conversation_history = deque(maxlen=max_history)

    # Compile conversation history into context
    conversation_context = "\n".join(conversation_history) if conversation_history else ""

    # Generate the response
    # Configure the Google GenAI model
    genai.configure(api_key=api_key_plot)
    model = genai.GenerativeModel('gemini-1.5-pro', generation_config={"temperature": 0.0, "top_p": 0.9, "top_k": 20})
    retriever = combine_text_and_images(pdf_list, chunk_size=512, api_key_1=os.environ['GOOGLE_API_KEY'])
    response = generate_response(question, retriever, model, conversation_context, api_key_plot)

    # Append question and response to conversation history
    conversation_history.append(f"Q: {question}\nA: {response}")

    # Display the answer
    print("Answer:", response)

    return response
def ask_ai_assistant(question):
    # key = open("key.txt", "r").read()
    # os.environ["GEMINI_API_KEY"] = key
    # genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    genai.configure(api_key="AIzaSyDXOyU0gGxUvX6kJfM9rCmndqvt3wrpLXI")
    model = genai.GenerativeModel('gemini-1.5-flash')
        # Define the structure prompt and question
    prompt = """
    You are an advanced AI assistant. Your purpose is to provide accurate, detailed, and well-referenced answers to questions.
    Very Important Note:Your responses must include:
    - Accurate and concise answers to the user's question.
    - References with full details such as:
    - Source name (e.g., book, article, or website).
    - Publication year (if available).
    - Direct links (URLs or DOIs) to the referenced material to enable verification.
    If you are unsure of a specific reference, indicate this clearly in your response.
    """
        # Combine the prompt and question
    input_text = f"{prompt}\n\n{question}"
    response = model.generate_content(input_text)
    return response.text





def Routing(pdf_list,question):
    """
    Route the question to the appropriate function based on its intent or the response.

    Args:
        question (str): The user's query.
        pdf_list (list): List of PDFs for analysis.
        structure_prompt (str): Prompt for AI Assistant.
        api_key_rag (str): API key for PDF retrieval agent.
        api_key_summary (str): API key for collective summary.

    Returns:
        None
    """
    # Define routes
    generate_figure_route = Route(
        name="generate_matplotlib_code",
        utterances=[
            "generate a chart", "visualize data", "show a plot", "plot data", "create a graph",
            "make a figure", "draw a chart", "display a plot", "render a visualization",
            "make a diagram", "show a scatter plot", "plot a line chart", "generate a bar chart",
            "plot a histogram", "create a heatmap", "illustrate data trends",
            "visualize data in a chart", "generate data visualization",
            "draw a figure for each data point", "display a chart for the figures",
            "create a graph of the data", "plot each figure from the data", "show figures and charts",
        ],
    )

    lama_route = Route(
        name="generate_lama_answer",
        utterances=[
            "The provided text does not contain any information",
            "Therefore, I cannot answer your question.",
            "The provided text does not contain the word",
            "The provided text is an excerpt from a research paper and does not contain information",
        ],
    )

    routes = [generate_figure_route, lama_route]

    # Initialize encoder and RouteLayer
    encoder = CohereEncoder(cohere_api_key=os.environ["COHERE_API_KEY"])  # Ensure you have a valid API key for Cohere
    layer = RouteLayer(encoder=encoder, routes=routes)

    # Step 2: Retrieve information from PDFs
    response = chat_with_pdf(pdf_list,question)
    print("PDF Agent Response:", response)
    # Step 3: Determine routing based on response
    route_name = layer(response).name
    # Call image_plot if the intent is for data visualization
    if route_name == "generate_matplotlib_code" or any(term in question.lower() for term in generate_figure_route.utterances):
        print("Plot Agent")
        directory_path="/screenshots"
        plt_data = analyze_images_in_directory(directory_path)
        image_plot(plt_data)
    # Call ask_ai_assistant if chat_with_pdf indicates insufficient information
    if route_name == "generate_lama_answer":
        print("Fallback to AI Assistant")
        response = ask_ai_assistant(question)
        print("AI Assistant Response:", response)
    # Handle unmatched routes
    else:
        print("No matching route found for the question.")
    return response


# Set page configuration
st.set_page_config(page_title="IntelliResearch", layout="wide")

# Apply dark theme and custom CSS
st.markdown(
    """
    <style>
    [Your custom CSS here, no changes needed]
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

# Initialize session state for chat messages
if "messages" not in st.session_state:
    st.session_state.messages = []

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
    api_key_summary = "AIzaSyDcbFtWlJXlRy-aAwteRmMY3wV1HHJ4Nfs"  # Replace with your actual API key
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

# Display the questions as clickable buttons
questions_placeholder.markdown(
    "<div class='questions-box'><p class='questions-title'>Generated Questions</p></div>",
    unsafe_allow_html=True
)

for i, question in enumerate(st.session_state["questions"]):
    if st.button(f"Q{i+1}: {question}"):
        # Add the clicked question to the chat messages and simulate user input
        st.session_state.messages.append({"type": "user", "content": question})
        bot_reply =Routing(pdf_paths,question)  # Replace with actual chatbot response
        st.session_state.messages.append({"type": "bot", "content": bot_reply})

# Chat Section
chat_ui = st.empty()
with chat_ui:
    for message in st.session_state.messages:
        if message["type"] == "user":
            st.markdown(f"<div class='user-message'>{message['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bot-message'>{message['content']}</div>", unsafe_allow_html=True)

# Handle user input from the text area
def handle_input():
    user_input = st.session_state.input_text.strip()
    if user_input:
        st.session_state.messages.append({"type": "user", "content": user_input})
        bot_reply = Routing(pdf_paths,question)   # Placeholder for model response
        st.session_state.messages.append({"type": "bot", "content": bot_reply})
        st.session_state.input_text = ""

st.text_area("Enter your query:", key="input_text", on_change=handle_input, placeholder="Ask something about the uploaded files...")
