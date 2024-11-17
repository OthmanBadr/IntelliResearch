import streamlit as st
import mimetypes
import genai  # Assuming you have GenAI integrated already

# Set page configuration
st.set_page_config(page_title="IntelliResearch", layout="wide")

# Apply dark theme and custom CSS for the entire interface
st.markdown(
    """
    <style>
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
        transition: all 0.3s ease;
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
    .chat-input-container {
        display: flex;
        justify-content: space-between;
        padding: 10px;
    }
    .chat-input {
        background-color: #1E1E1E;
        border: 1px solid #444;
        color: white;
        padding: 10px;
        border-radius: 8px;
        width: 80%;
    }
    .submit-button {
        background-color: #1A73E8;
        border: none;
        border-radius: 8px;
        color: white;
        padding: 10px 15px;
        cursor: pointer;
    }
    .submit-button:hover {
        background-color: #155DB2;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar for file uploads
st.sidebar.title("Sources")

# File Upload Section in Sidebar
uploaded_files = st.sidebar.file_uploader(
    "Upload files to include as sources",
    type=["pdf"],
    accept_multiple_files=True
)

# Display uploaded files in the sidebar
if uploaded_files:
    st.sidebar.markdown("### Uploaded Files")
    for file in uploaded_files:
        st.sidebar.markdown(f"📄 {file.name}")

# Header
st.markdown(
    """
    <div style="background-color:#000000; padding:10px; border-radius:5px;">
        <h2 style="color:white; text-align:center;">IntelliResearch</h2>
        <p style="color:gray; text-align:center;">Chat with your PDFs and get suggested questions</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Initialize session state for chat if not already initialized
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Summarization Section UI (Visible even without uploaded files)
st.write("### Summarize PDFs")
st.write("Click the button below to summarize your uploaded PDFs (functionality will be added later).")

# Button for Summarization UI
summarize_button = st.button("Summarize PDFs")

if summarize_button and uploaded_files:
    # Call the summarization function here
    api_key_summary = "AIzaSyBnUydStaEeHFdz0-Ek0yBZVJ2YZF9iq1c"  # Replace with actual API key
    result = generate_collective_summary_and_questions([file.name for file in uploaded_files], api_key_summary)

    # Display the summary
    st.write("### Summary")
    st.write(result["summary"])

    # Display the generated questions
    st.write("### Generated Questions")
    for question in result["questions"]:
        st.write(f"- {question}")

# Suggested questions for the user to interact with the chatbot
st.write("### Suggested Questions")
suggested_questions = [
    "What is the summary of the document?",
    "Can you explain the main points of the document?",
    "What are the key findings in the document?",
    "Please provide a detailed summary of the document.",
    "What is the purpose of this document?"
]

for question in suggested_questions:
    st.markdown(f"- {question}")

# Main Content (Chat Section)
st.write("### Chat with IntelliResearch")

# Chat container for message display
chat_ui = st.empty()  # Placeholder for the chat container

# Display the chat container with message bubbles
with chat_ui:
    # Create a div for chat messages
    st.markdown(
        """
        <div class="chat-container">
        </div>
        """, unsafe_allow_html=True
    )

    # Display previous messages (chat history)
    for message in st.session_state.messages:
        if message["type"] == "user":
            st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-message">{message["content"]}</div>', unsafe_allow_html=True)

# Define the function to handle the chat input
def handle_user_input():
    user_message = st.session_state.chat_input.strip()
    if user_message:
        # Add user input to messages
        st.session_state.messages.append({"type": "user", "content": user_message})

        # Placeholder: Respond with a dummy answer
        # Replace this with your model integration
        response = f"I'm analyzing your question: '{user_message}' and generating a response from the uploaded files."

        # Add chatbot response to messages
        st.session_state.messages.append({"type": "bot", "content": response})

        # Clear input field after submission (Streamlit automatically clears after rerun)
        st.session_state.chat_input = ""  # Clear input box

# Create the input box and submit button
with st.form(key="chat_form", clear_on_submit=True):
    chat_input = st.text_area(
        "Ask a question based on your uploaded PDFs:",
        key="chat_input",
        placeholder="Type your question here...",
        height=100
    )
    submit_button = st.form_submit_button(label="Submit")
    if submit_button:
        handle_user_input()  # Handle user input when submit button is clicked
