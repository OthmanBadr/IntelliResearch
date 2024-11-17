import streamlit as st
import mimetypes

# Set page configuration
st.set_page_config(page_title="IntelliResearch", layout="wide")

# Apply dark theme and custom CSS
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
if uploaded_files:
    st.sidebar.markdown("### Uploaded Files")
    for file in uploaded_files:
        st.sidebar.markdown(f"📄 {file.name}")

# Main Section Header
st.markdown(
    """
    <div style="text-align: center; margin-top: 20px;">
        <h2 style="color:white;">IntelliResearch</h2>
        <p style="color:gray;">Chat with your PDFs and get insightful summaries and questions</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Summary Section
st.markdown("<div class='summary-box'><p class='summary-title'>Summary</p><p>Summaries of uploaded documents will appear here.</p></div>", unsafe_allow_html=True)

# Questions Section
st.markdown("<div class='questions-box'><p class='questions-title'>Generated Questions</p><ul><li>What are the main points of the document?</li><li>Can you summarize the findings?</li><li>What is the purpose of the document?</li></ul></div>", unsafe_allow_html=True)

# Chat Section
st.markdown("### Chat with IntelliResearch")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat Display
chat_ui = st.empty()
with chat_ui:
    for message in st.session_state.messages:
        if message["type"] == "user":
            st.markdown(f"<div class='user-message'>{message['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bot-message'>{message['content']}</div>", unsafe_allow_html=True)

# Handle Chat Input
def handle_input():
    user_input = st.session_state.input_text.strip()
    if user_input:
        st.session_state.messages.append({"type": "user", "content": user_input})
        bot_reply = f"Processing your query: {user_input}"  # Placeholder for model response
        st.session_state.messages.append({"type": "bot", "content": bot_reply})
        st.session_state.input_text = ""

# Chat Input
st.text_area("Enter your query:", key="input_text", on_change=handle_input, placeholder="Ask something about the uploaded files...")

