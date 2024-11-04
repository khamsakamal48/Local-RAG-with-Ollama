# Importing necessary libraries for the app
import streamlit as st                                                                                                  # Web app framework
import os                                                                                                               # For file and directory operations
import tempfile                                                                                                         # For creating temporary files and directories
import shutil                                                                                                           # For file operations, like copying and deleting
import pdfplumber                                                                                                       # Library for extracting text and images from PDF files
import ollama                                                                                                           # LLM API for generating responses and embeddings
import chromadb                                                                                                         # ChromaDB for storing and querying embeddings

# Importing specific modules and classes
from chromadb.api.client import SharedSystemClient                                                                      # Manages system cache for ChromaDB
from httpx import ConnectError                                                                                          # To catch error when Ollama is not present
from langchain_community.document_loaders import UnstructuredPDFLoader                                                  # PDF loader for document processing
from langchain_text_splitters import RecursiveCharacterTextSplitter                                                     # Text splitter for document chunking
from typing import List, Tuple, Dict, Any                                                                               # Type hinting for functions
from PyPDF2 import PdfReader                                                                                            # PDF reading library for parsing PDF content
from sqlite3 import OperationalError                                                                                    # Handle error where query output is less than expected

# Function to extract model names from the provided model information
@st.cache_resource
def extract_model_names(models_info: Dict[str, List[Dict[str, Any]]]) -> Tuple[str, ...]:
    """
    Extracts model names from a dictionary of model information.

    Args:
        models_info (Dict[str, List[Dict[str, Any]]]): Dictionary containing model details with keys as model attributes.

    Returns:
        Tuple[str, ...]: Tuple of model names.
    """
    model_names = tuple(model['name'] for model in models_info['models'])                                               # Extracting model names into a tuple
    return model_names                                                                                                  # Returning tuple of model names


def chat_generate(model, prompt, options, context=None):
    """
    Generates a response from the LLM model based on the provided prompt and context.

    Args:
        model (str): Name of the LLM model to use.
        prompt (str): Prompt to send to the model.
        options (dict): Options for model generation, such as temperature and max length.
        context (str, optional): Additional context from the uploaded document.

    Returns:
        str: Generated response from the model.
    """

    if context:
        # Including context if available to generate response using the LLM model
        response = ollama.generate(model=model, prompt=f'Using this data: {context}. Respond to this prompt: {prompt}',
                                   stream=False, keep_alive=0, options=options)

    else:
        # Generating response without context
        response = ollama.generate(model=model, prompt=prompt, stream=False, keep_alive=0, options=options)

    # Returning the generated response
    return response['response']

# Function to parse PDF content into plain text
def parse_pdf(file):
    """
    Parses text from a PDF file.

    Args:
        file: Uploaded PDF file.

    Returns:
        str: Extracted text from the PDF.
    """

    # Initialize PdfReader with the PDF file
    reader = PdfReader(file)

    # Concatenate text from all pages
    text = ''.join(page.extract_text() for page in reader.pages)

    # Return extracted text
    return text

# Function to extract each page of a PDF as an image using pdfplumber
@st.cache_data
def extract_all_pages_as_images(file_upload) -> List[Any]:
    """
    Extracts all pages of a PDF as images.

    Args:
        file_upload: Uploaded PDF file.

    Returns:
        List[Any]: List of page images extracted from the PDF.
    """

    # Open PDF file
    with pdfplumber.open(file_upload) as pdf:
        # Convert each page to an image and return list
        return [page.to_image().original for page in pdf.pages]


# Function to initialize or reinitialize vector database based on conditions
def init_vector_db(embedding_model, file_upload, chunk_size, chunk_overlap, llm_params) -> chromadb.Collection:
    """
    Initializes or reinitialize the vector database for the uploaded PDF and selected embedding model.

    Args:
        embedding_model (str): Selected embedding model.
        file_upload (File): Uploaded PDF file.
        chunk_size (int): Size of text chunks to create embeddings.
        chunk_overlap (int): Overlap size for splitting text chunks.
        llm_params (dict): Parameters for the language model.

    Returns:
        chromadb.Collection: Collection with embedded vectors for querying.
    """

    # Set default values in session state
    st.session_state.setdefault('vector_db', None)
    st.session_state.setdefault('embedding_model', None)
    st.session_state.setdefault('uploaded_file', None)
    st.session_state.setdefault('chunk_size', None)
    st.session_state.setdefault('chunk_overlap', None)
    st.session_state.setdefault('llm_params', None)

    # Check if we need to reinitialize the vector_db
    is_new_document = file_upload and file_upload != st.session_state['uploaded_file']
    is_new_model = embedding_model and embedding_model != st.session_state['embedding_model']
    is_new_chunk_params = (chunk_size != st.session_state.get('chunk_size') or
                           chunk_overlap != st.session_state.get('chunk_overlap'))
    is_new_llm_params = llm_params and llm_params != st.session_state['llm_params']

    # Reinitialize if there's a new document, new embedding model, or updated chunk parameters
    if (is_new_document or is_new_model or is_new_chunk_params or is_new_llm_params or
            st.session_state['vector_db'] is None):
        # Update session state with the new file, model, and chunk parameters
        st.session_state['uploaded_file'] = file_upload
        st.session_state['embedding_model'] = embedding_model
        st.session_state['chunk_size'] = chunk_size
        st.session_state['chunk_overlap'] = chunk_overlap
        st.session_state['llm_params'] = llm_params

        # Clear the existing vector_db if any
        if st.session_state['vector_db'] is not None:
            del st.session_state['vector_db']

        # Create a temporary directory to save the uploaded file temporarily
        temp_dir = tempfile.mkdtemp()

        try:
            # Path for saving uploaded file
            path = os.path.join(temp_dir, file_upload.name)

            # Write file data to temp file
            with open(path, 'wb') as f:
                f.write(file_upload.getvalue())

            # Load and process the document
            loader = UnstructuredPDFLoader(path)

            # Load data from PDF
            data = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

            # Split document into chunks for embedding
            chunks = text_splitter.split_documents(data)

            # Extract text from each chunk
            docs = [chunk.page_content for chunk in chunks]

            # Clear ChromaDB system cache
            SharedSystemClient.clear_system_cache()  # Clear any previous cache

            # Initialize ChromaDB client
            client = chromadb.Client()

            # Set collection name for vector DB
            collection_name = 'docs'

            # Drop and recreate the collection to match the new embedding model's dimensionality
            if collection_name in [coll.name for coll in client.list_collections()]:
                client.delete_collection(collection_name)

            # Create new collection
            collection = client.create_collection(name=collection_name)

            # Generate embeddings and populate the collection
            for i, chunk in enumerate(docs):
                response = ollama.embeddings(model=embedding_model, prompt=chunk)
                embedding = response['embedding']
                collection.add(ids=[str(i)], embeddings=[embedding], documents=[chunk])

            # Set vector_db in session state to the new collection
            st.session_state['vector_db'] = collection
            st.toast('Vector DB created successfully for the new document, model, or chunk size change')

        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir)

    # Return vector DB collection
    return st.session_state['vector_db']


# Main function to run the Streamlit app
def main() -> None:
    """
    Main function to run the Retrieval Augmented Generation (RAG) app using Streamlit and Ollama.
    Allows users to upload a PDF, select models, and interact with an AI assistant.
    """

    # Set the app title and divider in the main page
    st.title('🤖 Retrieval Augmented Generation with Ollama', anchor=False)
    st.divider()

    # Initialize default session state variables if they don’t exist
    st.session_state.setdefault('vector_db', None)
    st.session_state.setdefault('uploaded_file', None)
    st.session_state.setdefault('parsed_content', "")
    st.session_state.setdefault('pdf_pages', [])
    st.session_state.setdefault('embedding_model', None)
    st.session_state.setdefault('llm_params', None)

    # Sidebar for model and parameter selection
    with st.sidebar:
        st.header('Model Selection')

        # List available models
        try:
            models_info = ollama.list()

        # Handle error when Ollama is not installed or accessible over default port
        except ConnectError:
            st.error('Ollama not installed or not accessible.')
            st.stop()

        # Extract model names for selection
        available_models = extract_model_names(models_info)

        # LLM model selection
        llm_model = st.sidebar.selectbox('Select LLM model', available_models)

        st.divider()
        st.subheader('Document Embedding and Splitting')
        st.caption('Will work only after the document has been uploaded')

        # Embedding model selection
        embedding_model = st.sidebar.selectbox("Select embedding model", available_models)
        chunk_size = st.sidebar.slider('Chunk size', min_value=1, max_value=10000, value=1500, step=500)
        overlap_size = st.sidebar.slider('Overlap size', min_value=1, max_value=1000, value=100)

        # Sidebar for LLM parameter adjustments
        st.divider()
        st.subheader('LLM Parameters')
        llm_params = {
            "temperature": st.sidebar.slider('Temperature', 0.0, 1.0, 0.1),
            "top_p": st.sidebar.slider('Top P', 0.0, 1.0, 0.9),
            "max_length": st.sidebar.slider('Max Length', 20, 500, 50)
            # Additional parameters can be added as needed
        }

    # Divide layout into two columns for PDF and parsed data views
    col1, col2 = st.columns(2)

    # Left column: Upload PDF and display PDF pages
    with col1:
        st.subheader('Load PDF')
        uploaded_file = st.file_uploader('Upload a PDF file', type='pdf', label_visibility='collapsed')

        if uploaded_file:
            # Parse PDF and display pages if a new file is uploaded
            if uploaded_file != st.session_state['uploaded_file']:
                with st.spinner("Parsing PDF..."):
                    st.session_state['parsed_content'] = parse_pdf(uploaded_file)

                # Extract PDF pages as images
                pdf_pages = extract_all_pages_as_images(uploaded_file)
                st.session_state['pdf_pages'] = pdf_pages

            # Initialize or reinitialize vector database for the new document or model
            init_vector_db(embedding_model, uploaded_file, chunk_size, overlap_size, llm_params)

        # Display each page image with adjustable zoom level
        if uploaded_file and st.session_state['pdf_pages']:
            zoom_level = col1.slider("Zoom Level", min_value=100, max_value=1000, value=700, step=50)
            with st.container(height=410, border=True):
                for page_image in st.session_state['pdf_pages']:
                    st.image(page_image, width=zoom_level)

    # Right column: Display parsed PDF text
    with col2:
        if uploaded_file:
            st.subheader('Parsed Text Content')

            if uploaded_file and st.session_state['parsed_content']:
                with st.container(height=660, border=True):
                    st.markdown(st.session_state['parsed_content'])

    # User input and response generation with the selected model
    st.divider()

    # Adds a subheader for the chat interface with the AI assistant
    st.subheader('Chat with AI')

    # Default messages displayed to the user based on whether a document is uploaded or not
    default_message = 'Hello there! 👋🏻 \n\n I am a helpful AI assistant!'
    default_message_with_doc = ('Hello there! 👋🏻 \n\n I am a helpful AI assistant! You can ask me anything '
                                'about the document you just uploaded.')

    # Creates a container to display chat messages, with a fixed height of 500 pixels
    messages = st.container(height=300)

    # Block to handle chat messages from the user and AI responses
    with st.chat_message('user'):
        # Checks if a PDF file has been uploaded
        if uploaded_file:

            # AI's initial message if a document is uploaded
            messages.chat_message('ai').write(default_message_with_doc)

        else:
            # AI's initial message if no document is uploaded
            messages.chat_message('ai').write(default_message)

        # Checks if there is any input from the user in the chat input box
        if prompt := st.chat_input('Say something'):
            # Displays the user's input in the chat
            messages.chat_message('human').write(prompt)

            # If a document is uploaded, uses the embedding model for querying
            if uploaded_file:
                # Embeds the user’s query prompt to generate a vector
                response = ollama.embeddings(prompt=prompt, model=embedding_model)

                # Queries the vector database to retrieve the top 3 relevant documents
                try:
                    results = st.session_state['vector_db'].query(query_embeddings=[response['embedding']], n_results=3)

                except OperationalError:
                    results = st.session_state['vector_db'].query(query_embeddings=[response['embedding']], n_results=1)

                # Gets the content of the most relevant document for context
                context_emb = results['documents'][0][0]

                # Generates a response from the AI model using the context from the retrieved document
                messages.chat_message('ai').write(chat_generate(llm_model, prompt, llm_params, context_emb))

            else:
                # Generates a response from the AI model without any document context
                messages.chat_message('ai').write(chat_generate(llm_model, prompt, llm_params))

# Configures the main page with title, icon, and layout settings
st.set_page_config(page_title='LLM & RAG with Ollama', page_icon='🤖', layout='wide')

# CSS style to hide the main menu and footer for a cleaner UI
hide_style = '''
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
'''

# Applies the CSS styling to hide elements on the page
st.markdown(hide_style, unsafe_allow_html=True)

# Entry point to run the main function if this script is executed directly
if __name__ == "__main__":
    main()
