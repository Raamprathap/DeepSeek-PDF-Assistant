import streamlit as st
import os
import zipfile
from langchain_community.document_loaders import (
    PDFPlumberLoader, TextLoader, CSVLoader, UnstructuredMarkdownLoader, PythonLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA

if 'questions' not in st.session_state:
    st.session_state.questions = []
if 'responses' not in st.session_state:
    st.session_state.responses = []
if 'counter' not in st.session_state:
    st.session_state.counter = 0

st.title("Deepseek File Assistant")

upload_type = st.selectbox("Select upload type", ["File Upload", "Folder Upload (ZIP)"])

def get_loader(file_path, file_type):
    file_type = file_type.lower()  # normalize extension
    if file_type == "pdf":
        return PDFPlumberLoader(file_path)
    elif file_type == "txt":
        return TextLoader(file_path)
    elif file_type == "csv":
        return CSVLoader(file_path)
    elif file_type == "md":
        return UnstructuredMarkdownLoader(file_path)
    elif file_type == "py":
        return PythonLoader(file_path)
    elif file_type in ["js", "java", "cpp", "c", "cs", "rb", "php", "go", "rs", "swift", "ts", "html", "css"]:
        return TextLoader(file_path)
    return None

documents = []
temp_dir = "temp_files"
os.makedirs(temp_dir, exist_ok=True)

if upload_type == "File Upload":
    uploaded_files = st.file_uploader("Upload files", type=[
        "pdf", "txt", "csv", "py", "md", "js", "java", "cpp", "c", "cs",
        "rb", "php", "go", "rs", "swift", "ts", "html", "css"
        ], accept_multiple_files=True)
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            loader = get_loader(file_path, uploaded_file.name.split('.')[-1])
            if loader:
                docs = loader.load()
                documents.extend(docs)
            else:
                st.write(f"No loader for {uploaded_file.name}")

elif upload_type == "Folder Upload (ZIP)":
    uploaded_zip = st.file_uploader("Upload ZIP file (containing folder)", type=["zip"])
    
    if uploaded_zip:
        zip_path = os.path.join(temp_dir, "uploaded_folder.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_zip.getvalue())

        extract_folder = os.path.join(temp_dir, "extracted_files")
        # Remove the folder if it already exists to avoid mixing previous extractions.
        if os.path.exists(extract_folder):
            import shutil
            shutil.rmtree(extract_folder)
        os.makedirs(extract_folder, exist_ok=True)

        # Extract ZIP contents
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_folder)
            st.write("ZIP file extracted successfully!")
        except Exception as e:
            st.error(f"Error extracting ZIP file: {e}")

        # Debug: list all files found
        for root, _, files in os.walk(extract_folder):
            st.write(f"Scanning folder: {root}")
            st.write(f"Found files: {files}")
            for file in files:
                # Skip macOS metadata files
                if file.startswith("._") or file == ".DS_Store":
                    st.write(f"Skipping metadata file: {file}")
                    continue

                file_path = os.path.join(root, file)
                loader = get_loader(file_path, file.split('.')[-1])
                if loader:
                    try:
                        docs = loader.load()
                        documents.extend(docs)
                    except Exception as e:
                        st.write(f"Error processing {file_path}: {e}")
                else:
                    st.write(f"No loader available for file: {file}")


if documents:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    documents = text_splitter.split_documents(documents)

    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector = FAISS.from_documents(documents, embedder)
    retriever = vector.as_retriever(search_kwargs={"k": 3})

    try:
        llm = Ollama(model="deepseek-coder:6.7b", base_url="http://localhost:11434")
    except Exception as e:
        st.error(f"Error connecting to Ollama: {str(e)}")
        st.stop()

    prompt = """
    1. Use the following pieces of context to answer the question at the end.
    2. If you don't know the answer, just say that "I don't know" but don't make up an answer on your own.
    3. Keep the answer crisp and limited to 3,4 sentences.
    Context: {context}
    Question: {question}
    Helpful Answer:"""

    QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt)
    llm_chain = LLMChain(llm=llm, prompt=QA_CHAIN_PROMPT, callbacks=None, verbose=True)
    document_prompt = PromptTemplate(input_variables=["page_content", "source"],
                                     template="Context:\ncontent:{page_content}\nsource:{source}")
    combine_documents_chain = StuffDocumentsChain(llm_chain=llm_chain,
                                                  document_variable_name="context",
                                                  document_prompt=document_prompt,
                                                  callbacks=None)
    qa = RetrievalQA(combine_documents_chain=combine_documents_chain,
                     retriever=retriever,
                     verbose=True,
                     return_source_documents=True)

    for i in range(len(st.session_state.questions)):
        st.write(f"Question: {st.session_state.questions[i]}")
        st.write("Response:")
        st.write(st.session_state.responses[i]["result"])
        st.write("\nSources:")
        for doc in st.session_state.responses[i]["source_documents"]:
            st.write(f"- {doc.metadata.get('source', 'Unknown')}")
        st.write("---")

    user_input = st.text_input("Ask a question related to the files:", key=f"input_{st.session_state.counter}")

    if user_input and user_input not in st.session_state.questions:
        with st.spinner("Processing..."):
            response = qa.invoke(user_input)
            st.session_state.questions.append(user_input)
            st.session_state.responses.append(response)
            st.session_state.counter += 1
            st.rerun()
else:
    st.write("Please upload files or a ZIP folder to proceed.")