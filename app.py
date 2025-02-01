import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA

# Streamlit app title
st.title("DeepSeek-PDF-Assistant")

# Load the PDF
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())

    # Load the PDF
    loader = PDFPlumberLoader("temp.pdf")
    docs = loader.load()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    documents = text_splitter.split_documents(docs)

    # Instantiate the embedding model with explicit model name
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    # Create the vector store and fill it with embeddings
    vector = FAISS.from_documents(documents, embedder)
    retriever = vector.as_retriever(search_kwargs={"k": 3})

    try:
        llm = Ollama(
            model="deepseek-r1",
            base_url="http://localhost:11434",
        )
    except Exception as e:
        st.error(f"Error connecting to Ollama: {str(e)}")
        st.stop()

    # Define the prompt
    prompt = """
    1. Use the following pieces of context to answer the question at the end.
    2. If you don't know the answer, just say that "I don't know" but don't make up an answer on your own.\n
    3. Keep the answer crisp and limited to 3,4 sentences.
    Context: {context}
    Question: {question}
    Helpful Answer:"""

    QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt)

    llm_chain = LLMChain(
        llm=llm,
        prompt=QA_CHAIN_PROMPT,
        callbacks=None,
        verbose=True)

    document_prompt = PromptTemplate(
        input_variables=["page_content", "source"],
        template="Context:\ncontent:{page_content}\nsource:{source}",
    )

    combine_documents_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context",
        document_prompt=document_prompt,
        callbacks=None)

    qa = RetrievalQA(
        combine_documents_chain=combine_documents_chain,
        retriever=retriever,
        verbose=True,
        return_source_documents=True)

    # User input
    user_input = st.text_input("Ask a question related to the PDF:")

    # Process user input
    if user_input:
        with st.spinner("Processing..."):
            # Use invoke instead of run and handle the response properly
            response = qa.invoke(user_input)
            st.write("Response:")
            st.write(response["result"])
            st.write("\nSources:")
            for doc in response["source_documents"]:
                st.write(f"- {doc.metadata.get('source', 'Unknown')}")
else:
    st.write("Please upload a PDF file to proceed.")
