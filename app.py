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

if 'questions' not in st.session_state:
    st.session_state.questions = []
if 'responses' not in st.session_state:
    st.session_state.responses = []
if 'counter' not in st.session_state:
    st.session_state.counter = 0

st.title("Deepseek PDF Assistant")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())

    loader = PDFPlumberLoader("temp.pdf")
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    documents = text_splitter.split_documents(docs)

    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    vector = FAISS.from_documents(documents, embedder)
    retriever = vector.as_retriever(search_kwargs={"k": 3})

    try:
        llm = Ollama(
            model="deepseek-r1:1.5b",
            base_url="http://localhost:11434",
        )
    except Exception as e:
        st.error(f"Error connecting to Ollama: {str(e)}")
        st.stop()

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

    for i in range(len(st.session_state.questions)):
        st.write(f"Question: {st.session_state.questions[i]}")
        st.write("Response:")
        st.write(st.session_state.responses[i]["result"])
        st.write("\nSources:")
        for doc in st.session_state.responses[i]["source_documents"]:
            st.write(f"- {doc.metadata.get('source', 'Unknown')}")
        st.write("---")

    user_input = st.text_input("Ask a question related to the PDF:", key=f"input_{st.session_state.counter}")

    if user_input and user_input not in st.session_state.questions:
        with st.spinner("Processing..."):
            response = qa.invoke(user_input)
            st.session_state.questions.append(user_input)
            st.session_state.responses.append(response)
            st.session_state.counter += 1
            st.rerun()
else:
    st.write("Please upload a PDF file to proceed.")
