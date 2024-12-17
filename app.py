from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import streamlit as st
import  re

# helper functions
def setup_db(embedding_model, encode_kwargs):
    loader = PyPDFLoader("knowledge_base.pdf")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    document = loader.load()
    texts = text_splitter.split_documents(document)
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model, encode_kwargs=encode_kwargs)
    db = FAISS.from_documents(texts, embeddings)
    return db

def get_retriever(search_type, search_kwargs, **db_kwargs):
    db = setup_db(**db_kwargs)
    retriever = db.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
    return retriever

def get_llm(hf_token, repo_id, model_kwargs):
    llm = HuggingFaceHub(
        repo_id=repo_id,
        model_kwargs=model_kwargs,
        huggingfacehub_api_token=hf_token
    )
    return llm

# initializations
embedding_model = "sentence-transformers/all-MiniLM-l6-v2"
hf_token = st.secrets["huggingfacehub_api_token"]
repo_id = "huggingfaceh4/zephyr-7b-alpha"
llm_model_kwargs = {"temperature":0.5, "max_length":64, "max_new_tokens":512}
prompt_template = """
Answer the question using the context below as an AI Assistant. Be clear and concise. If the context does not contain enough information to answer the question, respond with:
"Sorry, I didn’t understand your question. Do you want to connect with a live agent?"

Context: {context}
Question: {question}
Answer:
"""

retriever = get_retriever("similarity", {"k":4}, embedding_model=embedding_model, encode_kwargs={'normalize_embeddings':False})
llm = get_llm(hf_token, repo_id, llm_model_kwargs)
prompt = PromptTemplate(template=prompt_template, input_variables=["context","question"])
qa = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=retriever,
    return_source_documents=False, chain_type_kwargs={"prompt":prompt}
)

def get_answer(question):
    result = qa.invoke({"query":question})["result"]
    match = re.search(r"Answer:\s*(.*)", result, re.DOTALL)
    answer = match.group(1).strip()
    if "context" in answer:
        # Return a fallback message
        return "Sorry, I didn’t understand your question. Do you want to connect with a live agent?"
    return answer

# streamlit interface
st.title("Campus Companion")
st.text("Chat with the NMIMS Student Resource Book")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Let me help you")

if question:
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role":"user", "content":question})

    response = get_answer(question)
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role":"assistant", "content":response})