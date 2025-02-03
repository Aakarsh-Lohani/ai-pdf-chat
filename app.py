import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS

# Get text from each pdf file and return them as single string after concatenation
def get_pdf_text(pdf_files):
    text=""
    for pdf in pdf_files:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

# Get text chunks from the raw text
def get_text_chunks(raw_text):
    text_splitter=CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks=text_splitter.split_text(raw_text)
    return chunks

# Create vector store(knowledge base) from the text chunks using FAISS and OpenAI embeddings
def get_vectorstore(text_chunks):
    embeddings=HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore =FAISS.from_texts(texts=text_chunks,embedding=embeddings)
    return vectorstore


def main():
    load_dotenv()
    st.set_page_config(page_title="AI PDF Chat App", page_icon=":books:")
    st.header("AI PDF Chat App :books:")

    st.text_input("Ask your question about your documents: ")
    
    with st.sidebar:
        st.subheader("Your Documents")
        pdf_files=st.file_uploader("Upload your PDF file and click on Process", type=["pdf"], accept_multiple_files=True)

        if st.button("Process"):
            with st.spinner("Processing the pdf(s)..."):
                # get pdf text to process
                with st.spinner("Processing the raw text..."):
                    raw_text = get_pdf_text(pdf_files)
                    

                # get the chunks
                with st.spinner("Processing the text chunks..."):
                    text_chunks=get_text_chunks(raw_text)
                    
                # create vector store
                with st.spinner("Creating the vector store..."):
                    vectorstore=get_vectorstore(text_chunks)
                    st.write(vectorstore)


if __name__=="__main__":
    main()