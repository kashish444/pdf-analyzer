import os
import pickle
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.chains.question_answering import load_qa_chain
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS, VectorStore
from langchain.llms import OpenAI

# Load environment variables from .env file
load_dotenv()

# Retrieve OpenAI API key from environment variables
apikey = os.getenv('openai_api_key')

# Sidebar contents
with st.sidebar:
    st.title('🤗💬 PDF ANALYZER')
    add_vertical_space(5)

def main():
    st.header("Chat with PDF 💬")

    # Upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        # Extract text from the PDF
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Split the text into chunks for processing
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        # Use the PDF file name as the vector store name
        store_name = pdf.name[:-4]
        st.write(f'{store_name}')

        # Check if vector store file already exists
        if os.path.exists(f"{store_name}.pkl"):
            # Load vector store from file
            with open(f"{store_name}.pkl", "rb") as f:
                vectorstore = pickle.load(f)
        else:
            # Create OpenAI embeddings
            embeddings = OpenAIEmbeddings(openai_api_key=apikey)
            # Create vector store from text chunks
            vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
            # Save vector store to file
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(vectorstore, f)

        # Accept user questions/query
        query = st.text_input("Ask questions about your PDF file:")

        if query:
            # Perform similarity search on vector store
            docs = vectorstore.similarity_search(query=query, k=3)
            # Create OpenAI language model
            llm = OpenAI(openai_api_key=apikey)
            # Load question answering chain
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            # Generate response using question answering chain
            response = chain.run(input_documents=docs, question=query)
            st.write(response)

if __name__ == '__main__':
    main()
