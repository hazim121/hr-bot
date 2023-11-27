import streamlit as st

st.title("HR Bot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
   
#Function to get response from model
def model_bot(prompt):
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    from transformers import GPT2TokenizerFast
    from langchain.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.vectorstores import FAISS
    from langchain.chains.question_answering import load_qa_chain
    from langchain.llms import OpenAI
    from langchain.chains import ConversationalRetrievalChain
    import pypdf



    loader = PyPDFLoader("pages/Employee Handbook.pdf")  

    text = loader.load_and_split()
        
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    
    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text))
    
    text_splitter = RecursiveCharacterTextSplitter(
  
        chunk_size = 512,
        chunk_overlap  = 24,
        length_function = count_tokens,
    )
    
    chunks = text_splitter.create_documents([text])
    
    os.environ["OPENAI_API_KEY"] = "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"
    
    # Embed text and store embeddings
    # Get embedding model
    embeddings = OpenAIEmbeddings()  
    # Create vector database
    db = FAISS.from_documents(chunks, embeddings)
    
    chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")
    query = prompt    

    docs = db.similarity_search(query) 

    ans=chain.run(input_documents=docs, question=query)  


    return ans

# React to user input
if prompt := st.chat_input("Hi! How can i help you?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = model_bot(prompt)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
