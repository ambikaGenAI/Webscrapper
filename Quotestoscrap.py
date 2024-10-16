from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain,ConversationChain
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import SystemMessage
import streamlit as st
from streamlit_chat import message

import os
#from dotenv import load_dotenv
#load_dotenv()
#os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"]=st.secrets["OPENAI_API_KEY"]

#step1: Scrap the data from web for the given url
url="https://quotes.toscrape.com/"
loader=WebBaseLoader(url)
web_data=loader.load()
#print(web_data)

#step 2: Break them into chunks of data
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
text_data=text_splitter.split_documents(web_data)
#print(len(text_data))

#step 3: Convert then into vector db and save chromadb
enbeddings=OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore=Chroma.from_documents(text_data,enbeddings)

#step 4: retrival of data into chatgpt
llm=ChatOpenAI(model="gpt-4o",temperature=0.7)
qa=ConversationalRetrievalChain.from_llm(llm=llm,
                                retriever=vectorstore.as_retriever(),
                                memory=ConversationBufferMemory(memory_key="chat_history",return_messages=True))

#step 5: pass a question
#query="what was the quote said by Albert Einstein?"
#response=qa(query)
#print(f"Human:{query}")
#print(f"AI:{response}")



st.set_page_config(page_title="Quotes to Scrap Chatbot",page_icon="🦖")
st.title("Scrap Data from Quotes to Scrap.com Chatbot")
st.subheader("Visit https://quotes.toscrape.com/ to verify")

if "buffer_memory" not in st.session_state:
    st.session_state.buffer_memory=[]

if "messages" not in st.session_state.keys():
    st.session_state.messages=[{'role' : 'Assistant','content': 'How May i help you today?'}]
    

if prompt:= st.chat_input("Ask a Question from quotestoscrap.com"):
     st.session_state.messages.append({"role": "user", "content": prompt})
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message['content'])

if st.session_state.messages[-1]["role"] != "Assistant":
    with st.chat_message("Assistant"):
        with st.spinner("Web Scrap retriving from quotes to scrap.com"):
            response=qa(prompt)
            st.write(response["answer"])
            message={ "role":"Assistant","content": response["answer"]}
            st.session_state.messages.append(message)
system_message=" You are a Chatbot who will answer only from Given url . you will not answer any other questions."
conversation=ConversationChain(llm=llm)
conversation.memory.chat_memory.add_message(SystemMessage(content=system_message))
