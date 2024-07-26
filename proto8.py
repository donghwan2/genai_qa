# [배포 시에만] chromadb의 sqlite3 버전 문제 해결 
# requirements.txt 에 pysqlite3-binary 추가
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# proto8 : StatGPT-2.0(이전대화 기억)
# 대화 history가 저장되고,
# 사용자가 미리 저장한 Chroma DB의 데이터를 이해한 상태에서,
# 사용자의 질문에 대한 답변을 "langchain.RetrievalQA"으로 답변"하는 챗봇(출처 제공)
# + 이전 대화를 기억해서 답변 생성

import streamlit as st
import os
import time

# api key
# from dotenv import load_dotenv
# load_dotenv()
import openai
openai.api_key = st.secrets["OPENAI_API_KEY"]

# llm : langchain.ChatOpenAI
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain_openai import OpenAIEmbeddings
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma


### ★★★ 헤드 ★★★
st.markdown("# 🔎제2의나라 GPT")


### st.session_state에 대화 내용 저장

# 모델 초기화 with st.session_state
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"     # LLM 모델 설정 : "gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"

# 대화 초기화 with st.session_state
if "messages" not in st.session_state:
    st.session_state.messages = []

# 역할(role)과 대화내용(content) key 초기화
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


### Chroma db에 임베딩된 데이터 저장

# data디렉토리 문서들 로드하기
directoryloader = DirectoryLoader('./data', loader_cls=TextLoader)   # glob="*.txt", 

def data_to_db(loader):
    documents = loader.load()
    # print(documents)

    # Split texts
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
    texts = text_splitter.split_documents(documents)

    # db(chroma db)를 저장할 디렉토리 지정
    persist_directory = 'db'

    # 임베딩
    # embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    embedding = OpenAIEmbeddings()

    # db에 임베딩된 데이터 저장
    db = Chroma.from_documents(
        documents=texts, 
        embedding=embedding, 
        persist_directory=persist_directory) 

    # db 초기화
    db.persist()
    db = None

    db = Chroma(
        persist_directory=persist_directory, 
        embedding_function=embedding)

    return db

db = data_to_db(directoryloader)

### 사용자 입력이 들어왔을 때, 사용자와 챗봇의 대화 생성

# ★★★ 사용자 인풋 창 ★★★
if input := st.chat_input("What is up?"):   # ★★★ 사용자 인풋 창 ★★★
    
    # 사용자 입력을 st.session_state에 저장
    st.session_state.messages.append({"role": "user", "content": input})   

    # ★★★ 사용자 아이콘 ★★★
    with st.chat_message("user"):     
        # ★★★ 사용자 입력을 마크다운으로 출력
        st.markdown(input)        
        print("input:", input, "\n")

    # 챗봇 대답
    # ★★★ 챗봇 아이콘 ★★★
    with st.chat_message("assistant"):

        # 챗봇 대답 생성 : 빈 placeholder 생성 후 한 줄씩 채워가기
        message_placeholder = st.empty()    # 빈 placeholder 생성
        full_response = ""
    
        # ＠＠＠ 챗봇 대답(full_response->str) 생성 모델링

        # 답변 생성 모델 : langchain.RetrievalQA
        # db : ChromaDB

        ## db에서 유사한 문서 찾기(결과를 k개 반환)
        retriever = db.as_retriever(search_kwargs={"k": 2})

        ## RetrievalQA 구성하기
        qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model='gpt-4o'),      # gpt-4o, gpt-3.5-turbo
            # OpenAI("gpt-3.5-turbo"), 
            chain_type="stuff", 
            retriever=retriever, 
            return_source_documents=True)

        def process_llm_response(llm_response):
            source_list = []
            print(llm_response['result'])
            print('\n\nSources:')
            for source in llm_response["source_documents"]:
                print(source.metadata['source'])
                source_list.append(source.metadata['source'])
            print("\n")
            return source_list
        
        def qa_bot(query):
            llm_response = qa_chain(query)
            source_list = process_llm_response(llm_response)
            return llm_response['result'], source_list

        full_response, source_list = qa_bot(input)

        # ＠＠＠ 

        # ★★★ full_response(전체 답변 string)을 화면에 출력하기
        message_placeholder.markdown(full_response)
        st.write("출처 : ", source_list[0].replace('data\\', '').replace('.txt', '').replace('./data/',''))   # 출처 화면에 표시

    # 생성된 챗봇 답변을 st.session_state에 저장
    st.session_state.messages.append({"role": "assistant", "content": full_response})

    # 챗봇 답변을 채팅 화면에 표시
    message_placeholder.markdown(full_response)
    print("session_state: \n", st.session_state['messages'], "\n")

    # 사용자 인풋들만 따로 파악
    input_data = [x['content'] for x in st.session_state['messages'][0::2]]
    print(input_data)

    # 사용자 input을 input_data.txt 파일에 저장하기
    if not os.path.exists('data/input_data.txt'):
        # 파일이 없으면 'write' 모드로 파일을 생성
        with open('data/input_data.txt', 'w', encoding='utf-8') as file:
            # input 입력
            file.write(input + "\n\n")
    else:
        # 파일이 있으면 'append' 모드로 파일 열기
        with open('data/input_data.txt', 'a', encoding='utf-8') as file:
            # input 입력
            file.write(input + "\n\n")
    
    # input 데이터 db에 추가하기
    textloader = TextLoader("./data/input_data.txt")   # state_of_the_union.txt
    db = data_to_db(textloader)
    time.sleep(0.1)














