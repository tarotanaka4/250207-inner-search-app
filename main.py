from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

load_dotenv()

st.set_page_config(
    page_title="社内情報特化型生成AI検索アプリ"
)

st.markdown('## 社内情報特化型生成AI検索アプリ')

col1, col2 = st.columns([1, 3])

with col1:
    st.session_state.mode = st.radio(
        "回答モード",
        ['社内文書検索', '社内問い合わせ']
    )
    # st.session_state.mode = st.selectbox(label="モード", options=["社内文書検索", "社内問い合わせ"])
# st.divider()

with st.chat_message("assistant"):
    st.markdown("こちらは社内問い合わせ対応自動化の生成AIチャットボットです。上記の回答モードで「社内文書検索」か「社内問い合わせ」のどちらかを選択し、チャット欄から自由に質問してください。")
    st.markdown("**「社内文書検索」の場合の入力例**")
    st.info("2024年12月に行った、HealthXのマーケティング施策についてのMTG議事録")
    st.markdown("**「社内問い合わせ」の場合の入力例**")
    st.info("HealthXのプレミアムプランについて、具体的に教えて")

if "messages" not in st.session_state:
    st.session_state.messages = []

    embeddings = OpenAIEmbeddings()
    if os.path.isdir(".db"):
        db = Chroma(persist_directory=".db", embedding_function=embeddings)
    else:
        folder_name = "data"
        files = os.listdir(folder_name)

        docs = []
        for file in files:
            if file.endswith(".pdf"):
                loader = PyMuPDFLoader(f"{folder_name}/{file}")
            elif file.endswith(".docx"):
                loader = Docx2txtLoader(f"{folder_name}/{file}")
            else:
                continue
            pages = loader.load()
            docs.extend(pages)

        text_splitter = CharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=30,
            separator="\n",
        )
        splitted_pages = text_splitter.split_documents(docs)

        # client_settings = Settings(
        #     chroma_db_impl="duckdb+parquet",
        #     persist_directory=".db",
        #     anonymized_telemetry=False
        # )
        # client = chromadb.Client(client_settings)

        # db = Chroma.from_documents(splitted_pages, embedding=embeddings, persist_directory=".db", client=client)
        db = Chroma.from_documents(splitted_pages, embedding=embeddings, persist_directory=".db")
    retriever = db.as_retriever(search_kwargs={"k": 5})
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5)
    st.session_state.chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            if message["content"]["message_type"] == "document_search":
                st.markdown(message["content"]["main_message"])
                try:
                    st.success(f"{message['content']['main_choice']}（P.{message['content']['main_page_number']+1}）", icon=":material/description: ")
                except:
                    st.success(f"{message['content']['main_choice']}（ページ番号取得不可）", icon=":material/description: ")
                if message["content"]["sub_message"]:
                    st.markdown(message["content"]["sub_message"])
                    for sub_choice in message["content"]["sub_choices"]:
                        try:
                            st.info(f"{sub_choice['source']}（P.{sub_choice['page_number']+1}）", icon=":material/description:")
                        except:
                            st.info(f"{sub_choice['source']}（ページ番号取得不可）", icon=":material/description:")
            else:
                st.markdown(message["content"]["answer"])
                if message["content"]["message"]:
                    st.divider()
                    st.markdown(message["content"]["message"])
                    for file_path in message["content"]["file_path_list"]:
                        st.info(file_path, icon=":material/description:")

chat_message = st.chat_input("質問を入力してください。")

if chat_message:
    st.session_state.messages.append({"role": "user", "content": chat_message})
    with st.chat_message("user"):
        st.markdown(chat_message)

    result = st.session_state.chain.invoke(chat_message)
    with st.chat_message("assistant"):
        if st.session_state.mode == "社内文書検索":
            main_choice = result["source_documents"][0].metadata["source"]
            main_message = "入力内容のテーマに関する情報は、以下のファイルに存在する可能性が高いです。"
            st.markdown(main_message)

            content = {}
            content["main_message"] = main_message
            content["main_choice"] = main_choice
            
            try:
                main_page_number = result['source_documents'][0].metadata['page']
                content["main_page_number"] = main_page_number
                st.success(f"{main_choice}（P.{main_page_number+1}）", icon=":material/description:")
            except:
                st.success(f"{main_choice}（ページ番号取得不可）", icon=":material/description:")

            sub_choices = []
            duplicate_check_list = []
            for document in result["source_documents"][1:]:
                sub_file_path = document.metadata["source"]
                if sub_file_path in duplicate_check_list:
                    continue
                duplicate_check_list.append(sub_file_path)

                if sub_file_path == main_choice:
                    continue
                
                try:
                    sub_page_number = document.metadata["page"]
                    sub_choice = {"source": sub_file_path, "page_number": sub_page_number}
                except:
                    sub_choice = {"source": sub_file_path}
                sub_choices.append(sub_choice)
            
            # sub_choices = ["data/healthX_instructions.pdf", "data/20241207_MTG議事録_healthXのマーケティング施策について.docx"]
            if sub_choices:
                sub_message = "その他、候補を提示します。"
                st.markdown(sub_message)
                for sub_choice in sub_choices:
                    try:
                        st.info(f"{sub_choice['source']}（P.{sub_choice['page_number']+1}）", icon=":material/description:")
                    except:
                        st.info(f"{sub_choice['source']}（ページ番号取得不可）", icon=":material/description:")
                content["sub_message"] = sub_message
                content["sub_choices"] = sub_choices
            else:
                content["sub_message"] = ""
                content["sub_choices"] = []

            content["message_type"] = "document_search"
            st.session_state.messages.append({"role": "assistant", "content": content})
        elif st.session_state.mode == "社内問い合わせ":
            st.markdown(result["result"])

            content = {}
            content["message_type"] = "contact"
            content["answer"] = result["result"]

            file_path_list = []
            for document in result["source_documents"]:
                file_path = document.metadata["source"]
                if file_path in file_path_list:
                    continue
                file_path_list.append(file_path)
            
            
            if file_path_list:
                st.divider()
                message = "情報源"
                st.markdown(f"**{message}**")
                for file_path in file_path_list:
                    st.info(file_path, icon=":material/description:")
                content["message"] = message
                content["file_path_list"] = file_path_list
            else:
                content["message"] = ""
                content["file_path_list"] = []

            st.session_state.messages.append({"role": "assistant", "content": content})
