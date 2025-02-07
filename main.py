"""
このファイルは、Webアプリのメイン処理が記述されたファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
from dotenv import load_dotenv
import logging
import streamlit as st
import utils
from initialize import initialize
import components as cn
import constants as ct


############################################################
# 設定関連
############################################################
# ブラウザタブの表示文言を設定
st.set_page_config(
    page_title=ct.APP_NAME
)

# 「.env」ファイルで定義した環境変数の読み込み
load_dotenv()

# ログ出力を行うためのロガーの設定
logger = logging.getLogger(ct.LOGGER_NAME)


############################################################
# 初期化処理
############################################################
try:
    # 初期化処理（「initialize.py」の「initialize」関数を実行）
    initialize()
except Exception as e:
    # エラーログの出力
    logger.error(f"{ct.INITIALIZE_ERROR_MESSAGE}\n{e}")
    # エラーメッセージの画面表示
    st.error(utils.build_error_message(ct.INITIALIZE_ERROR_MESSAGE), icon=ct.ERROR_ICON)
    # 後続の処理を中断
    st.stop()

# アプリ起動時のログファイルへの出力
if not "initialized" in st.session_state:
    st.session_state.initialized = True
    logger.info(ct.APP_BOOT_MESSAGE)


############################################################
# 初期表示
############################################################
# タイトル表示
cn.display_app_title()

# サイドバー表示
cn.display_sidebar()

# 初期表示のAIメッセージ
cn.display_initial_ai_message()


############################################################
# 会話ログの表示
############################################################
try:
    # 会話ログの表示
    cn.display_conversation_log()
except Exception as e:
    # エラーログの出力
    logger.error(f"{ct.CONVERSATION_LOG_ERROR_MESSAGE}\n{e}")
    # エラーメッセージの画面表示
    st.error(utils.build_error_message(ct.CONVERSATION_LOG_ERROR_MESSAGE), icon=ct.ERROR_ICON)
    # 後続の処理を中断
    st.stop()


############################################################
# チャット入力の受け付け
############################################################
chat_message = st.chat_input(ct.CHAT_INPUT_HELPER_TEXT)


############################################################
# チャット送信時の処理
############################################################
if chat_message:
    # ==========================================
    # 1. ユーザーメッセージの処理
    # ==========================================
    # ユーザーメッセージのログ出力
    logger.info({"message": chat_message, "application_mode": st.session_state.mode})

    # ユーザーメッセージを表示
    with st.chat_message("user"):
        st.markdown(chat_message)

    # ==========================================
    # 2. LLMからの回答取得
    # ==========================================
    # 「st.spinner」でグルグル回っている間、表示の不具合が発生しないよう空のエリアを表示
    res_box = st.empty()
    # LLMによる回答生成（回答生成が完了するまでグルグル回す）
    with st.spinner(ct.SPINNER_TEXT):
        try:
            # 画面読み込み時に作成したRetrieverを使い、Chainを実行
            llm_response = utils.get_llm_response(chat_message)
        except Exception as e:
            # エラーログの出力
            logger.error(f"{ct.GET_LLM_RESPONSE_ERROR_MESSAGE}\n{e}")
            # エラーメッセージの画面表示
            st.error(utils.build_error_message(ct.GET_LLM_RESPONSE_ERROR_MESSAGE), icon=ct.ERROR_ICON)
            # 後続の処理を中断
            st.stop()
    
    # ==========================================
    # 3. LLMからの回答表示
    # ==========================================
    with st.chat_message("assistant"):
        try:
            # ==========================================
            # モードが「社内文書検索」の場合
            # ==========================================
            if st.session_state.mode == ct.ANSWER_MODE_1:
                # 入力内容と関連性が高い社内文書のありかを表示
                content = cn.display_search_llm_response(llm_response)

            # ==========================================
            # モードが「社内問い合わせ」の場合
            # ==========================================
            elif st.session_state.mode == ct.ANSWER_MODE_2:
                # 入力に対しての回答と、参照した文書のありかを表示
                content = cn.display_contact_llm_response(llm_response)
            
            # AIメッセージのログ出力
            logger.info({"message": content, "application_mode": st.session_state.mode})
        except Exception as e:
            # エラーログの出力
            logger.error(f"{ct.DISP_ANSWER_ERROR_MESSAGE}\n{e}")
            # エラーメッセージの画面表示
            st.error(utils.build_error_message(ct.DISP_ANSWER_ERROR_MESSAGE), icon=ct.ERROR_ICON)
            # 後続の処理を中断
            st.stop()

    # ==========================================
    # 4. 会話ログへの追加
    # ==========================================
    # 表示用の会話ログにユーザーメッセージを追加
    st.session_state.messages.append({"role": "user", "content": chat_message})
    # 表示用の会話ログにAIメッセージを追加
    st.session_state.messages.append({"role": "assistant", "content": content})