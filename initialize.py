"""
このファイルは、最初の画面読み込み時にのみ実行される初期化処理が記述されたファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
import os
import logging
from logging.handlers import TimedRotatingFileHandler
from uuid import uuid4
from dotenv import load_dotenv
import tiktoken
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import initialize_agent, AgentType
from langchain import SerpAPIWrapper
from langchain.tools import Tool
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import constants as ct
import components as cn
import utils


############################################################
# 設定関連
############################################################
# 「.env」ファイルで定義した環境変数の読み込み
load_dotenv()


############################################################
# 関数定義
############################################################

def initialize():
    """
    画面読み込み時に実行する初期化処理
    """
    # 初期化データの用意
    initialize_session_state()
    # ログ出力用にセッションIDを生成
    initialize_session_id()
    # ログ出力の設定
    initialize_logger()
    # RAGベクトルストアの初期化
    initialize_vector_store()
    # Agent Executorを作成
    initialize_agent_executor()


def initialize_logger():
    """
    ログ出力の設定
    """
    # 指定のログフォルダが存在すれば読み込み、存在しなければ新規作成
    os.makedirs(ct.LOG_DIR_PATH, exist_ok=True)
    
    # 引数に指定した名前のロガー（ログを記録するオブジェクト）を取得
    # 再度別の箇所で呼び出した場合、すでに同じ名前のロガーが存在していれば読み込む
    logger = logging.getLogger(ct.LOGGER_NAME)

    # すでにロガーにハンドラー（ログの出力先を制御するもの）が設定されている場合、同じログ出力が複数回行われないよう処理を中断する
    if logger.hasHandlers():
        return

    # 1日単位でログファイルの中身をリセットし、切り替える設定
    log_handler = TimedRotatingFileHandler(
        os.path.join(ct.LOG_DIR_PATH, ct.LOG_FILE),
        when="D",
        encoding="utf8"
    )
    # 出力するログメッセージのフォーマット定義
    # - 「levelname」: ログの重要度（INFO, WARNING, ERRORなど）
    # - 「asctime」: ログのタイムスタンプ（いつ記録されたか）
    # - 「lineno」: ログが出力されたファイルの行番号
    # - 「funcName」: ログが出力された関数名
    # - 「session_id」: セッションID（誰のアプリ操作か分かるように）
    # - 「message」: ログメッセージ
    formatter = logging.Formatter(
        f"[%(levelname)s] %(asctime)s line %(lineno)s, in %(funcName)s, session_id={st.session_state.session_id}: %(message)s"
    )

    # 定義したフォーマッターの適用
    log_handler.setFormatter(formatter)

    # ログレベルを「INFO」に設定
    logger.setLevel(logging.INFO)

    # 作成したハンドラー（ログ出力先を制御するオブジェクト）を、
    # ロガー（ログメッセージを実際に生成するオブジェクト）に追加してログ出力の最終設定
    logger.addHandler(log_handler)


def initialize_session_id():
    """
    セッションIDの作成
    """
    if "session_id" not in st.session_state:
        # ランダムな文字列（セッションID）を、ログ出力用に作成
        st.session_state.session_id = uuid4().hex


def initialize_session_state():
    """
    初期化データの用意
    """
    if "messages" not in st.session_state:
        # 「表示用」の会話ログを順次格納するリストを用意
        st.session_state.messages = []
        # 「LLMとのやりとり用」の会話ログを順次格納するリストを用意
        st.session_state.chat_history = []


def initialize_vector_store():
    """
    RAGベクトルストアの初期化
    会社法PDFを読み込み、ベクトル化して保存または読み込み
    """
    logger = logging.getLogger(ct.LOGGER_NAME)
    
    # すでにベクトルストアが作成済みの場合、後続の処理を中断
    if "vector_store" in st.session_state:
        return
    
    try:
        # すでに保存済みのベクトルストアがあれば読み込み
        if os.path.exists(ct.VECTOR_STORE_PATH):
            logger.info("既存のベクトルストアを読み込みます")
            embeddings = OpenAIEmbeddings()
            st.session_state.vector_store = FAISS.load_local(
                ct.VECTOR_STORE_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info("ベクトルストアの読み込みが完了しました")
        else:
            # PDFが存在しない場合はダウンロード
            if not os.path.exists(ct.COMPANY_LAW_PDF_PATH):
                logger.info("会社法PDFをダウンロードします")
                import urllib.request
                os.makedirs(os.path.dirname(ct.COMPANY_LAW_PDF_PATH), exist_ok=True)
                urllib.request.urlretrieve(ct.COMPANY_LAW_PDF_URL, ct.COMPANY_LAW_PDF_PATH)
                logger.info("会社法PDFのダウンロードが完了しました")
            
            # PDFを読み込み
            logger.info("会社法PDFを読み込みます")
            loader = ct.PyMuPDFLoader(ct.COMPANY_LAW_PDF_PATH)
            documents = loader.load()
            
            # テキストを分割
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=ct.CHUNK_SIZE,
                chunk_overlap=ct.CHUNK_OVERLAP
            )
            texts = text_splitter.split_documents(documents)
            logger.info(f"{len(texts)}個のチャンクに分割しました")
            
            # ベクトルストアを作成（バッチ処理で段階的に作成）
            logger.info("ベクトルストアを作成します（バッチ処理）")
            embeddings = OpenAIEmbeddings()
            
            # 最初のバッチでベクトルストアを初期化
            batch_size = ct.EMBEDDING_BATCH_SIZE
            first_batch = texts[:batch_size]
            st.session_state.vector_store = FAISS.from_documents(first_batch, embeddings)
            logger.info(f"1/{(len(texts) + batch_size - 1) // batch_size} バッチ完了")
            
            # 残りのバッチを追加
            for i in range(batch_size, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_vector_store = FAISS.from_documents(batch, embeddings)
                st.session_state.vector_store.merge_from(batch_vector_store)
                logger.info(f"{(i // batch_size) + 1}/{(len(texts) + batch_size - 1) // batch_size} バッチ完了")
            
            # ベクトルストアを保存
            os.makedirs(ct.VECTOR_STORE_PATH, exist_ok=True)
            st.session_state.vector_store.save_local(ct.VECTOR_STORE_PATH)
            logger.info("ベクトルストアの作成と保存が完了しました")
            
    except Exception as e:
        logger.error(f"ベクトルストアの初期化に失敗しました: {str(e)}")
        st.session_state.vector_store = None


def initialize_agent_executor():
    """
    画面読み込み時にAgent Executor（AIエージェント機能の実行を担当するオブジェクト）を作成
    """
    logger = logging.getLogger(ct.LOGGER_NAME)

    # すでにAgent Executorが作成済みの場合、後続の処理を中断
    if "agent_executor" in st.session_state:
        return
    
    # 消費トークン数カウント用のオブジェクトを用意
    st.session_state.enc = tiktoken.get_encoding(ct.ENCODING_KIND)
    
    st.session_state.llm = ChatOpenAI(model_name=ct.MODEL, temperature=ct.TEMPERATURE, streaming=True)


    # Web検索用のToolを設定するためのオブジェクトを用意
    search = SerpAPIWrapper()
    # Agent Executorに渡すTool一覧を用意
    tools = [
        # マーケティング戦略に関するアドバイス用のTool
        Tool.from_function(
            func=cn.get_marketing_strategy_advice,
            name=ct.MARKETING_STRATEGY_NAME,
            description=ct.MARKETING_STRATEGY_DESCRIPTION,
        ),

        # 営業戦略に関するアドバイス用のTool
        Tool.from_function(
            func=cn.get_sales_strategy_advice,
            name=ct.SALES_STRATEGY_TEMPLATE_NAME,
            description=ct.SALES_STRATEGY_TEMPLATE_DESCRIPTION, 
        ),

        # 採用戦略に関するアドバイス用のTool
        Tool.from_function(
            func=cn.get_recruitment_strategy_advice,
            name=ct.RECRUITMENT_STRATEGY_TEMPLATE_NAME,
            description=ct.RECRUITMENT_STRATEGY_TEMPLATE_DESCRIPTION, 
        ),

        # 組織戦略に関するアドバイス用のTool
        Tool.from_function(
            func=cn.get_organizational_storategy_advice,
            name=ct.ORGANIZATIONAL_STRATEGY_TEMPLATE_NAME,
            description=ct.ORGANIZATIONAL_STRATEGY_TEMPLATE_DESCRIPTION, 
        ),

        # 業務改善に関するアドバイス用のTool
        Tool.from_function(
            func=cn.get_buisiness_improvement_advice,
            name=ct.BUSINESS_IMPROVEMENT_NAME,
            description=ct.BUSINESS_IMPROVEMENT_DESCRIPTION, 
        ),

        # 健康管理に関するアドバイス用のTool
        Tool.from_function(
            func=cn.get_physical_health_advice,
            name=ct.PHYSICAL_HEALTH_TEMPLATE_NAME,
            description=ct.PHYSICAL_HEALTH_TEMPLATE_DESCRIPTION, 
        ),

        # メンタルヘルスに関するアドバイス用のTool
        Tool.from_function(
            func=cn.get_mental_health_advice,
            name=ct.MENTAL_HEALTH_TEMPLATE_NAME,
            description=ct.MENTAL_HEALTH_TEMPLATE_DESCRIPTION, 
        ),

        # 会社法に関するアドバイス用のTool
        Tool.from_function(
            func=cn.get_company_law_advice,
            name=ct.COMPANY_LAW_NAME,
            description=ct.COMPANY_LAW_DESCRIPTION, 
        ),

        # Web検索用のTool
        Tool(
            name = ct.SEARCH_WEB_INFO_TOOL_NAME,
            func=search.run,
            description=ct.SEARCH_WEB_INFO_TOOL_DESCRIPTION
        ),

        # Wikipedia検索用のTool
        Tool(
            name = ct.SEARCH_WIKIPEDIA_INFO_TOOL_NAME,
            func=utils.run_wikipedia_search,
            description=ct.SEARCH_WIKIPEDIA_INFO_TOOL_DESCRIPTION
        ),
    ]

    # Agent Executorの作成
    st.session_state.agent_executor = initialize_agent(
        llm=st.session_state.llm,
        tools=tools,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        max_iterations=ct.AI_AGENT_MAX_ITERATIONS,
        early_stopping_method="generate",
        handle_parsing_errors=True
    )