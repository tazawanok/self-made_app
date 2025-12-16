"""
このファイルは、画面表示に特化した関数定義のファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
import streamlit as st
import utils
import constants as ct
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain import LLMChain


############################################################
# 関数定義
############################################################

def set_sidebar_style():
    """
    サイドバーのスタイル設定
    """ 
    st.markdown(f"""
    <style>
        [data-testid="stSidebar"] {{
            background-color: {ct.SIDEBAR_COLOR};
        }}
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {{
            color: {ct.SIDEBAR_COLOR_TEXT};
        }}
    </style>
    """, unsafe_allow_html=True)



def display_app_title():
    """
    タイトル表示
    """
    st.markdown(f"## {ct.APP_NAME}")


def display_select_mode():
    """
    回答モードのラジオボタンを表示
    """
    # 回答モードを選択する用のラジオボタンを表示
    st.sidebar.markdown("**お悩み選択**")
    # 「label_visibility="collapsed"」とすることで、ラジオボタンを非表示にする
    st.session_state.mode = st.sidebar.radio(
        label="",
        options=[ct.ANSWER_MODE_1, ct.ANSWER_MODE_2],
        label_visibility="collapsed"
    )
    st.sidebar.divider()
    st.sidebar.write("")


def display_select_genre():
    """
    回答モードのラジオボタンを表示
    """
    # 回答モードを選択する用のラジオボタンを表示
    # 「label_visibility="collapsed"」とすることで、ラジオボタンを非表示にする
    st.sidebar.markdown("**ジャンル選択**")
    if st.session_state.mode == ct.ANSWER_MODE_1:
        st.session_state.mode_2 = st.sidebar.selectbox(
        label="",
        options=[ct.ANSWER_MODE_3, ct.ANSWER_MODE_4, ct.ANSWER_MODE_5, ct.ANSWER_MODE_6, ct.ANSWER_MODE_7, ct.ANSWER_MODE_10],
        label_visibility="collapsed"
        )
    else:
        st.session_state.mode_2 = st.sidebar.selectbox(
            label="",
            options=[ct.ANSWER_MODE_8, ct.ANSWER_MODE_9],
            label_visibility="collapsed"
            )
    st.sidebar.divider()
    st.sidebar.write("")


def is_mode_changed():
    """
    回答モード変更時の判定処理
    Returns:
        回答モードが変更された場合True、変更されていない場合False
    """
    # 現在の回答モードと前回の回答モードを比較し、変更されていればTrueを返す
    if st.session_state.mode != st.session_state.get("previous_mode", None):
        st.session_state.previous_mode = st.session_state.mode
        return True
    else:
        return False
    
def clear_conversation_log():
    """
    会話履歴のクリア
    """
    st.session_state.messages = []

def reset_genre_selection():
    """
    ジャンル選択の初期化
    """
    st.session_state.mode_2 = None



def display_initial_ai_message():
    """
    AIメッセージの初期表示
    """
    with st.chat_message("assistant"):
        st.success("こんにちは。私は企業と従業員の健康を手助けする生成AIチャットボットです。お悩みとジャンルを選択し、画面下部のチャット欄からメッセージを送信してください。")
    st.warning("具体的に入力したほうが期待通りの回答を得やすいです。")
    # 「st.code()」を使うとコードブロックの装飾で表示される
    # 「wrap_lines=True」で折り返し設定、「language=None」で非装飾とする
    # st.code("【入力例】\n社員の育成方針に関するMTGの議事録", wrap_lines=True, language=None)




def display_sidebar():
    st.sidebar.write("")
    # 「社内問い合わせ」の機能説明
    st.sidebar.markdown("**【「経営に関するお悩み相談」を選択した場合】**")
    st.sidebar.markdown("経営に関する質問・要望に対して、適切なアドバイスを得られます。")
    st.sidebar.code("【入力例】\n売上を伸ばすための新しいマーケティング戦略を提案して", wrap_lines=True, language=None)
    st.sidebar.markdown("**【「健康に関するお悩み」を選択した場合】**")
    st.sidebar.markdown("健康に関する質問・要望に対して、適切なアドバイスを得られます。")
    st.sidebar.code("【入力例】\n最近デスクワークで肩こりがひどいです。対策を教えてください。", wrap_lines=True, language=None)
    st.sidebar.divider()
    st.sidebar.markdown("※ 会話の途中でお悩みやジャンルを変更したい場合は、サイドバーから再度選択してください。会話履歴がクリアされ、新しいモードでの会話が開始されます。")



def display_conversation_log():
    """
    会話ログの一覧表示
    """
    # 会話ログのループ処理
    for message in st.session_state.messages:
        # 「message」辞書の中の「role」キーには「user」か「assistant」が入っている
        with st.chat_message(message["role"]):

            # ユーザー入力値の場合、そのままテキストを表示するだけ
            if message["role"] == "user":
                st.markdown(message["content"])
            
            # LLMからの回答の場合
            else:
                # LLMからの回答を表示
                st.markdown(message["content"]["answer"])

                # 参照元のありかを一覧表示（オプション）
                if "file_info_list" in message["content"]:
                    # 区切り線の表示
                    st.divider()
                    # 「情報源」の文字を太字で表示
                    st.markdown(f"##### {message['content']['message']}")
                    # ドキュメントのありかを一覧表示
                    for file_info in message["content"]["file_info_list"]:
                        # 参照元のありかに応じて、適したアイコンを取得
                        icon = utils.get_source_icon(file_info)
                        st.info(file_info, icon=icon)


def result_chain(param, system_template):
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", "{input}")
    ])
    chain = LLMChain(prompt=prompt, llm=llm)
    result = chain.run(param)
    return result

def display_contact_llm_response(llm_response):
    """
    Agent ExecutorからのLLM回答を表示
    
    Args:
        llm_response: Agent Executorからの回答（辞書または文字列）
        
    Returns:
        表示したコンテンツ（ログ出力用）
    """
    # Agent Executorの回答から出力テキストを取得
    if isinstance(llm_response, dict):
        answer = llm_response.get("output", "")
    else:
        # 文字列の場合はそのまま使用
        answer = str(llm_response)

    # ReActの思考ログ（Thought/Action）がそのまま返ってきた場合のフォールバック
    # Tool実行に失敗した際に中間ステップが混ざることがあるため、ユーザーには簡潔な案内のみ表示する。
    if "Action:" in answer and "Final Answer" not in answer:
        answer = "外部検索が完了しませんでした。キーワードを変えて再度お試しください。"
    
    # 回答を表示
    st.markdown(answer)
    
    # ログ用に回答内容を返す
    return {"mode": st.session_state.mode, "answer": answer}

def get_marketing_strategy_advice(param):
    system_template = ct.MARKEIING_STORATEGY_TEMPLATE
    result = result_chain(param, system_template)
    return result

def get_sales_strategy_advice(param):
    system_template = ct.SALES_STRATEGY_TEMPLATE
    result = result_chain(param, system_template)
    return result

def get_recruitment_strategy_advice(param):
    system_template = ct.RECRUITMENT_STRATEGY_TEMPLATE
    result = result_chain(param, system_template)
    return result

def get_organizational_storategy_advice(param):
    system_template = ct.ORGANIZATIONAL_STRATEGY_TEMPLATE
    result = result_chain(param, system_template)
    return result

def get_buisiness_improvement_advice(param):
    system_template = ct.BUSINESS_IMPROVEMENT_TEMPLATE
    result = result_chain(param, system_template)
    return result

def get_physical_health_advice(param):
    system_template = ct.PHYSICAL_HEALTH_TEMPLATE
    result = result_chain(param, system_template)
    return result

def get_mental_health_advice(param):
    system_template = ct.MENTAL_HEALTH_TEMPLATE
    result = result_chain(param, system_template)
    return result

def get_company_law_advice(param):
    """
    会社法に関する質問に対して、RAGを使って回答を生成
    """
    
    # ベクトルストアが初期化されていない場合はエラーメッセージ
    if "vector_store" not in st.session_state or st.session_state.vector_store is None:
        return "会社法の資料が読み込まれていません。アプリを再起動してください。"
    
    try:
        # 関連する文書を検索
        retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": ct.SEARCH_TOP_K})
        docs = retriever.get_relevant_documents(param)
        
        # 検索結果を文脈として結合
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # プロンプトテンプレートに文脈を埋め込み
        system_template = ct.COMPANY_LAW_TEMPLATE.format(context=context)
        result = result_chain(param, system_template)
        return result
    except Exception as e:
        return f"会社法の検索中にエラーが発生しました: {str(e)}"



