"""
このファイルは、画面表示以外の様々な関数定義のファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
import os
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI
import constants as ct
import requests
from urllib.parse import quote


############################################################
# 設定関連
############################################################
# 「.env」ファイルで定義した環境変数の読み込み
load_dotenv()


############################################################
# 関数定義
############################################################

def get_source_icon(source):
    """
    メッセージと一緒に表示するアイコンの種類を取得

    Args:
        source: 参照元のありか

    Returns:
        メッセージと一緒に表示するアイコンの種類
    """
    # 参照元がWebページの場合とファイルの場合で、取得するアイコンの種類を変える
    if source.startswith("http"):
        icon = ct.LINK_SOURCE_ICON
    else:
        icon = ct.DOC_SOURCE_ICON
    
    return icon


def build_error_message(message):
    """
    エラーメッセージと管理者問い合わせテンプレートの連結

    Args:
        message: 画面上に表示するエラーメッセージ

    Returns:
        エラーメッセージと管理者問い合わせテンプレートの連結テキスト
    """
    return "\n".join([message, ct.COMMON_ERROR_MESSAGE])


def _build_conversational_input(chat_message: str, max_turns: int = 4) -> str:
    """直近の会話ログとアプリのモード・ジャンルを踏まえた文脈付き入力テキストを生成する。

    Args:
        chat_message: ユーザーの最新入力
        max_turns: 直近何ペア分の会話を含めるか

    Returns:
        文脈付きの入力テキスト
    """
    mode = getattr(st.session_state, "mode", "")
    mode_2 = getattr(st.session_state, "mode_2", "")
    
    # ヘッダー作成：お悩み種別とジャンルの両方を含める
    header_parts = []
    if mode:
        header_parts.append(f"[お悩み種別: {mode}]")
    if mode_2:
        header_parts.append(f"[選択ジャンル: {mode_2}]")
    header = "\n".join(header_parts) if header_parts else ""

    # 表示用ログ（`messages`）から直近の会話を抽出（ユーザー/AIのテキストのみ使用）
    history = []
    if "messages" in st.session_state and isinstance(st.session_state.messages, list):
        # ユーザー/AIの発話本文があれば取り出す
        for m in st.session_state.messages:
            role = m.get("role") if isinstance(m, dict) else None
            content = m.get("content") if isinstance(m, dict) else None
            if role in ("user", "assistant") and isinstance(content, str):
                history.append((role, content))

    # 直近max_turnsペア分に圧縮
    history_text = []
    for role, content in history[-max_turns * 2:]:
        prefix = "ユーザー:" if role == "user" else "アシスタント:"
        history_text.append(f"{prefix} {content}")

    history_block = "\n".join(history_text) if history_text else ""

    # 最終的な入力テキスト
    parts = [p for p in [header, history_block, f"ユーザー: {chat_message}"] if p]
    return "\n\n".join(parts)


def get_llm_response(chat_message: str):
    """
    Agent Executorを使用して、直近の会話文脈を含めた入力で回答を取得する。

    Args:
        chat_message: ユーザー入力値

    Returns:
        文字列の回答
    """
    agent_executor = st.session_state.agent_executor

    contextual_input = _build_conversational_input(chat_message)

    result = agent_executor.invoke({"input": contextual_input})
    # AgentExecutorは標準で{"output": "..."}形式を返す
    return result.get("output", result)


def run_wikipedia_search(query: str) -> str:
    """Wikipediaで検索し、最上位の概要を返す簡易ツール。

    - まず検索APIで最上位ヒットのタイトルを取得
    - REST Summary APIで概要を取得
    - 失敗時はエラーメッセージを返す
    """
    try:
        # ひとまず日本語版を優先（必要ならct側で設定化可能）
        lang = "ja"
        session = requests.Session()

        # 検索API
        search_params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
        }
        search_url = f"https://{lang}.wikipedia.org/w/api.php"
        resp = session.get(search_url, params=search_params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        hits = data.get("query", {}).get("search", [])
        if not hits:
            return f"Wikipediaで該当記事が見つかりませんでした: {query}"

        title = hits[0].get("title")
        if not title:
            return f"Wikipedia検索結果の取得に失敗しました: {query}"

        # Summary API
        summary_url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{quote(title)}"
        sresp = session.get(summary_url, timeout=10)
        sresp.raise_for_status()
        sdata = sresp.json()
        extract = sdata.get("extract") or "概要が取得できませんでした。"
        page_url = sdata.get("content_urls", {}).get("desktop", {}).get("page")
        page_url = page_url or f"https://{lang}.wikipedia.org/wiki/{quote(title)}"

        return f"【Wikipedia】{title}\n{extract}\n\nURL: {page_url}"

    except Exception as e:
        return f"Wikipedia検索中にエラーが発生しました: {e}"