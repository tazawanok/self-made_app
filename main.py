"""
このファイルは、Webアプリのメイン処理が記述されたファイルです。
"""

############################################################
# 1. ライブラリの読み込み
############################################################
# 「.env」ファイルから環境変数を読み込むための関数
from dotenv import load_dotenv
# ログ出力を行うためのモジュール
import logging
# streamlitアプリの表示を担当するモジュール
import streamlit as st
# 音声録音用
from audio_recorder_streamlit import audio_recorder
# 音声認識用
import openai
import os
import hashlib
import tempfile
# （自作）画面表示以外の様々な関数が定義されているモジュール
import utils
# （自作）アプリ起動時に実行される初期化処理が記述された関数
from initialize import initialize
# （自作）画面表示系の関数が定義されているモジュール
import components as cn
# （自作）変数（定数）がまとめて定義・管理されているモジュール
import constants as ct


############################################################
# 2. 設定関連
############################################################
# ブラウザタブの表示文言を設定
st.set_page_config(
    page_title=ct.APP_NAME
)

# ログ出力を行うためのロガーの設定
logger = logging.getLogger(ct.LOGGER_NAME)




###########################################################
# 3. 初期化処理
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

# session_stateの初期化（音声・チャット関連）
if "transcribed_text" not in st.session_state:
    st.session_state.transcribed_text = None
if "last_audio_hash" not in st.session_state:
    st.session_state.last_audio_hash = None
if "processing_message" not in st.session_state:
    st.session_state.processing_message = False
if "audio_recorder_key" not in st.session_state:
    st.session_state.audio_recorder_key = 0
if "audio_error_count" not in st.session_state:
    st.session_state.audio_error_count = 0
if "openai_client" not in st.session_state:
    st.session_state.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


############################################################
# 4. 初期表示
############################################################

# サイドバーを水色にするカスタムCSS
cn.set_sidebar_style()

 #タイトル表示
cn.display_app_title()

# モード表示
cn.display_select_mode()

# ジャンル表示
cn.display_select_genre()

# 選択内容の表示（お悩み・ジャンル）
cn.display_selected_filters()

# モード変更時の処理
if cn.is_mode_changed():
    # 会話履歴のクリア
    cn.clear_conversation_log()
    # ジャンル選択の初期化
    cn.reset_genre_selection()
    # Streamlitを再実行してUIを更新
    st.rerun()

cn.display_sidebar()

# AIメッセージの初期表示
cn.display_initial_ai_message()


############################################################
# 5. 会話ログの表示
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
# 6. チャット入力の受け付け
############################################################
# 音声入力とテキスト入力を並べて配置
col1, col2 = st.columns([1, 5])

with col1:
    st.markdown("**🎤音声入力**")
    
    # 音声認識結果の確認画面が表示されている場合は、オーディオレコーダーを無効化
    if not st.session_state.get("transcribed_text"):
        audio_bytes = audio_recorder(
            text="",
            recording_color=ct.RECODING_COLOR,
            neutral_color=ct.NEUTRAL_COLOR,
            icon_name=ct.ICON_NAME,
            icon_size=ct.ICON_SIZE,
            pause_threshold=ct.PAUSE_THRESHOLD,
            sample_rate=ct.SAMPLE_RATE,
            key=f"audio_recorder_{st.session_state.audio_recorder_key}"
        )
    else:
        audio_bytes = None

with col2:
    chat_message_input = st.chat_input(ct.CHAT_INPUT_HELPER_TEXT)
    
    # テキスト入力があった場合、session_stateに保存
    if chat_message_input:
        st.session_state.chat_message_to_send = chat_message_input
        # テキスト入力時は音声入力の状態をリセット
        st.session_state.transcribed_text = None
        st.session_state.last_audio_hash = None
        # audio_recorderをリセット（キーを変更してコンポーネントを再生成）
        st.session_state.audio_recorder_key += 1

# 音声入力の処理（メッセージ処理中でない場合のみ）
if audio_bytes and not st.session_state.get("chat_message_to_send") and not st.session_state.processing_message:
    # 現在の音声データのハッシュを計算（SHA256を使用）
    current_audio_hash = hashlib.sha256(audio_bytes).hexdigest()
    
    # 前回と異なる音声データかチェック
    if current_audio_hash != st.session_state.last_audio_hash:
        # 一時ファイルのパス（try-finallyで確実に削除）
        temp_file = None
        try:
            # 音声データのハッシュを保存
            st.session_state.last_audio_hash = current_audio_hash
            
            # 一意な一時ファイルを作成
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            temp_file.write(audio_bytes)
            temp_file.close()
            
            # OpenAI Whisper APIで音声をテキストに変換
            with open(temp_file.name, "rb") as audio_file:
                transcript = st.session_state.openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="ja"
                )
            
            st.session_state.transcribed_text = transcript.text
            
            # 成功したらエラーカウントをリセット
            st.session_state.audio_error_count = 0
                
        except Exception as e:
            logger.error(f"音声認識エラー: {e}")
            st.session_state.audio_error_count += 1
            
            # 1回目のエラーはメッセージを表示しない（2回目以降は表示）
            if st.session_state.audio_error_count > 1:
                st.error("音声の認識に失敗しました。もう一度お試しください。", icon=ct.ERROR_ICON)
        
        finally:
            # 一時ファイルを確実に削除
            if temp_file and os.path.exists(temp_file.name):
                try:
                    os.remove(temp_file.name)
                except:
                    pass

# 音声認識結果の確認画面（メッセージ処理中でない場合のみ表示）
if st.session_state.transcribed_text and not st.session_state.processing_message:
    st.info("🎤 音声が認識されました。以下のテキストで送信しますか？")
    
    # 認識されたテキストを表示・編集可能にする
    edited_message = st.text_area(
        "認識されたテキスト（編集可能）:",
        value=st.session_state.transcribed_text,
        height=80,
        label_visibility="collapsed"
    )
    
    # 送信ボタン
    col_send, col_cancel = st.columns(2)
    with col_send:
        if st.button("✓ 送信", use_container_width=True, key="send_button"):
            # session_stateに保存して次のセクションで処理
            st.session_state.chat_message_to_send = edited_message
            # 音声関連の状態を完全にリセット
            st.session_state.transcribed_text = None
            st.session_state.last_audio_hash = None
            st.rerun()
    
    with col_cancel:
        if st.button("✕ キャンセル", use_container_width=True, key="cancel_button"):
            # 音声関連の状態を完全にリセット
            st.session_state.transcribed_text = None
            st.session_state.last_audio_hash = None
            st.rerun()

# session_stateから送信予定のメッセージを取得
chat_message = None
if "chat_message_to_send" in st.session_state and st.session_state.chat_message_to_send:
    chat_message = st.session_state.chat_message_to_send


############################################################
# 7. チャット送信時の処理
############################################################
if chat_message:
    # メッセージ処理中フラグを立てる
    st.session_state.processing_message = True
    
    # ==========================================
    # 7-1. ユーザーメッセージの表示
    # ==========================================
    # ユーザーメッセージのログ出力
    logger.info({"message": chat_message, "application_mode": st.session_state.mode})

    # ユーザーメッセージを表示
    with st.chat_message("user"):
        st.markdown(chat_message)

    # ==========================================
    # 7-2. LLMからの回答取得
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
    # 7-3. LLMからの回答表示
    # ==========================================
    with st.chat_message("assistant"):
        try:
            # モードに応じた表示関数のマッピング
            mode_handlers = {
                ct.ANSWER_MODE_3: cn.display_contact_llm_response,  # マーケティング
                ct.ANSWER_MODE_4: cn.display_contact_llm_response,  # 営業
                ct.ANSWER_MODE_5: cn.display_contact_llm_response,  # 採用
                ct.ANSWER_MODE_6: cn.display_contact_llm_response,  # 組織戦略
                ct.ANSWER_MODE_7: cn.display_contact_llm_response,  # 業務改善
                ct.ANSWER_MODE_8: cn.display_contact_llm_response,  # 身体の健康
                ct.ANSWER_MODE_9: cn.display_contact_llm_response,  # メンタルヘルス
                ct.ANSWER_MODE_10: cn.display_contact_llm_response,  # 会社法
            }
            
            # 対応する表示関数を取得して実行
            display_func = mode_handlers.get(st.session_state.mode_2)
            if display_func:
                content = display_func(llm_response)
            else:
                # デフォルト処理
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
    # 7-4. 会話ログへの追加
    # ==========================================
    # 表示用の会話ログにユーザーメッセージを追加
    st.session_state.messages.append({"role": "user", "content": chat_message})
    # 表示用の会話ログにAIメッセージを追加
    st.session_state.messages.append({"role": "assistant", "content": content})
    
    # ==========================================
    # 7-5. セッション状態のクリーンアップ
    # ==========================================
    # 入力関連の状態をリセット（重要：st.rerunの前に実行）
    st.session_state.chat_message_to_send = None
    st.session_state.transcribed_text = None
    st.session_state.last_audio_hash = None
    st.session_state.processing_message = False
    # audio_recorderをリセット（キーを変更してコンポーネントを再生成）
    st.session_state.audio_recorder_key += 1
    
    # 画面を再読み込みして次の入力を受け付ける
    st.rerun()
