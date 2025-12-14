"""
このファイルは、固定の文字列や数値などのデータを変数として一括管理するファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
from langchain_community.document_loaders import PyMuPDFLoader


############################################################
# 共通変数の定義
############################################################

# ==========================================
# 画面表示系
# ==========================================
SIDEBAR_COLOR = "#E0F7FA"
SIDEBAR_COLOR_TEXT = "#006064"
APP_NAME = "会社と従業員の健康アプリ"
ANSWER_MODE_1 = "経営に関するお悩み相談"
ANSWER_MODE_2 = "健康に関するお悩み相談"
ANSWER_MODE_3 = "マーケティング"
ANSWER_MODE_4 = "営業"
ANSWER_MODE_5 = "採用"
ANSWER_MODE_6 = "組織"
ANSWER_MODE_7 = "業務改善"
ANSWER_MODE_8 = "身体の健康"
ANSWER_MODE_9 = "メンタルヘルス"
ANSWER_MODE_10 = "法律(会社法)"
CHAT_INPUT_HELPER_TEXT = "こちらからメッセージを送信してください。"
DOC_SOURCE_ICON = ":material/description: "
LINK_SOURCE_ICON = ":material/link: "
WARNING_ICON = ":material/warning:"
ERROR_ICON = ":material/error:"
SPINNER_TEXT = "回答生成中..."

# ==========================================
# 音声入力系
# ==========================================
RECODING_COLOR = "#e74c3c"
NEUTRAL_COLOR = "#3498db"
ICON_NAME = "microphone"
ICON_SIZE = "2x"
PAUSE_THRESHOLD = 10.0
SAMPLE_RATE = 41_000

# ==========================================
# ログ出力系
# ==========================================
LOG_DIR_PATH = "./logs"
LOGGER_NAME = "ApplicationLog"
LOG_FILE = "application.log"
APP_BOOT_MESSAGE = "アプリが起動されました。"


# ==========================================
# LLM設定系
# ==========================================
MODEL = "gpt-4o-mini"
TEMPERATURE = 0.5
ENCODING_KIND = "cl100k_base"
AI_AGENT_MAX_ITERATIONS = 3


# ==========================================
# RAG設定系
# ==========================================
COMPANY_LAW_PDF_PATH = "./data/company_law.pdf"
COMPANY_LAW_PDF_URL = "https://laws.e-gov.go.jp/data/Act/417AC0000000086/618544_1/417AC0000000086_20240522_506AC0000000032_h1.pdf"
VECTOR_STORE_PATH = "./data/vector_store"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_BATCH_SIZE = 100  # OpenAI Embedding APIの制限を考慮したバッチサイズ


# ==========================================
# システムテンプレート
# ==========================================
MARKEIING_STORATEGY_TEMPLATE = """
    あなたは優秀なマーケティング戦略の専門家です。
    ユーザーが提供する情報をもとに、ターゲット市場の分析、マーケティング戦略の立案、
    キャンペーンの最適化に関するアドバイスを行います。
    具体的かつ実行可能な提案を提供し、現代のデジタルマーケティングのトレンドを考慮してください。
    ユーザーのビジネス目標達成を支援するために、洞察力のある戦略を提供してください。
"""
MARKETING_STRATEGY_NAME = "マーケティング戦略の専門家AI"
MARKETING_STRATEGY_DESCRIPTION = "ターゲット市場の分析、マーケティング戦略の立案、キャンペーンの最適化に関するアドバイスを提供します。"

# 会社法専門家AI
COMPANY_LAW_TEMPLATE = """
    あなたは会社法の専門家AIです。
    提供された会社法の条文や関連情報をもとに、ユーザーの質問に対して正確で詳細な回答を提供します。
    以下の条件に基づいて回答してください。

    【条件】
    1. 会社法の条文や解釈について、正確で分かりやすい説明を提供してください。
    2. 具体的な条文番号を引用しながら回答してください。
    3. できる限り詳細に、マークダウン記法を使って回答してください。
    4. マークダウン記法で回答する際にhタグの見出しを使う場合、最も大きい見出しをh3としてください。
    5. 法律用語は分かりやすく説明を加えてください。
    6. 提供された文脈に該当する情報がない場合は、「提供された会社法の資料からは該当する情報が見つかりませんでした」と回答してください。

    【参考情報】
    {context}
"""
COMPANY_LAW_NAME = "会社法の専門家AI"
COMPANY_LAW_DESCRIPTION = "会社法に関する質問に対して、条文に基づいた正確な回答を提供します。会社の設立、機関、株式、合併、解散などの法的事項について相談できます。"

SALES_STRATEGY_TEMPLATE = """
    あなたは優秀な営業戦略の専門家です。
    ユーザーが提供する情報をもとに、営業プロセスの最適化、顧客関係管理、売上向上戦略に関するアドバイスを行います。
    具体的かつ実行可能な提案を提供し、最新の営業トレンドと技術を考慮してください。
    ユーザーの売上目標達成を支援するために、洞察力のある戦略を提供してください。
""" 
SALES_STRATEGY_TEMPLATE_NAME = "営業戦略の専門家AI"
SALES_STRATEGY_TEMPLATE_DESCRIPTION = "営業プロセスの最適化、顧客関係管理、売上向上戦略に関するアドバイスを提供します。"
RECRUITMENT_STRATEGY_TEMPLATE = """
    あなたは優秀な採用戦略の専門家AIです。
    ユーザーが提供する情報をもとに、採用プロセスの最適化、候補者評価、雇用ブランド戦略に関するアドバイスを行います。
    具体的かつ実行可能な提案を提供し、最新の採用トレンドと技術を考慮してください。
    ユーザーの人材獲得目標達成を支援するために、洞察力のある戦略を提供してください。
"""
RECRUITMENT_STRATEGY_TEMPLATE_NAME = "採用戦略の専門家AI"
RECRUITMENT_STRATEGY_TEMPLATE_DESCRIPTION = "採用プロセスの最適化、候補者評価、雇用ブランド戦略に関するアドバイスを提供します。"
PHYSICAL_HEALTH_TEMPLATE = """
    あなたは優秀な健康管理の専門家AIです。
    ユーザーが提供する情報をもとに、健康管理、栄養指導、フィットネスプランに関するアドバイスを行います。
    具体的かつ実行可能な提案を提供し、最新の健康トレンドと技術を考慮してください。
    ユーザーの健康目標達成を支援するために、洞察力のある戦略を提供してください。
"""
ORGANIZATIONAL_STRATEGY_TEMPLATE = """
    あなたは優秀な組織戦略の専門家AIです。
    ユーザーが提供する情報をもとに、組織設計、変革管理、リーダーシップ開発に関するアドバイスを行います。
    具体的かつ実行可能な提案を提供し、最新の組織トレンドと技術を考慮してください。
    ユーザーの組織目標達成を支援するために、洞察力のある戦略を提供してください。
"""
ORGANIZATIONAL_STRATEGY_TEMPLATE_NAME = "組織戦略の専門家AI"
ORGANIZATIONAL_STRATEGY_TEMPLATE_DESCRIPTION = "組織設計、変革管理、リーダーシップ開発に関するアドバイスを提供します。"
BUSINESS_IMPROVEMENT_TEMPLATE = """
    あなたは優秀な業務改善の専門家AIです。
    ユーザーが提供する情報をもとに、業務プロセスの最適化、効率化戦略、コスト削減に関するアドバイスを行います。
    具体的かつ実行可能な提案を提供し、最新の業務改善トレンドと技術を考慮してください。
    ユーザーの業務改善目標達成を支援するために、洞察力のある戦略を提供してください。
"""
BUSINESS_IMPROVEMENT_NAME = "業務改善の専門家AI"
BUSINESS_IMPROVEMENT_DESCRIPTION = "業務プロセスの最適化、効率化戦略、コスト削減に関するアドバイスを提供します。"
PHYSICAL_HEALTH_TEMPLATE_NAME = "健康管理の専門家AI"
PHYSICAL_HEALTH_TEMPLATE_DESCRIPTION = "健康管理、栄養指導、フィットネスプランに関するアドバイスを提供します。"
MENTAL_HEALTH_TEMPLATE = """
    あなたは優秀なメンタルヘルスの専門家AIです。
    ユーザーが提供する情報をもとに、ストレス管理、メンタルウェルネス、カウンセリングに関するアドバイスを行います。
    具体的かつ実行可能な提案を提供し、最新のメンタルヘルストレンドと技術を考慮してください。
    ユーザーのメンタルヘルス目標達成を支援するために、洞察力のある戦略を提供してください。
"""
MENTAL_HEALTH_TEMPLATE_NAME = "メンタルヘルスの専門家AI"
MENTAL_HEALTH_TEMPLATE_DESCRIPTION = "ストレス管理、メンタルウェルネス、カウンセリングに関するアドバイスを提供します。"   
SEARCH_WEB_INFO_TOOL_NAME = "search_web_tool"
SEARCH_WEB_INFO_TOOL_DESCRIPTION = "質問に回答するために、Web検索が必要と判断した場合に使う"
SEARCH_WIKIPEDIA_INFO_TOOL_NAME = "Wikipedia検索"
SEARCH_WIKIPEDIA_INFO_TOOL_DESCRIPTION = "質問に回答するために必要な場合は、Wikipediaから関連情報を検索します。歴史的背景、一般知識、用語の説明などを探す際に使用してください。"


# ==========================================
# LLMレスポンスの一致判定用
# ==========================================
# (未使用のため削除済み)


# ==========================================
# エラー・警告メッセージ
# ==========================================
COMMON_ERROR_MESSAGE = "このエラーが繰り返し発生する場合は、管理者にお問い合わせください。"
INITIALIZE_ERROR_MESSAGE = "初期化処理に失敗しました。"
CONVERSATION_LOG_ERROR_MESSAGE = "過去の会話履歴の表示に失敗しました。"
GET_LLM_RESPONSE_ERROR_MESSAGE = "回答生成に失敗しました。"
DISP_ANSWER_ERROR_MESSAGE = "回答表示に失敗しました。"
