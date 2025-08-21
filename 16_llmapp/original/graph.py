import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.tools.tavily_search import TavilySearchResults
from typing import Annotated
from typing_extensions import TypedDict

# 環境変数を読み込む
load_dotenv(".env")
os.environ['OPENAI_API_KEY'] = os.environ['API_KEY']

# 使用するモデル名
MODEL_NAME = "gpt-4o-mini" 

# MemorySaverインスタンスの作成
memory = MemorySaver()

# グラフを保持する変数の初期化
graph = None

# キャラクター設定用
role = "あなたは猫の執事です。猫の口調で話します。"

# ===== Stateクラスの定義 =====
class State(TypedDict):
    messages: Annotated[list, add_messages]

# ===== グラフの構築 =====
def build_graph(model_name, memory):
    # グラフのインスタンスを作成
    graph_builder = StateGraph(State)

    # ツールノードを作成
    tool = TavilySearchResults(max_results=2)
    tools = [tool]
    tool_node = ToolNode(tools)
    graph_builder.add_node("tools", tool_node)

    # チャットボットノードの作成
    llm = ChatOpenAI(model_name=model_name)
    llm_with_tools = llm.bind_tools(tools)
    
    # チャットボットの実行方法を定義
    def chatbot(state: State):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}
    
    # グラフにチャットボットノードを追加
    graph_builder.add_node("chatbot", chatbot)

    # ツール呼出と判断したらツールノードを呼ぶ
    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
    )
    # ツールが呼び出されるたびに、チャットボットに戻って次のステップを決定
    # ツールからチャットボットへの戻りエッジを作成
    graph_builder.add_edge("tools", "chatbot")
    # 開始ノードの指定
    graph_builder.set_entry_point("chatbot")
    
    return graph_builder.compile(checkpointer=memory)

# ===== グラフを実行する関数 =====
def stream_graph_updates(graph: StateGraph, user_message: str, thread_id):
    response = graph.invoke(
        {"messages": [
            ("system", role),
            ("user", user_message)]},
        {"configurable": {"thread_id": thread_id}},
        stream_mode="values"
    )
    return response["messages"][-1].content

# ===== 応答を返す関数 =====
def get_bot_response(user_message, memory, thread_id):
    global graph
    # グラフがまだ作成されていない場合、新しいグラフを作成
    if graph is None:
        graph = build_graph(MODEL_NAME, memory)

    # グラフを実行してボットの応答を取得
    return stream_graph_updates(graph, user_message, thread_id)

# ===== メッセージの一覧を取得する関数 =====
def get_messages_list(memory, thread_id):
    messages = []
    # メモリからメッセージを取得
    memories = memory.get({"configurable": {"thread_id": thread_id}})['channel_values']['messages']
    for message in memories:
        if isinstance(message, HumanMessage):
            # ユーザーからのメッセージ
            messages.append({'class': 'user-message', 'text': message.content.replace('\n', '<br>')})
        elif isinstance(message, AIMessage) and message.content != "":
            # ボットからのメッセージ（最終回答）
            messages.append({'class': 'bot-message', 'text': message.content.replace('\n', '<br>')})
    return messages