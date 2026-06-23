"""AI 智能体：优先使用 OpenAI/langchain，如无密钥或依赖则回退到本地规则生成简要复盘。"""
import os
import math
from typing import Optional, List
import pandas as pd
# 尝试自动加载 .env，使得单独运行测试脚本或模块时也能读取到 DEEPSEEK_KEY/DEEPSEEK_API_KEY
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv()
except Exception:
    pass

OPENAI_KEY = os.getenv("OPENAI_API_KEY")

langchain_available = False
LangChainLLM = None
LangChainInit = None
Tool = None
initialize_agent = None
LLMChain = None
PromptTemplate = None
ConversationBufferMemory = None

# 尝试多种导入路径以兼容不同版本的 langchain
try:
    # 新版常见：ChatOpenAI 位于 chat_models
    # 尝试 init_chat_model 或 ChatOpenAI
    try:
        from langchain.chat_models import init_chat_model as LangChainInit
    except Exception:
        from langchain.chat_models import ChatOpenAI as LangChainLLM
    from langchain.agents import initialize_agent
    try:
        from langchain.tools import Tool
    except Exception:
        from langchain.agents import Tool
    from langchain.memory import ConversationBufferMemory
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
    langchain_available = True
except Exception:
    try:
        # 兼容旧路径
        from langchain.llms import OpenAI as LangChainLLM
        from langchain.agents import initialize_agent
        try:
            from langchain.tools import Tool
        except Exception:
            from langchain.agents import Tool
        from langchain.memory import ConversationBufferMemory
        from langchain.chains import LLMChain
        from langchain.prompts import PromptTemplate
        langchain_available = True
    except Exception:
        # 无法使用 langchain 的 agent 功能
        langchain_available = False

# 最终根据实际导入结果设定可用标志（兼容部分成功的情况）
if LangChainInit is not None or LangChainLLM is not None:
    langchain_available = True

try:
    import openai
    openai_available = True
except Exception:
    openai = None
    openai_available = False


def _local_summary(df: pd.DataFrame, top_n: int = 6) -> str:
    """基于表格的简单规则生成短线复盘（当无外部LLM时使用）。"""
    if df is None or df.empty:
        return "当日未抓取到涨停数据，无法生成复盘。"

    df2 = df.copy()
    for c in ("封单额", "涨幅", "连板"):
        if c in df2.columns:
            df2[c] = pd.to_numeric(df2[c], errors='coerce').fillna(0)

    top_industries = []
    if "行业" in df2.columns:
        top_industries = df2["行业"].value_counts().head(3).index.tolist()

    candidates = df2.sort_values(by=[col for col in ("连板", "封单额") if col in df2.columns], ascending=False)
    candidates = candidates.head(top_n)

    lines = []
    lines.append("【市场主线】")
    if top_industries:
        lines.append("热点行业：" + "、".join(top_industries))
    else:
        lines.append("未能明显识别今日板块主线。")

    lines.append("\n【关注个股（简要）】")
    for _, row in candidates.iterrows():
        name = row.get("名称") or row.get("代码")
        lb = int(row.get("连板", 0)) if "连板" in row.index else 0
        fb = int(row.get("封单额", 0)) if "封单额" in row.index else 0
        lines.append(f"{name}：连板 {lb}，封单额 {fb}，涨幅 {row.get('涨幅', '')}")

    lines.append("\n【仓位与风险】")
    lines.append("建议控制仓位，注意炸板风险；仅作信息参考，不构成投资建议。")

    return "\n".join(lines)


def _make_stock_tool(get_stock_info_callable):
    """返回一个 LangChain Tool，包装传入的获取个股信息函数（字符串 -> 字符串）。"""

    def _tool_func(code: str) -> str:
        try:
            info = get_stock_info_callable(code)
            return str(info)
        except Exception as e:
            return f"获取个股信息失败：{e}"

    return Tool(name="get_stock_info", func=_tool_func, description="输入股票代码，返回该股票的基本信息与委托簿摘要。")


def create_agent_with_tools(get_stock_info_callable, temperature: float = 0.2):
    """初始化并返回一个 LangChain agent（executor）和 memory。若 langchain 不可用则返回 (None, None)。"""
    if not langchain_available:
        return None, None

    # 调试信息，帮助定位初始化失败的原因
    try:
        print(f"DEBUG: langchain_available={langchain_available}, LangChainInit={'yes' if LangChainInit is not None else 'no'}, LangChainLLM={'yes' if LangChainLLM is not None else 'no'}")
    except Exception:
        pass

    # 如果存在 init_chat_model（新版 langchain），使用它创建 chat model 并用 LLMChain 封装
    if LangChainInit is not None:
        # 优先尝试 DeepSeek（无 OpenAI 密钥时可使用），否则回退到 OpenAI
        # 支持两种环境变量命名：DEEPSEEK_API_KEY（推荐）或 DEEPSEEK_KEY（兼容旧 .env）
        deepseek_key = os.getenv('DEEPSEEK_API_KEY') or os.getenv('DEEPSEEK_KEY')
        try:
            print(f"DEBUG: DEEPSEEK key present: {bool(deepseek_key)}")
            # 若用户在 .env 使用了 DEEPSEEK_KEY，为兼容性把它同步到 DEEPSEEK_API_KEY
            if deepseek_key and not os.getenv('DEEPSEEK_API_KEY'):
                os.environ['DEEPSEEK_API_KEY'] = deepseek_key
                print('DEBUG: copied DEEPSEEK_KEY -> DEEPSEEK_API_KEY in os.environ')
            if deepseek_key:
                # 尝试用 DeepSeek provider（需要安装 langchain-deepseek）
                try:
                    chat = LangChainInit("deepseek:default", temperature=temperature)
                except Exception:
                    # 某些集成可能要求不同的 model id，尝试只使用 provider
                    chat = LangChainInit("deepseek", temperature=temperature)
            else:
                # 回退到 OpenAI（如果配置了 OPENAI_API_KEY）
                chat = LangChainInit("openai:gpt-3.5-turbo", temperature=temperature)

            # 确认相关组件可用
            if PromptTemplate is None or LLMChain is None or ConversationBufferMemory is None:
                print('DEBUG: PromptTemplate/LLMChain/ConversationBufferMemory not available; cannot use init_chat_model path')
                return None, None

            prompt = PromptTemplate(input_variables=["text"], template="{text}")
            llm_chain = LLMChain(llm=chat, prompt=prompt)
            memory = ConversationBufferMemory(memory_key="chat_history")

            class SimpleAgent:
                def __init__(self, chain, memory):
                    self.chain = chain
                    self.memory = memory

                def run(self, text: str):
                    return self.chain.run(text)

            return SimpleAgent(llm_chain, memory), memory
        except Exception as e:
            import traceback
            print(f"DEBUG: init_chat_model path failed: {e}")
            traceback.print_exc()
            return None, None

    # 兼容老接口，若存在 LangChainLLM 则使用 initialize_agent
    try:
        print("DEBUG: attempting legacy LangChainLLM path")
        llm = LangChainLLM(temperature=temperature)
        tools = [_make_stock_tool(get_stock_info_callable)]
        memory = ConversationBufferMemory(memory_key="chat_history")
        agent = initialize_agent(tools, llm, agent="zero-shot-react-description", memory=memory, verbose=False)
        return agent, memory
    except Exception as e:
        import traceback
        print(f"DEBUG: legacy LangChainLLM path failed: {e}")
        traceback.print_exc()
        return None, None


def generate_daily_report_with_agent(df: pd.DataFrame, agent, memory, top_n: int = 8) -> str:
    """使用 LangChain agent 生成结构化复盘（agent 必须事先创建）。"""
    if df is None or df.empty:
        return "当日未抓取到涨停数据，无法生成复盘。"
    if agent is None:
        return _local_summary(df, top_n=top_n)

    preview = df.head(top_n).fillna("").to_dict(orient="records")
    prompt = (
        "请基于下面的涨停股列表，生成一篇面向短线交易者的复盘：\n"
        "要求：1) 给出今日市场主线和板块热点；2) 挑出 3-6 只具有短线逻辑的个股并给出理由；"
        "3) 如需更详细个股信息，可调用工具 `get_stock_info` 获取委托簿或历史信息；4) 给出仓位与风险提示。\n\n"
        f"数据样例：{preview}\n\n请用中文输出，结构化清晰。\n"
    )

    try:
        res = agent.run(prompt)
        return res
    except Exception as e:
        return _local_summary(df, top_n=top_n) + f"\n\n（Agent 调用失败：{e}，已回退本地结果）"


def generate_daily_report(df: pd.DataFrame, get_stock_info_callable=None, top_n: int = 8):
    """向后兼容的入口：优先使用 LangChain agent（若可用），否则回落到 OpenAI 或本地规则。

    参数 `get_stock_info_callable(code)`：必须是一个函数，接受股票代码返回信息（dict/str）。agent 需要它来构建工具。
    """
    # 如果 LangChain 可用且提供了工具函数，优先构建 agent
    if langchain_available and get_stock_info_callable is not None:
        agent, memory = create_agent_with_tools(get_stock_info_callable)
        if agent is not None:
            return generate_daily_report_with_agent(df, agent, memory, top_n=top_n)

    # 否则回退到原有 OpenAI 或本地生成策略
    if OPENAI_KEY and openai_available:
        try:
            openai.api_key = OPENAI_KEY
            preview = df.head(top_n).fillna("").to_dict(orient="records")
            prompt = (
                "请基于下面的涨停股数据，生成一篇面向短线交易者的复盘：\n"
                "要求：1) 概括今日主线与板块；2) 列出 3-6 只重点关注个股并给出逻辑要点；3) 给出仓位与风险提示。\n\n"
                f"数据(仅供参考)：{preview}\n\n请用中文输出，结构清晰。"
            )

            resp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=800,
            )
            text = resp["choices"][0]["message"]["content"]
            return text
        except Exception:
            return _local_summary(df, top_n=top_n)

    return _local_summary(df, top_n=top_n)
