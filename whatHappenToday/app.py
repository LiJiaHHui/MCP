import streamlit as st
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

# --- 1. 环境设置与加载 ---
# 从 .env 文件加载环境变量 (OPENAI_API_KEY)
load_dotenv()


def check_api_key():
    """检查 OpenAI API Key 是否已设置"""
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OpenAI API Key 未设置！请在项目根目录下创建一个 .env 文件，并写入 OPENAI_API_KEY='sk-...'")
        st.stop()


# --- 2. 核心 AI 逻辑 ---
def generate_summary(dialogue_text: str) -> str:
    """
    使用 LangChain 和大语言模型分析并总结对话。

    Args:
        dialogue_text: 包含对话记录的字符串。

    Returns:
        由 AI 生成的 Markdown 格式的总结报告。
    """

    # --- Prompt Template (提示词模板) ---
    # 这是整个项目的灵魂。我们通过这个模板精确地指导 LLM 如何行动。
    # 一个好的模板是高质量输出的保证。
    prompt_template = """
    你是一名顶级的项目经理和资深技术架构师。你的任务是专业、深入地分析以下团队对话记录。
    请根据对话内容，以结构化、时间线清晰的格式，生成一份完整的事件复盘报告。

    你的报告必须包含以下部分，并严格遵循要求：

    ### 1. 事件概述
    用一句话高度概括整个事件的核心内容。

    ### 2. 问题时间线 (Timeline)
    严格按照时间顺序，详细列出每个关键节点的人物、行为和发现。
    - **10:00 AM (小李):** 发现并报告了什么问题（附上关键日志）。
    - **10:02 AM (老王):** 提出了初步的猜测。
    - **10:05 AM (老王):** 给出了具体的排查指令（如 `top` 命令）。
    - **10:08 AM (小李):** 执行指令并反馈了什么关键信息（如 `top` 截图内容）。
    - **...以此类推，直到问题解决。**

    ### 3. 问题根本原因 (Root Cause Analysis)
    清晰、准确地指出导致问题的技术根本原因。

    ### 4. 解决方案与执行
    描述最终采用的技术解决方案以及由谁完成。

    ### 5. 总结与反思
    提炼出本次事件的关键教训，以及未来可以如何改进以避免类似问题。

    请确保你的报告完全基于以下提供的对话原文，保持客观，不要添加任何对话中未提及的信息。
    请使用 Markdown 格式进行输出，确保格式清晰美观。

    ---
    【对话记录原文】
    {dialogue}
    ---
    """

    # 创建 Prompt 模板实例
    prompt = ChatPromptTemplate.from_template(prompt_template)

    # 初始化 LLM 模型
    # temperature=0.1 使得输出更稳定、可复现
    llm = ChatOpenAI(temperature=0.1, model_name="gpt-4o-mini")

    # 创建 LLMChain，将 Prompt 和 LLM "链接" 在一起
    chain = LLMChain(llm=llm, prompt=prompt)

    # 执行链，并获取结果
    try:
        response = chain.invoke({"dialogue": dialogue_text})
        # 返回结果中的文本部分
        return response.get('text', '抱歉，未能生成报告。')
    except Exception as e:
        st.error(f"调用 API 时出错: {e}")
        return None


# --- 3. Streamlit 用户界面 ---
def main():
    st.set_page_config(page_title="智能工作日志分析器", page_icon="🤖", layout="wide")

    check_api_key()

    st.title("智能工作日志分析器")
    st.caption("?What Happened Today - Powered by LangChain & Streamlit")

    st.markdown("""
    **Hi！** 这是一个用于快速展示的 MCP 项目。
    它能将非结构化的工作对话（如一次线上问题排查的聊天记录）自动梳理成一份带时间线的、结构清晰的复盘报告。
    """)

    # 左右布局
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📋 原始对话记录")
        try:
            with open("conversation_example.txt", "r", encoding="utf-8") as f:
                example_text = f.read()
        except FileNotFoundError:
            example_text = "示例文件 (conversation_example.txt) 未找到。"

        dialogue_input = st.text_area(
            "请将对话记录粘贴于此:",
            value=example_text,
            height=500,
            label_visibility="collapsed"
        )

    with col2:
        st.subheader("✨ AI 生成的复盘报告")

        if 'summary' not in st.session_state:
            st.session_state.summary = "点击左侧按钮开始生成报告..."

        if st.button("🚀 生成总结报告", type="primary", use_container_width=True):
            if not dialogue_input.strip():
                st.warning("请输入对话内容！")
            else:
                with st.spinner("AI 正在深度分析中，请稍候..."):
                    summary_output = generate_summary(dialogue_input)
                    if summary_output:
                        st.session_state.summary = summary_output
                    else:
                        st.session_state.summary = "报告生成失败，请检查 API Key 或网络。"

        # 使用 Markdown 组件展示报告，并设置边框和内边距
        st.markdown(f'{st.session_state.summary}', unsafe_allow_html=True)

        if __name__ == "__main__":
            main()
