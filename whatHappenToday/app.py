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
    严格按照时间顺序，简洁列出每个关键且重要节点的人物和行为。
    - **10:00 AM (小李):** 发现并报告了什么问题（附上关键日志）。
    - **10:02~10:05 AM (老王):** 提出了初步的猜测。给出了具体的排查指令（如 `top` 命令）。
    - **10:08 AM (小李):** 执行指令并反馈了什么关键信息（如 `top` 截图内容）。
    - **...以此类推，直到问题解决。**

    ### 3. 根本原因 (Root Cause Analysis)
    一句话清晰、准确地指出导致问题的技术根本原因。

    ### 4. 解决方案
    描述最终采用的技术解决方案以及由谁完成。

    ### 5. 总结与反思
    一句话提炼出本次事件的关键教训，以及未来可以如何改进以避免类似问题。
    请确保你的报告完全基于以下提供的对话原文，保持客观，不要添加任何对话中未提及的信息。
    请使用 Markdown 格式进行输出，确保格式清晰美观。不出现代码块，用空格+斜体代替。

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
    st.set_page_config(page_title="What Happened Today", page_icon="❓", layout="wide")
    check_api_key()

    # st.title("What Happened Today")
    # st.caption("")
    # 大标题
    st.markdown('<p class="main-title">❓What Happened Today</p>', unsafe_allow_html=True)

    # 描述
    st.markdown('<p class="description">Extract structured report from any chat — just by copy</p>', unsafe_allow_html=True)
    # st.markdown("""
    # **Hi！** 这是一个用于快速梳理总结工作对话的MCP项目。
    # 它能将非结构化的工作对话（如一次线上问题排查的聊天记录）自动梳理成一份带时间线的、结构清晰的复盘报告。😀
    # """)

    # 左右布局
    col1, col2 = st.columns(2)



    with col1:
        st.subheader("📋 Original conversation")
        try:
            with open("conversation_example.txt", "r", encoding="utf-8") as f:
                example_text = f.read()
        except FileNotFoundError:
            example_text = "示例文件 (conversation_example.txt) 未找到。"

        dialogue_input = st.text_area(
            "请将对话记录粘贴于此:",
            value=example_text,
            height=700,
            label_visibility="collapsed"
        )

    with col2:
        st.subheader("✨ AI-generated review reports")

        if 'summary' not in st.session_state:
            st.session_state.summary = "empty..."

        if st.button("Start Now", type="primary", use_container_width=True):
            if not dialogue_input.strip():
                st.warning("Please input chat~")
            else:
                with st.spinner("AI is in the process of deep analysis, please wait..."):
                    summary_output = generate_summary(dialogue_input)
                    if summary_output:
                        st.session_state.summary = summary_output
                    else:
                        st.session_state.summary = "Report generation failed, check the API Key or network."

        # 使用 Markdown 组件展示报告，并设置边框和内边距
        st.markdown(f'{st.session_state.summary}', unsafe_allow_html=True)
        st.markdown("""
            <style>
            /* 主标题样式 */
            .main-title {
                font-size: 56px !important;
                font-weight: bold;
                text-align: center;
                color: #000; /* 白色字体 */
                padding-top: 40px;
            }
            /* 副标题/描述样式 */
            .description {
                font-size: 20px !important;
                text-align: center;
                color: #B0B0B0; /* 灰色字体 */
                padding-bottom: 40px;
            }
            /* Streamlit 主体背景色 */
            .stApp {
                background-color:  #eeebe8; /* 黑色背景 */
            }
            /* Tab 标签样式 */
            .stTabs [data-baseweb="tab-list"] {
                justify-content: center;
            }
            .stTabs [data-baseweb="tab"] {
                height: 50px;
                white-space: pre-wrap;
                background-color: #1a1a1a;
                border-radius: 8px;
                margin: 0 5px;
                color: #f2ddcc; /* Tab 未选中时字体颜色 */
                
            }
            .stTabs [aria-selected="true"] {
                background-color: #333333;
                color: #f2ddcc; /* Tab 选中时字体颜色 */
            }
            .st-emotion-cache-bfgnao p{
                font-weight: 650;
                font-size: 1.0625rem;
                line-height: 1.625rem;
                font-variation-settings: "opsz" 40, "wght" 650;
                font-synthesis: none;
            }
            </style>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
