import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

# Page configuration
st.set_page_config(
    page_title="AI Chat Assistant",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for attractive styling
st.markdown("""
<style>
    /* Main background - Modern dark theme */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        min-height: 100vh;
    }

    /* Chat message styling */
    .stChatMessage {
        border-radius: 20px;
        padding: 15px 20px;
        margin: 12px 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }

    /* User message background */
    .stChatMessage[data-testid="stChatMessage"]:has(.stMarkdown p) {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    }

    /* Assistant message background */
    .stChatMessage[data-testid="stChatMessage"]:has(.stMarkdown p) {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d4a6f 100%);
    }

    /* Chat input container */
    .stChatInputContainer {
        padding: 20px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 25px;
        margin-bottom: 20px;
    }

    /* Chat input styling */
    div[data-testid="stChatInput"] textarea {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        border: 2px solid rgba(99, 102, 241, 0.3);
        color: #1a1a2e;
        font-size: 16px;
    }

    div[data-testid="stChatInput"] textarea:focus {
        border-color: #6366f1;
        box-shadow: 0 0 20px rgba(99, 102, 241, 0.3);
    }

    /* Title styling */
    h1 {
        text-align: center;
        background: linear-gradient(90deg, #f472b6, #a78bfa, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5em;
        font-weight: 700;
        text-shadow: 0 0 40px rgba(167, 139, 250, 0.3);
        margin-bottom: 10px;
    }

    /* Subtitle */
    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1.1em;
        margin-bottom: 30px;
    }

    /* Welcome cards */
    .welcome-container {
        display: flex;
        justify-content: center;
        gap: 15px;
        flex-wrap: wrap;
        margin: 20px 0;
    }

    .welcome-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.2) 0%, rgba(139, 92, 246, 0.2) 100%);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 15px;
        padding: 20px;
        cursor: pointer;
        transition: all 0.3s ease;
        min-width: 150px;
        text-align: center;
    }

    .welcome-card:hover {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.4) 0%, rgba(139, 92, 246, 0.4) 100%);
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(99, 102, 241, 0.3);
    }

    .welcome-card .icon {
        font-size: 2em;
        margin-bottom: 10px;
    }

    .welcome-card .text {
        color: #e2e8f0;
        font-size: 0.95em;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    .stSidebar h2 {
        color: #f472b6;
        font-size: 1.5em;
    }

    .stSidebar h3 {
        color: #a78bfa;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.5);
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(139, 92, 246, 0.15) 100%);
        border-radius: 15px;
        padding: 15px;
        border: 1px solid rgba(99, 102, 241, 0.3);
        margin: 10px 0;
    }

    .metric-value {
        font-size: 2em;
        font-weight: 700;
        color: #a78bfa;
    }

    .metric-label {
        color: #94a3b8;
        font-size: 0.9em;
    }

    /* Slider styling */
    .stSlider > div > div {
        background: rgba(99, 102, 241, 0.3);
    }

    /* Selectbox styling */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 10px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #64748b;
        padding: 20px;
        margin-top: 30px;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.85em;
        font-weight: 600;
        margin: 5px 0;
    }

    .status-good {
        background: rgba(34, 197, 94, 0.2);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }

    .status-warning {
        background: rgba(251, 191, 36, 0.2);
        color: #fbbf24;
        border: 1px solid rgba(251, 191, 36, 0.3);
    }

    .status-danger {
        background: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }

    /* Info boxes */
    .info-box {
        background: rgba(59, 130, 246, 0.1);
        border-left: 4px solid #60a5fa;
        border-radius: 10px;
        padding: 15px;
        margin: 15px 0;
    }

    .info-box-title {
        color: #60a5fa;
        font-weight: 600;
        margin-bottom: 8px;
    }

    .info-box-content {
        color: #94a3b8;
        font-size: 0.95em;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "max_turns" not in st.session_state:
    st.session_state.max_turns = 10
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = "You are a helpful, friendly, and knowledgeable AI assistant. You provide clear, accurate, and well-explained answers."
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.7
if "model_name" not in st.session_state:
    st.session_state.model_name = "minimax-m2.5:cloud"

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("")

    # Model selection
    st.markdown("### 🤖 Model")
    model_options = ["minimax-m2.5:cloud", "llama3.2:cloud", "qwen2.5:cloud"]
    selected_model = st.selectbox(
        "Choose AI Model",
        options=model_options,
        index=model_options.index(st.session_state.model_name) if st.session_state.model_name in model_options else 0,
        label_visibility="collapsed"
    )
    st.session_state.model_name = selected_model

    st.markdown("")

    # Temperature slider
    st.markdown("### 🌡️ Creativity Level")
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.temperature,
        step=0.1,
        help="Lower = more focused, Higher = more creative"
    )
    st.session_state.temperature = temperature

    # Temperature indicator
    if temperature <= 0.3:
        temp_desc = "📊 **Precise** - Best for factual answers"
    elif temperature <= 0.6:
        temp_desc = "⚖️ **Balanced** - Good for general chat"
    else:
        temp_desc = "🎨 **Creative** - Best for brainstorming"
    st.markdown(f"*{temp_desc}*")

    st.markdown("")
    st.markdown("---")
    st.markdown("")

    # System prompt
    st.markdown("### 📝 System Prompt")
    system_prompt = st.text_area(
        "Customize AI behavior",
        value=st.session_state.system_prompt,
        height=100,
        label_visibility="collapsed"
    )
    st.session_state.system_prompt = system_prompt

    st.markdown("")
    st.markdown("---")
    st.markdown("")

    # Chat stats
    st.markdown("### 📊 Chat Statistics")
    current_turn = len(st.session_state.chat_history) // 2
    remaining = st.session_state.max_turns - current_turn

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{current_turn}</div>
        <div class="metric-label">Turns Used</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{remaining}</div>
        <div class="metric-label">Turns Remaining</div>
    </div>
    """, unsafe_allow_html=True)

    # Status badge
    if remaining > 5:
        status_class = "status-good"
        status_text = "✅ Good"
    elif remaining > 2:
        status_class = "status-warning"
        status_text = "⚠️ Getting Low"
    else:
        status_class = "status-danger"
        status_text = "🔴 Almost Full"

    st.markdown(f'<span class="status-badge {status_class}">{status_text}</span>', unsafe_allow_html=True)

    st.markdown("")
    st.markdown("---")
    st.markdown("")

    # Actions
    st.markdown("### 🛠️ Actions")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    with col2:
        if st.button("📋 Export", use_container_width=True):
            st.info("Export feature coming soon!")

    st.markdown("")
    st.markdown("---")
    st.markdown("")

    # Tips
    with st.expander("💡 Tips & Tricks"):
        st.markdown("""
        - **Be specific** with your questions for better answers
        - **Adjust temperature** for different tasks
        - **Clear chat** when switching topics
        - **Customize system prompt** for specific needs
        """)

# Main content
st.markdown('<h1>✨ AI Chat Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Your intelligent companion for questions, creativity, and more</p>', unsafe_allow_html=True)

# Quick action cards (only show when chat is empty)
if len(st.session_state.chat_history) == 0:
    st.markdown('<div class="welcome-container">', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="welcome-card" onclick="document.querySelector('[data-testid="stChatInput"] textarea').value='Explain quantum computing in simple terms'">
            <div class="icon">🔬</div>
            <div class="text">Explain a Concept</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="welcome-card">
            <div class="icon">💻</div>
            <div class="text">Write Code</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="welcome-card">
            <div class="icon">✍️</div>
            <div class="text">Creative Writing</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="welcome-card">
            <div class="icon">🧮</div>
            <div class="text">Solve a Problem</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Quick suggestion buttons
    st.markdown("### 💭 Try asking:")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔬 Explain quantum computing", use_container_width=True, key="q1"):
            st.session_state.pending_question = "Explain quantum computing in simple terms"
            st.rerun()
    with col2:
        if st.button("💻 Write a Python function", use_container_width=True, key="q2"):
            st.session_state.pending_question = "Write a Python function to sort a list"
            st.rerun()
    with col3:
        if st.button("📝 Give productivity tips", use_container_width=True, key="q3"):
            st.session_state.pending_question = "Give me 5 productivity tips for remote work"
            st.rerun()

    st.markdown("")

# Info box
if len(st.session_state.chat_history) == 0:
    st.markdown("""
    <div class="info-box">
        <div class="info-box-title">👋 Welcome!</div>
        <div class="info-box-content">
            Start a conversation by typing a message below or try one of the suggestions above.
            Customize settings in the sidebar to tailor responses to your needs.
        </div>
    </div>
    """, unsafe_allow_html=True)

# Display chat history
for i, message in enumerate(st.session_state.chat_history):
    if isinstance(message, HumanMessage):
        with st.chat_message("user", avatar="👤"):
            st.markdown(f"<span style='color: #f0f0f0'>{message.content}</span>", unsafe_allow_html=True)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(f"<span style='color: #e0e7ff'>{message.content}</span>", unsafe_allow_html=True)

# Check context window status
def check_context_status():
    current_turn = len(st.session_state.chat_history) // 2
    remaining = st.session_state.max_turns - current_turn
    return current_turn, remaining

# Handle pending question from quick actions
if "pending_question" in st.session_state and st.session_state.pending_question:
    prompt_text = st.session_state.pending_question
    st.session_state.pending_question = None
else:
    prompt_text = None

# Chat input
user_input = st.chat_input("Type your message here...", key="chat_input")
if user_input or prompt_text:
    current_prompt = user_input if user_input else prompt_text
    current_turn, remaining = check_context_status()

    if current_turn >= st.session_state.max_turns:
        with st.chat_message("user", avatar="👤"):
            st.markdown(f"<span style='color: #f0f0f0'>{current_prompt}</span>", unsafe_allow_html=True)

        with st.chat_message("assistant", avatar="⚠️"):
            st.warning(
                "**Context window is full!** The AI may not follow the previous thread properly. "
                "Please clear the chat history from the sidebar to start fresh."
            )
    else:
        # Display user message
        with st.chat_message("user", avatar="👤"):
            st.markdown(f"<span style='color: #f0f0f0'>{current_prompt}</span>", unsafe_allow_html=True)

        # Generate and display AI response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("🤔 Thinking..."):
                try:
                    # Initialize LLM with current settings
                    llm = ChatOllama(
                        model=st.session_state.model_name,
                        temperature=st.session_state.temperature
                    )
                    prompt_template = ChatPromptTemplate.from_messages([
                        ("system", st.session_state.system_prompt),
                        MessagesPlaceholder(variable_name="chat_history"),
                        ("human", "{question}")
                    ])
                    chain = prompt_template | llm | StrOutputParser()

                    response = chain.invoke({
                        "question": current_prompt,
                        "chat_history": st.session_state.chat_history
                    })

                    # Add warning if running low on turns
                    if remaining <= 2:
                        response += f"\n\n⚠️ *Only {remaining} turn(s) left before the context window fills up.*"

                    st.markdown(f"<span style='color: #e0e7ff'>{response}</span>", unsafe_allow_html=True)

                    # Update chat history
                    st.session_state.chat_history.append(HumanMessage(content=current_prompt))
                    st.session_state.chat_history.append(AIMessage(content=response))

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

# Footer
st.markdown("""
<div class="footer">
    <p>Powered by <strong>LangChain</strong> & <strong>Ollama</strong> | Built with <strong>Streamlit</strong></p>
</div>
""", unsafe_allow_html=True)
