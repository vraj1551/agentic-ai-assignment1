# app.py

import streamlit as st
import asyncio
from tasks.task1_search_agent import fetch_search_results, summarize_with_gemini
from tasks.task2_async_agent import run_async_search_agent
from tasks.task3_custom_prompt import run_chat
from tasks.task4_rag_simple import rag_with_groq  # ✅ Added for Task 4

# --- Sidebar ---
st.sidebar.title("AI Agent Tasks")  # ✅ Removed 🧠
st.sidebar.markdown("**22000884 - Vraj Vyas**")  # ✅ Added name + roll number

# Task Buttons instead of Radio
task_options = [
    "Web Search Agent",
    "Web Search Agent (Async)",
    "System + User Prompt",
    "Using RAG"
]

if "selected_task" not in st.session_state:
    st.session_state.selected_task = task_options[0]

for option in task_options:
    if st.sidebar.button(option, use_container_width=True):
        st.session_state.selected_task = option

task = st.session_state.selected_task  # ✅ Task selection persists

# --- Main Display ---
st.title("🤖 Multi-Agent Search Assistant")
st.subheader(f"Current Task: {task}")

# Session state for task-specific chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {opt: [] for opt in task_options}

current_history = st.session_state.chat_history[task]

# --- Chat Display ---
for msg in current_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- TASK LOGIC ---

if task == "System + User Prompt":
    with st.form("prompt_form"):
        system_prompt = st.text_area("🧠 System Prompt", placeholder="Define AI behavior here...", height=120)
        user_prompt = st.text_area("👤 User Prompt", placeholder="Ask your question...", height=100)
        submitted = st.form_submit_button("Generate Response")

    if submitted and user_prompt:
        current_history.append({"role": "user", "content": f"**System Prompt:**\n```{system_prompt}```\n\n**User Prompt:**\n{user_prompt}"})
        with st.chat_message("user"):
            st.markdown(f"**System Prompt:**\n```{system_prompt}```\n\n**User Prompt:**\n{user_prompt}")

        with st.chat_message("assistant"):
            with st.spinner("Generating..."):
                response, error = run_chat(system_prompt, user_prompt)
                if error:
                    st.error(error)
                    st.markdown(f"Raw Output:\n```{response}```")
                    current_history.append({"role": "assistant", "content": f"{error}\n\n```{response}```"})
                else:
                    formatted = f"### 🍽 Summary:\n{response.summary}\n\n### ✅ Steps:\n" + \
                                "\n".join([f"- {s}" for s in response.steps]) + \
                                "\n\n### 💡 Tips:\n" + "\n".join([f"- {t}" for t in response.tips])
                    st.markdown(formatted)
                    current_history.append({"role": "assistant", "content": formatted})

else:
    # ChatGPT-style prompt box
    if prompt := st.chat_input("What would you like to know?"):
        current_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Working on your request..."):
                if task == "Web Search Agent":
                    try:
                        results = fetch_search_results(prompt)
                        response = summarize_with_gemini(results, prompt)
                    except Exception as e:
                        response = f"❌ Error: {e}"

                elif task == "Web Search Agent (Async)":
                    try:
                        response = asyncio.run(run_async_search_agent(prompt))
                    except Exception as e:
                        response = f"❌ Error: {e}"

                elif task == "Using RAG":  # ✅ Task 4
                    try:
                        response = rag_with_groq(prompt)
                    except Exception as e:
                        response = f"❌ RAG Error: {e}"

                else:
                    response = f"🚧 This task ({task}) is not implemented yet."

            st.markdown(response)
            current_history.append({"role": "assistant", "content": response})
