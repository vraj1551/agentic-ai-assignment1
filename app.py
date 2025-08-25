# app.py

import streamlit as st
import asyncio
from tasks.task1_search_agent import fetch_search_results, summarize_with_gemini
from tasks.task2_async_agent import run_async_search_agent

# Sidebar Task Selection
st.sidebar.title("🧠 AI Agent Tasks")
task = st.sidebar.radio("Select Task", [
    "Web Search Agent",
    "Web Search Agent (Async)",
    "System + User Prompt",
    "Using RAG"
])

st.title("🤖 Multi-Agent Search Assistant")
st.subheader(f"Current Task: {task}")

# --- Session State Management ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {
        "Web Search Agent": [],
        "Web Search Agent (Async)": [],
        "System + User Prompt": [],
        "Using RAG": []
    }

# Get current chat history for selected task
current_history = st.session_state.chat_history[task]

# --- Display Chat History for selected task ---
for msg in current_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat Input Logic ---
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

            else:
                response = f"🚧 This task ({task}) is not implemented yet."

        st.markdown(response)
        current_history.append({"role": "assistant", "content": response})
