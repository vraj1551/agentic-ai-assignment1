# tasks/task2_async_agent.py

import os
import serpapi
import asyncio
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Setup Gemini + SerpAPI
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def validate_api_keys():
    if not SERPAPI_API_KEY or not GEMINI_API_KEY:
        raise ValueError("Missing SerpAPI or Gemini API Key.")

async def run_async_search_agent(user_prompt: str):
    validate_api_keys()

    # Step 1: Search
    serpapi_client = serpapi.Client(api_key=SERPAPI_API_KEY)
    try:
        data = serpapi_client.search(
            q=user_prompt,
            engine="google",
            hl="en",
            gl="us",
            num=5
        )
        items = data.get("organic_results", [])[:5]
        context = [
            {"title": item.get("title"),
             "snippet": item.get("snippet"),
             "link": item.get("link")}
            for item in items
        ]
    except Exception as e:
        return f"🔴 SerpAPI Error: {e}"

    # Step 2: Summarize with Gemini
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=GEMINI_API_KEY  # ✅ FIX HERE
    )
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are an expert researcher. Provide an in-depth, well-structured answer to the user's prompt only by using the context provided. Give output in Tabular format with rankings if required "),
        ("user", "The user searched for {prompt} and these are the search results: {context}. Provide a meaningful output using your knowledge and the context in a well-structured format. Use bullet points, tabular outputs, and other structures wherever needed.")
    ])

    output_parser = StrOutputParser()
    chain = prompt_template | llm | output_parser

    final_answer = await chain.ainvoke({
        "context": context,
        "prompt": user_prompt
    })
    
    return final_answer
