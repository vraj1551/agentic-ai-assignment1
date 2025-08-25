import os
import serpapi
import google.generativeai as genai      # ✅ correct import
from dotenv import load_dotenv

load_dotenv()

# API key setup
serpapi_client = serpapi.Client(api_key=os.getenv("SERPAPI_API_KEY"))
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


def fetch_search_results(query, num_results=5):
    try:
        data = serpapi_client.search(
            q=query,
            engine="google",
            hl="en",
            gl="us",
            num=num_results
        )
    except Exception as e:
        msg = str(e)
        if "401" in msg:
            raise RuntimeError("SerpAPI Unauthorized (401). Check your SERPAPI_API_KEY.") from e
        else:
            raise RuntimeError(f"SerpAPI Error: {msg}") from e

    items = data.get("organic_results", [])[:num_results]
    return [
        {"title": item.get("title"),
         "snippet": item.get("snippet"),
         "link": item.get("link")}
        for item in items
    ]

def summarize_with_gemini(results, query):
    bullets = "\n".join(
        f"- **{r['title']}** ({r['link']}): {r['snippet']}"
        for r in results
    )
    prompt = (
        f"You are an expert research assistant. The user searched for “{query}” "
        f"and these are the top results:\n\n{bullets}\n\n"
        "Please provide:\n"
        "1. A one-sentence overview\n"
        "2. Three bullet-point key takeaways\n"
        "3. Tabular output of listings if required.\n"
        "4. A suggestion for next steps or deeper reading."
    )
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text
