import os
import json
import re
from groq import Groq
from pydantic import BaseModel
from typing import List, Optional, Tuple, Union

class RecipeResponse(BaseModel):
    summary: str
    steps: List[str]
    tips: Optional[List[str]] = []

API_KEY = os.getenv("GROQ_API_KEY")  # ✅ Secure: No hardcoded key
client = Groq(api_key=API_KEY)

FORMAT_INSTRUCTION = """
You MUST reply only in valid JSON using the following format:
{
  "summary": "brief summary of your idea",
  "steps": ["step 1", "step 2", "..."],
  "tips": ["tip 1", "tip 2", "..."]
}
Do not add any markdown or explanation. Just reply with a raw JSON object.
"""

def try_parse_json(raw: str) -> Tuple[Union[RecipeResponse, str], Union[str, None]]:
    try:
        parsed = RecipeResponse(**json.loads(raw))
        return parsed, None
    except Exception:
        match = re.search(r"{.*}", raw, re.DOTALL)
        if match:
            try:
                parsed = RecipeResponse(**json.loads(match.group()))
                return parsed, None
            except Exception as e:
                return raw, f"⚠️ Parsing failed after cleanup attempt: {e}"
        return raw, "⚠️ Could not find valid JSON format in the response."

def run_chat(user_system_prompt: str, user_prompt: str) -> Tuple[Union[RecipeResponse, str], Union[str, None]]:
    full_system_prompt = (user_system_prompt or "You are a helpful assistant.") + "\n\n" + FORMAT_INSTRUCTION

    messages = [
        {"role": "system", "content": full_system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        res = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages,
            temperature=0.7,
        )
    except Exception as e:
        return "", f"❌ Groq API Error: {e}"

    raw_output = res.choices[0].message.content.strip()
    return try_parse_json(raw_output)
