import time
import json
import os
from openai import OpenAI
import os
from dotenv import load_dotenv
from google import genai
from typing import Optional, List, Dict, Any, Tuple

load_dotenv()

# OpenAI Client 초기화
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Gemini Client 초기화
gemini_client = genai.Client()

# Gemini Latency
def gemini_ans_with_latency(system: str, user: str, model: str = "gemini-2.5-flash", temperature: float = 0.0) -> Tuple[str, float]:
    contents: List[Dict[str, Any]] = [
        {"role": "user", "parts": [{"text": user}]}
    ]
    
    start_time = time.time()
    response = gemini_client.models.generate_content(
        model=model,
        contents=contents,
        config={
            "temperature": temperature,
            "system_instruction": system,
        }
    )
    end_time = time.time()
        
    latency = end_time - start_time
    return (response.text or "").strip(), latency

# GPT Latency
def gpt_ans_with_latency(system: str, user: str, model: str = "gpt-4o", temperature: float = 0.0) -> Tuple[str, float]:
    start_time = time.time()
    resp = openai_client.chat.completions.create(
        model=model,
        messages=[
                {"role": "user", "content": user},
        ],
        temperature=temperature,
    )
    end_time = time.time()

    latency = end_time - start_time
    return (resp.choices[0].message.content or "").strip(), latency

def check_json_robustness(text: str) -> bool:
    try:
        # LLM은 종종 텍스트와 JSON을 함께 출력하므로, JSON 블록을 찾아야 합니다.
        # 이 예시에서는 응답의 마지막 라인에 JSON이 있다고 가정하고 간단히 처리합니다.
        json_line = text.strip().split('\n')[-1]
        json.loads(json_line)
        return True
    except:
        return False