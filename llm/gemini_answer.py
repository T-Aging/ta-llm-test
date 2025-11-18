import os
from google import genai
from dotenv import load_dotenv
from typing import Optional, List, Dict, Any

load_dotenv()

# GEMINI_API_KEY 환경 변수가 설정되어 있다면 Client가 자동으로 사용
client = genai.Client()


def gemini_ans(system: str, user: str, model: str = "gemini-2.5-flash", temperature: float = 0.0) -> str:
    contents: List[Dict[str, str]] = [
        {"role": "user", "parts": [{"text": user}]}
    ]

    try:
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config={
                "temperature": temperature,
                "system_instruction": system, # 시스템 역할 지정
            }
        )
        
        return (response.text or "").strip()
    
    except Exception as e:
        print(f"Gemini API 호출 중 오류 발생: {e}")
        return f"ERROR: LLM 호출 실패 - {e}"