import os, sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

import json
from test_llm import (
    gpt_ans_with_latency,
    gemini_ans_with_latency,
)
from load_prompt import load_prompt, render_prompt
from cache.session import l1_snapshot_load, get_menu_prompt
from cache.call_llm import agent_answer

STORE_ID = "001"
MENU_VER = 1
USER_TEXT = "좀 달달하고 부드러운 메뉴 추천해줘. 커피는 아니었으면 좋겠어."

def redis_cache_test():
    print("\n=== Redis 캐시 hit/miss 테스트 ===\n")
    print(f"- store_id: {STORE_ID}, menu_ver: {MENU_VER}")
    print(f"- user_text: {USER_TEXT}\n")

    # 1차 호출: 반드시 MISS (또는 아직 캐시가 없다면)
    start = time.time()
    resp1 = agent_answer(STORE_ID, MENU_VER, USER_TEXT)
    t1 = time.time() - start

    print("[1차 호출]")
    print(f"  latency: {t1:.4f} sec")
    print(f"  cache_hit: {resp1.get('cache_hit')}")
    print(f"  reply: {resp1.get('reply')}\n")

    # 2차 호출: 같은 질문 → fuzzy/exact 매칭으로 HIT 기대
    start = time.time()
    resp2 = agent_answer(STORE_ID, MENU_VER, USER_TEXT)
    t2 = time.time() - start

    print("[2차 호출]")
    print(f"  latency: {t2:.4f} sec")
    print(f"  cache_hit: {resp2.get('cache_hit')}")
    print(f"  reply: {resp2.get('reply')}\n")

    print("=== Redis 캐시 테스트 종료 ===\n")
    return resp1, resp2
    

l1_snapshot_load("001", 1)
menu_prompt = get_menu_prompt("001", 1)
user_text="커피는 싫고 좀 달달하고 부드러운 메뉴 추천"

SYSTEM_PROMPT_NAME = "system_prompt.txt"
USER_PROMPT_NAME   = "ans_prompt.txt"

def llm_comparison():
    print("\n=== LLM 모델 비교 테스트 시작 ===\n")
    print(f"- 사용자 질문: {user_text}")
    print(f"- 메뉴 스냅샷: {menu_prompt}\n")

    results = {}

    # 1) GPT-4o 테스트
    try:
        gpt_reply, gpt_latency = gpt_ans_with_latency(
            system=load_prompt(SYSTEM_PROMPT_NAME),
            user=render_prompt(USER_PROMPT_NAME, menu_prompt=menu_prompt, user_text=user_text),
            model="gpt-4o",
            temperature=0.0
        )
        results["gpt-4o"] = {
            "reply": gpt_reply,
            "latency": gpt_latency
        }
    except Exception as e:
        results["gpt-4o"] = {
            "error": str(e)
        }

    # 2) Gemini 테스트
    try:
        gem_reply, gem_latency = gemini_ans_with_latency(
            system=load_prompt(SYSTEM_PROMPT_NAME),
            user=render_prompt(USER_PROMPT_NAME, menu_prompt=menu_prompt, user_text=user_text),
            model="gemini-2.5-flash",
            temperature=0.0
        )
        results["gemini-2.5-flash"] = {
            "reply": gem_reply,
            "latency": gem_latency
        }
    except Exception as e:
        results["gemini-2.5-flash"] = {
            "error": str(e)
        }

    print("\n=== 모델 비교 결과 ===\n")

    for model_name, info in results.items():
        print(f"▶ {model_name}")
        print("-" * 40)

        if "error" in info:
            print("Error:", info["error"])
        else:
            print(f"Latency: {info['latency']:.4f} sec")
            print(f"Reply:\n{info['reply']}\n")

    print("\n=== 테스트 종료 ===\n")
    return results


if __name__ == "__main__":
    llm_comparison()
    redis_cache_test()
