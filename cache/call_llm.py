import os, hashlib
from typing import Dict
import redis

from cache.session import l1_snapshot_load, get_menu_prompt
from llm.generate_ans import gen_ans

# --- Redis 접속 ---
_r = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", "6379")),
    db=int(os.getenv("REDIS_DB", "0")),
    decode_responses=True,
)

REDIS_TTL = int(os.getenv("REDIS_FAQ_TTL", "86400"))  # 1일

def agent_answer(store_id: str, menu_ver: int, user_text: str) -> Dict:
    # 1) L1 조회
    l1_snapshot_load(store_id, menu_ver)
    menu_prompt = get_menu_prompt(store_id, menu_ver)

    # 2) L2 캐시 조회
    # 

    # 3) 캐시 미스 → LLM 호출
    answer = gen_ans(menu_prompt=menu_prompt, user_text=user_text)

    # 4) L2 캐시에 저장 (임베딩 고려 필요)
    # 

    return {
        "reply": answer,
        "store_id": store_id,
        "menu_version": menu_ver,
    }
