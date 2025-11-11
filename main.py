from fastapi import FastAPI
from cache.call_llm import agent_answer
from cache.session import l1_snapshot_load

app = FastAPI(title="AI Agent Test", version="1.0.0")

@app.get("/")
def root():
    return {"message": "AI Agent API is running"}

# L1 조회
@app.post("/session/start")
def start_session(store_id: str, menu_version: int):
    snapshot = l1_snapshot_load(store_id, menu_version)
    return {"status": "L1 cached", "menu_count": len(snapshot["menus"])}

# 사용자 발화 이후, L1(redis) 캐시 Hit/Miss에 따른 답변 생성
@app.post("/converse")
def converse(store_id: str, menu_version: int, text: str):
    result = agent_answer(store_id, menu_version, text)
    return result

# from fastapi import FastAPI
# from dotenv import load_dotenv
# import os
# import redis

# load_dotenv()

# app = FastAPI()

# # Redis 연결
# r = redis.Redis(
#     host=os.getenv("REDIS_HOST"),
#     port=int(os.getenv("REDIS_PORT")),
#     db=int(os.getenv("REDIS_DB"))
# )

# @app.get("/ping")
# def ping():
#     r.set("hello", "redis")
#     return {"ping": r.get("hello")}