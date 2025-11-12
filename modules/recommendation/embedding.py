import os, numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(encoding="utf-8-sig")
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY 누락: .env 확인하세요")

client = OpenAI(api_key=api_key)
EMBED_MODEL = "text-embedding-3-small"

def embed_texts(texts: list[str]) -> list[np.ndarray]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [np.array(d.embedding, dtype=np.float32) for d in resp.data]

def embed_text(text: str) -> np.ndarray:
    return embed_texts([text])[0]
