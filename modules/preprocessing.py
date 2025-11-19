# reprocess.py
import os, re
import pandas as pd
from konlpy.tag import Okt

# --- 경로 ---
BASE = os.path.dirname(__file__)
DATA = os.path.join(BASE, "data")

# --- 불용어 로드 ---
def load_stopwords(path=None):
    if path is None: path = os.path.join(DATA, "stopwords.txt")
    sw = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                w = line.strip()
                if w and not w.startswith("#"):
                    sw.append(w)
    return sw

# --- 사용자 사전 로드 → Okt에 주입 ---
def build_okt(user_dict_path=None):
    if user_dict_path is None:
        user_dict_path = os.path.join(DATA, "user_dict.txt")
    okt = Okt()
    if os.path.exists(user_dict_path):
        with open(user_dict_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"): continue
                parts = re.split(r"[,|\t]\s*", s)
                if len(parts) == 2:
                    word, tag = parts[0].strip(), parts[1].strip()
                    if word and tag:
                        try: okt.add_dictionary(word, tag)
                        except: pass
    return okt

# --- 토크나이저 ---
def tokenizer(text, okt, stopwords):
    if not isinstance(text, str): return ""
    pairs = okt.pos(text)
    keep = []
    for w, t in pairs:
        if t not in ["Josa","Punctuation","Verb","Number","Foreign"]:
            if w not in stopwords:
                keep.append(w)
    return " ".join(keep)

if __name__ == "__main__":
    # 데이터 로드
    df = pd.read_csv(os.path.join(DATA, "DataFrame.csv"))
    stopwords = load_stopwords()
    okt = build_okt()

    # 1) 제목/책소개 모두 토큰화
    df_token = pd.DataFrame({
        "제목": df["제목"].apply(lambda x: tokenizer(x, okt, stopwords)),
        "책소개": df["책소개"].apply(lambda x: tokenizer(x, okt, stopwords))
    })
    # 2) 책소개만 토큰화 (제목은 원문 유지)
    df_token1 = pd.DataFrame({
        "제목": df["제목"],
        "책소개": df["책소개"].apply(lambda x: tokenizer(x, okt, stopwords))
    })

    # 저장
    df_token.to_json(os.path.join(DATA, "책_제목_책소개_토큰화.json"),
                     orient="records", force_ascii=False, indent=2)
    df_token1.to_json(os.path.join(DATA, "책소개만_토큰화.json"),
                      orient="records", force_ascii=False, indent=2)

    print("JSON 2개 생성 완료")
