# app.py
import os
from typing import List, Dict

from flask import Flask, request, jsonify
from google.cloud import firestore  # pip install google-cloud-firestore

from modules.recommendation.service import compute_recommendations
from modules.preprocessing import load_stopwords, build_okt, tokenizer

from modules.recommendation.dto import (
    RecommendRequest,
    RecommendResponse,
    BookRecommendationDTO,
)

# ============================================================
# 0. 전역 설정: Firestore, 형태소 분석기(Okt), 불용어 로딩
# ============================================================

db = firestore.Client()

# ⚠ reprocess.py에서 정의한 함수 사용
STOPWORDS = load_stopwords()
OKT = build_okt()


def tokenize_book_text(book: Dict) -> str:
    """
    책 1권에 대해 title + author + description을 합쳐서
    reprocess.tokenizer()로 토큰화한 문자열을 반환.
    """
    # None 방지용 기본값
    title = book.get("title", "") or ""
    author = book.get("author", "") or ""
    desc = book.get("description", "") or ""

    raw_text = f"{title} {author} {desc}"
    return tokenizer(raw_text, OKT, STOPWORDS)


def attach_tokenized_content(books: List[Dict]) -> None:
    """
    각 book dict에 'content' 필드를 추가.
    service.compute_recommendations()에서 이 필드를 우선적으로 사용하도록 수정할 예정.
    """
    for b in books:
        b["content"] = tokenize_book_text(b)


# ============================================================
# 1. Books 카탈로그 전체 가져오기
# ============================================================

def fetch_catalog_books() -> List[Dict]:
    """
    Books 컬렉션 전체를 가져와서 카탈로그 리스트로 반환.

    ⚠ 전제 (추측입니다):
    - 컬렉션 이름: "Books"
    - 필드:
        isbn, title, author, publisher,
        publishDate, category, description,
        imageUrl, pageCount, popularity_count(선택)
    """
    catalog: List[Dict] = []

    books_col = db.collection("Books")  # <- 실제 컬렉션 이름 확인 후 수정

    docs = books_col.stream()
    for doc in docs:
        data = doc.to_dict() or {}

        # 여기서 'id'는 추천 알고리즘에서 사용하는 식별자
        # isbn을 그대로 id로 사용 (중복되지 않는다고 가정)
        book = {
            "id": data.get("isbn") or doc.id,  # isbn이 비어있으면 문서 id 사용
            "isbn": data.get("isbn", ""),
            "title": data.get("title", ""),
            "author": data.get("author", ""),
            "publisher": data.get("publisher", ""),
            "publishDate": data.get("publishDate", ""),
            "category": data.get("category", ""),
            "description": data.get("description", ""),
            "imageUrl": data.get("imageUrl", ""),
            "pageCount": data.get("pageCount", 0),
            "popularity_count": data.get("popularity_count", 0),
        }
        catalog.append(book)

    return catalog


def build_catalog_index_by_isbn(catalog: List[Dict]) -> Dict[str, Dict]:
    """
    isbn -> book dict 형태의 인덱스를 만들어서,
    사용자 저장 isbn을 빠르게 매핑할 수 있게 한다.
    """
    index: Dict[str, Dict] = {}
    for b in catalog:
        isbn = b.get("isbn")
        if isbn:
            index[isbn] = b
    return index


# ============================================================
# 2. 사용자 저장 isbn 목록 가져오기
# ============================================================

def fetch_user_saved_isbns(user_id: str) -> List[str]:
    """
    사용자가 저장한 책의 isbn 목록만 Firestore에서 가져온다.

    ⚠ 전제 (추측입니다):
    - 컬렉션 구조:
        users/{user_id}/savedBooks/{doc_id}
        각 문서 필드: { isbn: "XXXXXXXXXX" }
    """
    isbns: List[str] = []

    # 실제 구조: users, savedBooks 이름은 프로젝트에 맞게 수정
    saved_col = db.collection("users").document(user_id).collection("savedBooks")
    docs = saved_col.stream()
    for doc in docs:
        data = doc.to_dict() or {}
        isbn = data.get("isbn")
        if isbn:
            isbns.append(isbn)

    return isbns


def build_user_books_from_isbns(
    user_isbns: List[str],
    catalog_index: Dict[str, Dict],
) -> List[Dict]:
    """
    사용자 저장 isbn 리스트를, Books 카탈로그에서 실제 도서 정보로 변환.

    - 결과는 service.compute_recommendations()에서 요구하는
      user_books 형식([{id, title, author, description, ...}])으로 맞춤.
    """
    user_books: List[Dict] = []

    for isbn in user_isbns:
        book = catalog_index.get(isbn)
        if not book:
            # Books 컬렉션에 없는 isbn일 경우 스킵
            continue

        # 이미 catalog에서 'id'를 isbn으로 설정했으므로 그대로 사용
        user_books.append(
            {
                "id": book["id"],
                "isbn": book.get("isbn", ""),
                "title": book.get("title", ""),
                "author": book.get("author", ""),
                "description": book.get("description", ""),
                # 필요하면 다른 필드도 포함 가능
            }
        )

    return user_books


# ============================================================
# 3. Flask 라우트
# ============================================================

app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"}), 200


@app.route("/api/recommend", methods=["POST"])
def recommend_books():
    body = request.get_json(silent=True) or {}

    # 1) 요청 DTO 파싱 & 검증
    try:
        req_dto = RecommendRequest.from_dict(body)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    user_id = req_dto.user_id
    top_k = req_dto.top_k

    # 2) Books 전체 카탈로그 불러오기
    catalog = fetch_catalog_books()
    if not catalog:
        resp_dto = RecommendResponse.empty(
            user_id=user_id,
            message="Books 컬렉션에 도서가 없습니다.",
        )
        return jsonify(resp_dto.to_dict()), 200

    catalog_index = build_catalog_index_by_isbn(catalog)

    # 3) 사용자 저장 isbn 목록 불러오기
    user_isbns = fetch_user_saved_isbns(user_id)
    if not user_isbns:
        resp_dto = RecommendResponse.empty(
            user_id=user_id,
            message="사용자가 저장한 책이 없습니다.",
        )
        return jsonify(resp_dto.to_dict()), 200

    # 4) isbn -> 실제 책 정보로 변환
    user_books = build_user_books_from_isbns(user_isbns, catalog_index)
    if not user_books:
        resp_dto = RecommendResponse.empty(
            user_id=user_id,
            message="저장된 isbn 중 Books 컬렉션에 존재하는 책이 없습니다.",
        )
        return jsonify(resp_dto.to_dict()), 200

    # 5) 토큰화 content 필드 추가
    attach_tokenized_content(user_books)
    attach_tokenized_content(catalog)

    # 6) 추천 계산
    try:
        raw_recommendations = compute_recommendations(
            user_books=user_books,
            catalog=catalog,
            top_k=top_k,
        )
    except Exception as e:
        return jsonify({
            "error": "추천 생성 중 오류가 발생했습니다.",
            "detail": str(e),
        }), 500

    # 7) 추천 결과 -> DTO 리스트로 변환
    items: List[BookRecommendationDTO] = []
    for r in raw_recommendations:
        # compute_recommendations가 반환하는 dict의 키 이름에 맞게
        # dto.BookRecommendationDTO.from_dict()만 잘 맞춰주면 됨
        items.append(BookRecommendationDTO.from_dict(r))

    resp_dto = RecommendResponse(
        user_id=user_id,
        count=len(items),
        items=items,
        message=None,
    )

    return jsonify(resp_dto.to_dict()), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
