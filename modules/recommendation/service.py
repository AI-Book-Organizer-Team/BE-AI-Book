from typing import List, Dict, Optional
import numpy as np
from .embedding import embed_text
from .similarity import cosine
from .hybrid_scoring import calculate_batch_hybrid_scores, format_recommendation_response

def compute_recommendations(
    user_books: List[Dict],           # [{'id','title','author','description'}...]
    catalog: List[Dict],              # 전체 도서(동일 키들 + 'popularity_count' 포함 권장)
    user_profile: Optional[Dict] = None,
    top_k: int = 5,
    content_weight: float = 0.7,
    popularity_weight: float = 0.3,
) -> List[Dict]:
    # 1) 사용자 평균 임베딩
    user_vecs = []
    for b in user_books:
        text = f"{b['title']} by {b.get('author','')}: {b.get('description','')}"
        user_vecs.append(embed_text(text))
    if not user_vecs:
        return []
    avg_vec = np.mean(user_vecs, axis=0)

    # 2) 카탈로그 유사도
    content_scores = {}
    for book in catalog:
        if any(book['id'] == ub['id'] for ub in user_books):
            continue
        text = f"{book['title']} by {book.get('author','')}: {book.get('description','')}"
        vec = embed_text(text)
        sim = cosine(avg_vec, vec)
        content_scores[book['id']] = (sim + 1) / 2  # [-1,1] → [0,1]

    # 3) 하이브리드 입력
    books_for_scoring = {}
    for book in catalog:
        bid = book['id']
        if bid in content_scores:
            books_for_scoring[bid] = {
                'content_score': content_scores[bid],
                'popularity_count': int(book.get('popularity_count', 0)),
            }

    # 4) 하이브리드 계산
    hybrid = calculate_batch_hybrid_scores(
        books_for_scoring,
        content_weight=content_weight,
        popularity_weight=popularity_weight,
    )

    # 5) 응답 포맷
    metadata = {
        b['id']: {
            'title': b['title'],
            'author': b.get('author', ''),
            'description': b.get('description', ''),
        } for b in catalog
    }
    return format_recommendation_response(hybrid, metadata, top_k=top_k)
