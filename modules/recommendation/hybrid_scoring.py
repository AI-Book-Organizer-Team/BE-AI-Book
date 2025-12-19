"""
하이브리드 추천 점수 계산 모듈
개발자 B 담당: log 스케일 가중치 적용 및 최종 점수 계산
"""
import math
from typing import Dict, List, Tuple, Optional

def calculate_log_weight(count: int) -> float:
    """
    대출/보유 횟수를 log 스케일로 변환
    log(1 + count)를 사용하여 극단값의 영향을 줄임
    
    Args:
        count: 대출 횟수 또는 앱 내 보유자 수
        
    Returns:
        log(1 + count) 값
        
    Example:
        >>> calculate_log_weight(0)    # 0.0
        >>> calculate_log_weight(10)   # 2.398
        >>> calculate_log_weight(100)  # 4.615
        >>> calculate_log_weight(1000) # 6.908
    """
    return math.log(1 + count)


def normalize_log_weight(log_weight: float, max_log_weight: float) -> float:
    """
    log 가중치를 0~1 범위로 정규화
    
    Args:
        log_weight: log(1 + count) 값
        max_log_weight: 데이터셋에서 가장 큰 log 값
        
    Returns:
        0~1 범위로 정규화된 값
    """
    if max_log_weight == 0:
        return 0
    return min(log_weight / max_log_weight, 1.0)


def calculate_hybrid_score(
    content_score: float,
    popularity_count: int,
    max_count: Optional[int] = None,
    content_weight: float = 0.7,
    popularity_weight: float = 0.3
) -> Dict[str, float]:
    """
    콘텐츠 기반 점수와 인기도를 결합한 하이브리드 점수 계산
    
    Args:
        content_score: 콘텐츠 기반 유사도 점수 (0~1)
        popularity_count: 대출 횟수 또는 보유자 수
        max_count: 전체 데이터셋의 최대 count (정규화용)
        content_weight: 콘텐츠 점수 가중치 (기본 70%)
        popularity_weight: 인기도 점수 가중치 (기본 30%)
        
    Returns:
        {
            'content_score': 콘텐츠 점수,
            'popularity_count': 원본 카운트,
            'log_weight': log 변환 값,
            'normalized_popularity': 정규화된 인기도 점수,
            'hybrid_score': 최종 하이브리드 점수
        }
    """
    # 가중치 합이 1인지 확인
    assert abs(content_weight + popularity_weight - 1.0) < 0.001, \
        "가중치 합은 1이어야 합니다"
    
    # log 스케일 변환
    log_weight = calculate_log_weight(popularity_count)
    
    # 정규화 (max_count가 주어진 경우)
    if max_count is not None:
        max_log = calculate_log_weight(max_count)
        normalized_popularity = normalize_log_weight(log_weight, max_log)
    else:
        # max_count가 없으면 임의로 정규화 (log(1001) ≈ 6.9 기준)
        normalized_popularity = min(log_weight / 6.9, 1.0)
    
    # 하이브리드 점수 계산
    hybrid_score = (content_weight * content_score + 
                   popularity_weight * normalized_popularity)
    
    return {
        'content_score': content_score,
        'popularity_count': popularity_count,
        'log_weight': log_weight,
        'normalized_popularity': normalized_popularity,
        'hybrid_score': hybrid_score
    }


def calculate_batch_hybrid_scores(
    books_data: Dict[str, Dict],
    content_weight: float = 0.7,
    popularity_weight: float = 0.3
) -> List[Dict]:
    """
    여러 책에 대한 하이브리드 점수를 일괄 계산
    
    Args:
        books_data: {
            '책ID': {
                'content_score': float,  # 콘텐츠 유사도
                'popularity_count': int   # 대출/보유 횟수
            }
        }
        content_weight: 콘텐츠 가중치
        popularity_weight: 인기도 가중치
        
    Returns:
        하이브리드 점수로 정렬된 책 정보 리스트
    """
    if not books_data:
        return []
    
    # 최대 count 찾기 (정규화용)
    max_count = max(book['popularity_count'] for book in books_data.values())
    
    results = []
    for book_id, data in books_data.items():
        scores = calculate_hybrid_score(
            content_score=data['content_score'],
            popularity_count=data['popularity_count'],
            max_count=max_count,
            content_weight=content_weight,
            popularity_weight=popularity_weight
        )
        scores['book_id'] = book_id
        results.append(scores)
    
    # 하이브리드 점수로 내림차순 정렬
    results.sort(key=lambda x: x['hybrid_score'], reverse=True)
    
    return results


def format_recommendation_response(
    hybrid_scores: List[Dict],
    books_metadata: Dict[str, Dict],
    top_k: int = 10
) -> List[Dict]:
    """
    API 응답 형식으로 추천 결과 포맷팅
    
    Args:
        hybrid_scores: calculate_batch_hybrid_scores의 결과
        books_metadata: {
            '책ID': {
                'title': str,
                'author': str,
                'description': str
            }
        }
        top_k: 반환할 추천 개수
        
    Returns:
        API 응답용 추천 리스트
    """
    recommendations = []
    
    for score_data in hybrid_scores[:top_k]:
        book_id = score_data['book_id']
        metadata = books_metadata.get(book_id, {})
        
        recommendation = {
            'id': book_id,
            'title': metadata.get('title', 'Unknown'),
            'author': metadata.get('author', 'Unknown'),
            'description': metadata.get('description', ''),
            'isbn': metadata.get('isbn'),
            'imageUrl': metadata.get('imageUrl'),
            'category': metadata.get('category'),
            # Android/DTO 호환 키
            'score': round(score_data['hybrid_score'], 3),
            'content_score': round(score_data['content_score'], 3),
            'popularity_score': round(score_data['normalized_popularity'], 3),
            'hybrid_score': round(score_data['hybrid_score'], 3),
            'reason': f"콘텐츠 유사도 {score_data['content_score']:.1%}, "
                     f"인기도 상위 {score_data['normalized_popularity']:.1%}"
        }
        recommendations.append(recommendation)
    
    return recommendations


# 데모/샘플 코드는 tests/hybrid_scoring_demo.py 참고
