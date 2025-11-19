# recommendation/dto.py
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any


# =============================
# 1) 요청 DTO
# =============================

@dataclass
class RecommendRequest:
    """
    /api/recommend 요청 본문용 DTO
    """
    user_id: str
    top_k: int = 5

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RecommendRequest":
        """
        dict(JSON) -> DTO 변환 + 간단한 검증
        """
        if data is None:
            raise ValueError("요청 바디가 비어 있습니다.")

        user_id = data.get("user_id")
        if not user_id:
            raise ValueError("user_id 필드는 필수입니다.")

        top_k_raw = data.get("top_k", 5)
        try:
            top_k = int(top_k_raw)
        except Exception:
            top_k = 5

        if top_k <= 0:
            top_k = 5

        return cls(user_id=user_id, top_k=top_k)


# =============================
# 2) 개별 책 추천 DTO
# =============================

@dataclass
class BookRecommendationDTO:
    """
    한 권의 추천 결과 표현용 DTO
    """
    id: str
    title: str
    author: str
    description: str
    score: float

    # 선택 필드들
    isbn: Optional[str] = None
    imageUrl: Optional[str] = None
    category: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BookRecommendationDTO":
        """
        service.compute_recommendations() 결과(dict)를 DTO로 변환.
        이때 키 이름은 프로젝트에 맞추면 됨.
        """
        return cls(
            id=str(data.get("id", "")),
            title=data.get("title", "") or "",
            author=data.get("author", "") or "",
            description=data.get("description", "") or "",
            score=float(data.get("score", 0.0)),
            isbn=data.get("isbn"),
            imageUrl=data.get("imageUrl"),
            category=data.get("category"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        JSON 응답으로 내보낼 때 사용
        """
        return asdict(self)


# =============================
# 3) 추천 응답 DTO
# =============================

@dataclass
class RecommendResponse:
    """
    /api/recommend 응답 전체 DTO
    """
    user_id: str
    count: int
    items: List[BookRecommendationDTO]
    message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        JSON 응답용 dict 변환
        """
        return {
            "user_id": self.user_id,
            "count": self.count,
            "items": [item.to_dict() for item in self.items],
            "message": self.message,
        }

    @classmethod
    def empty(cls, user_id: str, message: str) -> "RecommendResponse":
        """
        추천이 없을 때 간단히 만들기 위한 헬퍼
        """
        return cls(user_id=user_id, count=0, items=[], message=message)
