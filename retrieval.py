"""TF-IDF retrieval over the extracted answer-key, used when a student's
answer can't be matched to a question by number (missing/garbled numbering)."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from models import QAItem


def normalize_qno(qno: str) -> str:
    return "".join(ch for ch in qno.strip().lower() if ch.isalnum())


class AnswerKeyIndex:
    def __init__(self, items: list[QAItem]):
        self.items = items
        self.by_qno = {normalize_qno(it.question_no): it for it in items}
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = self._vectorizer.fit_transform(it.question_text for it in items)

    def lookup_by_number(self, qno: str) -> QAItem | None:
        return self.by_qno.get(normalize_qno(qno))

    def retrieve(self, query_text: str, top_k: int = 1) -> list[QAItem]:
        """Semantic-ish fallback retrieval by TF-IDF similarity against question text."""
        query_vec = self._vectorizer.transform([query_text])
        sims = cosine_similarity(query_vec, self._matrix)[0]
        ranked = sorted(range(len(self.items)), key=lambda i: sims[i], reverse=True)
        return [self.items[i] for i in ranked[:top_k]]
