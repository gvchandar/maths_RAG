"""Shared data models for the maths answer-key RAG grader."""

from pydantic import BaseModel


class QAItem(BaseModel):
    question_no: str
    question_text: str
    correct_answer: str
    max_marks: float = 1.0


class AnswerKey(BaseModel):
    items: list[QAItem]


class GradeResult(BaseModel):
    question_no: str
    matched_question_no: str
    student_answer: str
    is_correct: bool
    score: float
    max_marks: float
    corrected_solution: str
    feedback: str
