"""Grade a student's (or several students') submitted Maths answers against
the RAG answer key produced by extract_answer_key.py, and compute a score.

Input format (plain text, one answer per line - keep each answer on a single
line, don't wrap it across multiple lines):
    1. x = 5
    2) 42
    Q3: the answer is 17

Multiple students in one file: separate blocks with a "Student: <name>" line.
Lines with no leading question number are matched to the best remaining
question via TF-IDF retrieval instead of an exact number lookup.

Usage:
    python grade_answers.py --answer-key answer_key.json --file student_answers.txt
    python grade_answers.py --answer-key answer_key.json --text "1. 42\n2. x=5"
    cat student_answers.txt | python grade_answers.py --answer-key answer_key.json
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field

import anthropic

from models import AnswerKey, GradeResult, QAItem
from retrieval import AnswerKeyIndex

MODEL = "claude-opus-5"

STUDENT_HEADER_RE = re.compile(r"^\s*student\s*[:\-]?\s*(.+)$", re.IGNORECASE)
QNO_LINE_RE = re.compile(r"^\s*(?:q\.?\s*)?(\d+[a-zA-Z]?)\s*[\.\):]\s*(.*)$", re.IGNORECASE)

GRADING_INSTRUCTIONS = """\
You are grading a student's answer to a Maths exam question.

Question: {question_text}
Official correct answer / solution: {correct_answer}
Maximum marks for this question: {max_marks}

Student's answer: {student_answer}

Compare the student's work to the official answer. Check the final answer AND
the method/working. Award partial credit when the method is correct but there
is an arithmetic slip, and award full marks only when the final answer and
reasoning are both correct. Explain what is wrong (if anything) and show the
corrected solution.
"""


@dataclass
class ParsedAnswer:
    student_id: str
    question_no: str | None
    answer_text: str


def parse_student_answers(text: str) -> list[ParsedAnswer]:
    parsed: list[ParsedAnswer] = []
    current_student = "default"

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        header_match = STUDENT_HEADER_RE.match(line)
        if header_match:
            current_student = header_match.group(1).strip()
            continue

        qno_match = QNO_LINE_RE.match(line)
        if qno_match:
            parsed.append(ParsedAnswer(current_student, qno_match.group(1), qno_match.group(2).strip()))
        else:
            parsed.append(ParsedAnswer(current_student, None, line))

    return parsed


def resolve_question(entry: ParsedAnswer, index: AnswerKeyIndex) -> tuple[QAItem, bool]:
    """Returns (matched QAItem, was_exact_number_match)."""
    if entry.question_no is not None:
        item = index.lookup_by_number(entry.question_no)
        if item is not None:
            return item, True
    # RAG fallback: retrieve the best-matching question by content similarity.
    return index.retrieve(entry.answer_text, top_k=1)[0], False


def grade_one(client: anthropic.Anthropic, item: QAItem, entry: ParsedAnswer, matched_exact: bool, model: str) -> GradeResult:
    prompt = GRADING_INSTRUCTIONS.format(
        question_text=item.question_text,
        correct_answer=item.correct_answer,
        max_marks=item.max_marks,
        student_answer=entry.answer_text,
    )
    response = client.messages.parse(
        model=model,
        max_tokens=4096,
        thinking={"type": "adaptive"},
        output_config={"effort": "high"},
        messages=[{"role": "user", "content": prompt}],
        output_format=GradeResult,
    )
    result = response.parsed_output
    result.question_no = entry.question_no or item.question_no
    result.matched_question_no = item.question_no
    result.student_answer = entry.answer_text
    result.max_marks = item.max_marks
    return result


def grade_submission(answer_key: AnswerKey, student_text: str, model: str = MODEL) -> dict[str, list[GradeResult]]:
    index = AnswerKeyIndex(answer_key.items)
    entries = parse_student_answers(student_text)
    client = anthropic.Anthropic()

    results_by_student: dict[str, list[GradeResult]] = {}
    for entry in entries:
        item, matched_exact = resolve_question(entry, index)
        result = grade_one(client, item, entry, matched_exact, model)
        results_by_student.setdefault(entry.student_id, []).append(result)
    return results_by_student


def print_report(results_by_student: dict[str, list[GradeResult]]):
    for student_id, results in results_by_student.items():
        print(f"\n=== {student_id} ===")
        total_score = 0.0
        total_max = 0.0
        for r in sorted(results, key=lambda r: r.question_no):
            status = "correct" if r.is_correct else "incorrect"
            print(f"  Q{r.question_no} [{status}] {r.score}/{r.max_marks}")
            if not r.is_correct:
                print(f"    feedback: {r.feedback}")
            total_score += r.score
            total_max += r.max_marks
        pct = (100 * total_score / total_max) if total_max else 0.0
        print(f"  --- Total: {total_score}/{total_max} ({pct:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--answer-key", required=True, help="Path to answer_key.json from extract_answer_key.py")
    parser.add_argument("--file", help="Path to a text file with student answers")
    parser.add_argument("--text", help="Student answers passed inline as a string")
    parser.add_argument("-o", "--output", default="graded_report.json")
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()

    with open(args.answer_key) as f:
        answer_key = AnswerKey.model_validate(json.load(f))

    if args.text is not None:
        student_text = args.text
    elif args.file is not None:
        with open(args.file) as f:
            student_text = f.read()
    else:
        student_text = sys.stdin.read()

    results_by_student = grade_submission(answer_key, student_text, model=args.model)
    print_report(results_by_student)

    serializable = {
        student_id: [r.model_dump() for r in results]
        for student_id, results in results_by_student.items()
    }
    with open(args.output, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSaved detailed report -> {args.output}")


if __name__ == "__main__":
    main()
