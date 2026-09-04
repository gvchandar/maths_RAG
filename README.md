# Maths RAG Grader

Extracts a structured answer key from a Maths question-paper-and-answers PDF,
then grades student answers against it (retrieval + LLM grading) and scores them.

## Setup

```bash
cd maths-rag-grader
python3.11 -m venv .venv
./.venv/bin/pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
```

## 1. Build the answer key (RAG knowledge base) from the PDF

```bash
./.venv/bin/python extract_answer_key.py /path/to/paper.pdf -o answer_key.json
```

This reads the PDF and asks Claude to extract every question, its official
answer/solution, and its marks into `answer_key.json`. Do this once per exam.

## 2. Grade student answers

Answers are plain text, one per line, numbered to match the question paper:

```
1. x = 5
2) 42
Q3: the answer is 17
```

For multiple students in one file, separate blocks with `Student: <name>`.
Unnumbered lines are matched to the closest question via TF-IDF retrieval.

```bash
# from a file
./.venv/bin/python grade_answers.py --answer-key answer_key.json --file student_answers.txt

# pasted inline
./.venv/bin/python grade_answers.py --answer-key answer_key.json --text "1. 42
2. x=5"

# piped
cat student_answers.txt | ./.venv/bin/python grade_answers.py --answer-key answer_key.json
```

Prints a per-question, per-student score report to the console and saves full
detail (corrected solution + feedback per question) to `graded_report.json`.

## How the RAG part works

- `extract_answer_key.py` turns the PDF into structured `QAItem`s
  (question, official answer, marks) - the retrieval corpus.
- `retrieval.py` looks a student's answer up by question number first;
  if that fails, it falls back to TF-IDF similarity search over the question
  text to find the right question to grade against.
- `grade_answers.py` sends the retrieved question + official answer + the
  student's answer to Claude, which checks both the final answer and the
  method, awards partial credit, and returns a structured score.
