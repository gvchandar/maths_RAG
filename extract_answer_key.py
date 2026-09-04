"""Extract a structured answer key (question + official answer + marks) from a
Maths question-paper-and-answers PDF. This structured key is the RAG knowledge
base that grade_answers.py retrieves from and grades against.

Usage:
    python extract_answer_key.py path/to/paper.pdf -o answer_key.json
"""

import argparse
import base64
import json
import sys

import anthropic

from models import AnswerKey

MODEL = "claude-opus-5"

EXTRACTION_INSTRUCTIONS = """\
This PDF contains a Maths question paper together with its official answers
(and/or worked solutions). Extract every question into a structured list.

For each question (treat each sub-part, e.g. "3(a)"/"3(b)", as its own item):
- question_no: the question/part label exactly as printed (e.g. "1", "2a", "3(b)")
- question_text: the full question text
- correct_answer: the official final answer; include key working/solution steps
  if the paper shows them, since they are useful for grading later
- max_marks: the marks allocated to this question if stated in the paper,
  otherwise default to 1

Do not skip any question. Do not invent answers that aren't in the document -
if the paper only gives a final answer with no working, just extract that.
"""


def extract_answer_key(pdf_path: str, model: str = MODEL) -> AnswerKey:
    with open(pdf_path, "rb") as f:
        pdf_b64 = base64.standard_b64encode(f.read()).decode("utf-8")

    client = anthropic.Anthropic()
    response = client.messages.parse(
        model=model,
        max_tokens=16000,
        thinking={"type": "adaptive"},
        output_config={"effort": "high"},
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": pdf_b64,
                        },
                    },
                    {"type": "text", "text": EXTRACTION_INSTRUCTIONS},
                ],
            }
        ],
        output_format=AnswerKey,
    )
    return response.parsed_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf_path", help="Path to the Maths question paper + answers PDF")
    parser.add_argument("-o", "--output", default="answer_key.json", help="Where to save the extracted answer key JSON")
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()

    try:
        answer_key = extract_answer_key(args.pdf_path, model=args.model)
    except FileNotFoundError:
        print(f"PDF not found: {args.pdf_path}", file=sys.stderr)
        sys.exit(1)
    except anthropic.APIStatusError as e:
        print(f"Anthropic API error: {e}", file=sys.stderr)
        sys.exit(1)

    with open(args.output, "w") as f:
        json.dump(answer_key.model_dump(), f, indent=2)

    print(f"Extracted {len(answer_key.items)} questions -> {args.output}")
    for item in answer_key.items:
        print(f"  Q{item.question_no} ({item.max_marks} marks): {item.question_text[:70]}")


if __name__ == "__main__":
    main()
