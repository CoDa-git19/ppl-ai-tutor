"""Gradio web app: ask the tutor, or generate a practice quiz."""
from __future__ import annotations

import gradio as gr

from ppl_tutor.engine import Tutor, format_sources_md, render_quiz_md

try:
    TUTOR = Tutor()
    LOAD_ERROR = ""
except Exception as exc:  # show the problem in the UI instead of crashing
    TUTOR, LOAD_ERROR = None, str(exc)

EXAMPLES = [
    ["Phạm vi tĩnh (static scope) khác phạm vi động (dynamic scope) như thế nào?"],
    ["Explain how FIRST and FOLLOW sets are used in LL(1) parsing."],
    ["Closure là gì? Cho một ví dụ."],
    ["What is the difference between strong typing and static typing?"],
]


def on_ask(question: str, style: str, k: float):
    if LOAD_ERROR:
        return f"⚠️ {LOAD_ERROR}", ""
    if not question.strip():
        return "Type a question first.", ""
    try:
        answer = TUTOR.ask(question.strip(), k=int(k), style=style)
    except Exception as exc:
        return f"⚠️ {exc}", ""
    return answer.text, format_sources_md(answer.hits)


def on_quiz(topic: str, n: float, difficulty: str, language: str):
    if LOAD_ERROR:
        return f"⚠️ {LOAD_ERROR}"
    try:
        questions, hits = TUTOR.quiz(topic, n=int(n), difficulty=difficulty, language=language)
    except Exception as exc:
        return f"⚠️ {exc}"
    return render_quiz_md(questions, hits)


with gr.Blocks(title="PPL AI Tutor") as demo:
    gr.Markdown(
        "# PPL AI Tutor\n"
        "Retrieval-augmented tutor for **Principles of Programming Languages**. "
        "Answers are grounded in course notes with numbered citations; "
        "the quiz tab writes practice questions from the same notes."
    )
    with gr.Tab("Ask the tutor"):
        question = gr.Textbox(label="Your question (Vietnamese or English)", lines=2)
        with gr.Row():
            style = gr.Radio(["concise", "step-by-step", "beginner"], value="step-by-step",
                             label="Explanation style")
            k = gr.Slider(2, 8, value=5, step=1, label="Excerpts to retrieve")
        ask_btn = gr.Button("Ask", variant="primary")
        answer_md = gr.Markdown()
        sources_md = gr.Markdown()
        gr.Examples(EXAMPLES, inputs=[question])
        ask_btn.click(on_ask, [question, style, k], [answer_md, sources_md])
        question.submit(on_ask, [question, style, k], [answer_md, sources_md])

    with gr.Tab("Quiz generator"):
        topic = gr.Textbox(label="Topic (leave empty for a random mix)", placeholder="e.g. LL(1) parsing")
        with gr.Row():
            n_q = gr.Slider(3, 10, value=5, step=1, label="Number of questions")
            difficulty = gr.Radio(["easy", "medium", "hard"], value="medium", label="Difficulty")
            language = gr.Radio(["Vietnamese", "English"], value="Vietnamese", label="Language")
        quiz_btn = gr.Button("Generate quiz", variant="primary")
        quiz_md = gr.Markdown()
        quiz_btn.click(on_quiz, [topic, n_q, difficulty, language], [quiz_md])

if __name__ == "__main__":
    demo.launch()
