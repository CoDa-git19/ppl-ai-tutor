"""Load course notes (.tex, .md, .txt, .pdf) into titled sections."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

SUPPORTED = {".tex", ".md", ".txt", ".pdf"}

HEADING_RE = re.compile(r"\\(chapter|section|subsection|subsubsection)\*?\s*(\[[^\]]*\])?\s*\{")
CODE_ENV_RE = re.compile(
    r"\\begin\{(lstlisting|verbatim|minted|Verbatim)\}(\{[^}]*\}|\[[^\]]*\])*(.*?)\\end\{\1\}",
    re.DOTALL,
)
INLINE_CODE_RE = re.compile(r"\\(?:verb|lstinline)\*?(?:\[[^\]]*\])?([^\w\s{])(.*?)\1")
COMMENT_RE = re.compile(r"(?<!\\)%.*")
LEVEL = {"chapter": 0, "section": 1, "subsection": 2, "subsubsection": 3}


@dataclass
class Section:
    source: str
    title: str
    text: str
    page: int | None = None


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _balanced(text: str, start: int) -> tuple[str, int]:
    """Return the content of a {...} group whose opening brace is at text[start-1]."""
    depth, i = 1, start
    while i < len(text) and depth:
        if text[i] == "{" and text[i - 1] != "\\":
            depth += 1
        elif text[i] == "}" and text[i - 1] != "\\":
            depth -= 1
        i += 1
    return text[start : i - 1], i


def _latex_to_text(latex: str) -> str:
    try:
        from pylatexenc.latex2text import LatexNodes2Text

        return LatexNodes2Text(math_mode="verbatim").latex_to_text(latex)
    except Exception:  # fall back to a crude strip if pylatexenc chokes
        text = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", latex)
        return re.sub(r"[{}]", "", text)


def load_tex(path: Path) -> list[Section]:
    raw = _read(path)
    m = re.search(r"\\begin\{document\}(.*?)\\end\{document\}", raw, re.DOTALL)
    body = m.group(1) if m else raw

    code_blocks: list[str] = []

    def protect(match: re.Match) -> str:
        code_blocks.append(match.group(match.lastindex).strip("\n"))
        return f"\n\nXXCODEBLOCK{len(code_blocks) - 1}XX\n\n"

    inline_code: list[str] = []

    def protect_inline(match: re.Match) -> str:
        inline_code.append(match.group(2))
        return f"XXINLINE{len(inline_code) - 1}XX"

    body = CODE_ENV_RE.sub(protect, body)
    body = INLINE_CODE_RE.sub(protect_inline, body)
    body = COMMENT_RE.sub("", body)

    parts: list[tuple[str, str]] = []
    stack: list[tuple[int, str]] = []
    pos, current = 0, "Introduction"
    for hm in HEADING_RE.finditer(body):
        if hm.start() < pos:
            continue
        parts.append((current, body[pos : hm.start()]))
        title_tex, end = _balanced(body, hm.end())
        title = " ".join(_latex_to_text(title_tex).split()) or "Untitled"
        level = LEVEL[hm.group(1)]
        stack = [s for s in stack if s[0] < level] + [(level, title)]
        current = " › ".join(t for _, t in stack)
        pos = end
    parts.append((current, body[pos:]))

    sections = []
    for title, chunk in parts:
        text = _latex_to_text(chunk)
        for i, code in enumerate(code_blocks):
            text = text.replace(f"XXCODEBLOCK{i}XX", f"\n```\n{code}\n```\n")
        for i, code in enumerate(inline_code):
            text = text.replace(f"XXINLINE{i}XX", f"`{code}`")
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if len(text) >= 20:
            sections.append(Section(path.name, title, text))
    return sections


def load_markdown(path: Path) -> list[Section]:
    sections, title, buf = [], "Introduction", []
    stack: list[tuple[int, str]] = []
    in_code = False
    for line in _read(path).splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
        hm = None if in_code else re.match(r"^(#{1,4})\s+(.*)", line)
        if hm:
            if "".join(buf).strip():
                sections.append(Section(path.name, title, "\n".join(buf).strip()))
            level = len(hm.group(1))
            stack = [s for s in stack if s[0] < level] + [(level, hm.group(2).strip())]
            title, buf = " › ".join(t for _, t in stack), []
        else:
            buf.append(line)
    if "".join(buf).strip():
        sections.append(Section(path.name, title, "\n".join(buf).strip()))
    return sections


def load_txt(path: Path) -> list[Section]:
    text = _read(path).strip()
    return [Section(path.name, path.stem, text)] if text else []


def load_pdf(path: Path) -> list[Section]:
    from pypdf import PdfReader

    sections = []
    for i, page in enumerate(PdfReader(str(path)).pages, start=1):
        text = (page.extract_text() or "").strip()
        if len(text) >= 40:
            sections.append(Section(path.name, f"Page {i}", text, page=i))
    return sections


LOADERS = {".tex": load_tex, ".md": load_markdown, ".txt": load_txt, ".pdf": load_pdf}


def load_folder(folder: Path) -> list[Section]:
    sections: list[Section] = []
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            try:
                found = LOADERS[path.suffix.lower()](path)
            except Exception as exc:  # one bad file must not stop the whole build
                print(f"  ! skipped {path.name}: {exc}")
                continue
            print(f"  - {path.name}: {len(found)} sections")
            sections.extend(found)
    return sections
