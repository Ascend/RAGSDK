# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Education math RAG best-practice demo.

This script keeps the demo runnable without external model services. It mirrors
the key RAG steps used in production: document normalization, chunking,
retrieval, and prompt assembly. In production, replace the local lexical
retriever with RAGSDK embedding/reranker services.
"""

import argparse
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


FORMULA_REPLACEMENTS = {
    "（": "(",
    "）": ")",
    "，": ",",
    "。": ".",
    "＋": "+",
    "－": "-",
    "×": "*",
    "÷": "/",
    "＝": "=",
    "≤": "<=",
    "≥": ">=",
}

DEFAULT_QUESTIONS = [
    "一元二次方程 x^2 - 5x + 6 = 0 怎么因式分解？",
    "三角形相似判定需要哪些条件？",
    "数学知识库里 chunk size 应该怎么选？",
]

SELF_CHECK_CASES = {
    "一元二次方程 x^2 - 5x + 6 = 0 怎么因式分解？": "一元二次方程",
    "三角形相似判定需要哪些条件？": "三角形相似",
    "数学知识库里 chunk size 应该怎么选？": "数学知识库构建建议",
}


@dataclass
class KnowledgeChunk:
    """A retrieved unit with enough metadata to build a RAG prompt."""

    chunk_id: str
    title: str
    content: str


def normalize_math_text(text: str) -> str:
    """Normalize common OCR and formula variants before chunking."""

    normalized = text
    for source, target in FORMULA_REPLACEMENTS.items():
        normalized = normalized.replace(source, target)
    normalized = re.sub(r"\$\s*(.*?)\s*\$", r"\1", normalized)
    normalized = re.sub(r"\\\((.*?)\\\)", r"\1", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def split_by_markdown_heading(text: str) -> List[tuple[str, str]]:
    """Split markdown into titled sections while preserving heading context."""

    sections: List[tuple[str, List[str]]] = []
    current_title = "课程资料"
    current_lines: List[str] = []

    for line in text.splitlines():
        heading = re.match(r"^(#{1,6})\s+(.+)$", line.strip())
        if heading:
            if current_lines:
                sections.append((current_title, current_lines))
            current_title = heading.group(2).strip()
            current_lines = []
            continue
        if line.strip():
            current_lines.append(line.strip())

    if current_lines:
        sections.append((current_title, current_lines))

    return [(title, "\n".join(lines)) for title, lines in sections]


def split_text_with_overlap(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Split text into character chunks with overlap for recall stability."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    text = normalize_math_text(text)
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = end - chunk_overlap
    return [chunk for chunk in chunks if chunk]


def load_knowledge_chunks(path: Path, chunk_size: int, chunk_overlap: int) -> List[KnowledgeChunk]:
    """Load a markdown knowledge file and split it into retrievable chunks."""

    text = path.read_text(encoding="utf-8")
    chunks: List[KnowledgeChunk] = []
    for section_index, (title, section_text) in enumerate(split_by_markdown_heading(text), start=1):
        section_chunks = split_text_with_overlap(section_text, chunk_size, chunk_overlap)
        for chunk_index, content in enumerate(section_chunks, start=1):
            chunks.append(
                KnowledgeChunk(
                    chunk_id=f"s{section_index:02d}-c{chunk_index:02d}",
                    title=title,
                    content=content,
                )
            )
    return chunks


def tokenize(text: str) -> List[str]:
    """Tokenize Chinese text, English words, numbers, and common formula signs."""

    text = normalize_math_text(text.lower())
    tokens = re.findall(r"[\u4e00-\u9fff]|[a-z]+|\d+(?:\.\d+)?|[+\-*/=^()<>]+", text)
    return [token for token in tokens if token.strip()]


class LocalTfidfRetriever:
    """Small local retriever used only to make the best-practice demo executable."""

    def __init__(self, chunks: Sequence[KnowledgeChunk]):
        self.chunks = list(chunks)
        self.term_freqs = [Counter(tokenize(chunk.title + " " + chunk.content)) for chunk in self.chunks]
        document_count = len(self.term_freqs)
        document_freq = Counter()
        for term_freq in self.term_freqs:
            document_freq.update(term_freq.keys())
        self.idf = {token: math.log((document_count + 1) / (freq + 1)) + 1.0 for token, freq in document_freq.items()}

    def _vector(self, tokens: Iterable[str]) -> Counter:
        counts = Counter(tokens)
        return Counter({token: count * self.idf.get(token, 1.0) for token, count in counts.items()})

    @staticmethod
    def _cosine(left: Counter, right: Counter) -> float:
        shared = set(left) & set(right)
        numerator = sum(left[token] * right[token] for token in shared)
        left_norm = math.sqrt(sum(value * value for value in left.values()))
        right_norm = math.sqrt(sum(value * value for value in right.values()))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return numerator / (left_norm * right_norm)

    def retrieve(self, query: str, top_k: int) -> List[tuple[float, KnowledgeChunk]]:
        query_vector = self._vector(tokenize(query))
        scored = []
        for term_freq, chunk in zip(self.term_freqs, self.chunks):
            chunk_vector = self._vector(term_freq.elements())
            scored.append((self._cosine(query_vector, chunk_vector), chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[:top_k]


def build_prompt(question: str, retrieved_chunks: Sequence[tuple[float, KnowledgeChunk]]) -> str:
    """Build a production-style RAG prompt from retrieved context."""

    context_blocks = []
    for rank, (score, chunk) in enumerate(retrieved_chunks, start=1):
        context_blocks.append(f"[{rank}] {chunk.title} ({chunk.chunk_id}, score={score:.3f})\n{chunk.content}")
    context = "\n\n".join(context_blocks)
    return (
        "你是教培行业的数学知识问答助手。请只依据给定资料回答，"
        "公式保持规范写法，必要时给出步骤。\n\n"
        f"资料：\n{context}\n\n"
        f"问题：{question}\n"
        "回答："
    )


def run_demo(args: argparse.Namespace) -> int:
    chunks = load_knowledge_chunks(Path(args.knowledge_file), args.chunk_size, args.chunk_overlap)
    retriever = LocalTfidfRetriever(chunks)
    questions = list(SELF_CHECK_CASES) if args.self_check else ([args.question] if args.question else DEFAULT_QUESTIONS)

    print(f"Loaded {len(chunks)} chunks from {args.knowledge_file}")
    self_check_passed = True
    for question in questions:
        print("\n" + "=" * 80)
        print(f"Question: {question}")
        retrieved = retriever.retrieve(question, args.top_k)
        for rank, (score, chunk) in enumerate(retrieved, start=1):
            print(f"Top {rank}: {chunk.title} / {chunk.chunk_id} / score={score:.3f}")
            print(f"  {chunk.content[:120]}")
        if args.show_prompt:
            print("\nPrompt:")
            print(build_prompt(question, retrieved))
        if args.self_check:
            expected_title = SELF_CHECK_CASES[question]
            top_title = retrieved[0][1].title if retrieved else ""
            passed = top_title == expected_title
            self_check_passed = self_check_passed and passed
            print(f"Self-check: expected top title '{expected_title}', got '{top_title}' -> {passed}")

    return 0 if self_check_passed else 1


def parse_args() -> argparse.Namespace:
    current_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run the education math RAG best-practice demo.")
    parser.add_argument(
        "--knowledge_file",
        default=str(current_dir / "data" / "math_lesson.md"),
        help="Path to the markdown knowledge file.",
    )
    parser.add_argument("--question", default="", help="Question to retrieve. Empty means running built-in examples.")
    parser.add_argument("--top_k", type=int, default=2, help="Number of chunks to retrieve.")
    parser.add_argument("--chunk_size", type=int, default=260, help="Chunk size for math course documents.")
    parser.add_argument("--chunk_overlap", type=int, default=40, help="Chunk overlap for preserving formula context.")
    parser.add_argument("--show_prompt", action="store_true", help="Print the assembled RAG prompt.")
    parser.add_argument(
        "--self_check",
        action="store_true",
        help="Run built-in retrieval checks and return non-zero on failure.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run_demo(parse_args()))
