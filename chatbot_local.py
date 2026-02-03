#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from textwrap import fill

def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def split_paragraphs(text: str) -> list[str]:
    blocks = [block.strip() for block in re.split(r"\n\s*\n", text)]
    return [block for block in blocks if block]

def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[\wʻ’'-]+", text.lower()))

def score_block(block: str, query_terms: set[str]) -> int:
    if not query_terms:
        return 0
    block_terms = tokenize(block)
    return len(block_terms & query_terms)

def rank_blocks(blocks: list[str], query: str, top_k: int = 3) -> list[str]:
    query_terms = tokenize(query)
    scored = [
        (score_block(block, query_terms), block)
        for block in blocks
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    return [block for score, block in scored[:top_k] if score > 0]

def format_answer(blocks: list[str]) -> str:
    if not blocks:
        return (
            "Ничего релевантного не найдено в локальной базе. "
            "Попробуйте переформулировать вопрос или уточнить тему."
        )
    parts = []
    for idx, block in enumerate(blocks, start=1):
        parts.append(f"Фрагмент {idx}:\n{block}")
    return "\n\n".join(parts)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Локальный чатбот по базе знаний (прототип)."
    )
    parser.add_argument(
        "--source",
        default="regulation_uz.txt",
        help="Путь к текстовому файлу базы знаний",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=3,
        help="Сколько фрагментов возвращать",
    )
    parser.add_argument(
        "--wrap",
        type=int,
        default=120,
        help="Ширина переноса строк для вывода",
    )
    args = parser.parse_args()

    source_path = Path(args.source)
    if not source_path.exists():
        raise SystemExit(f"Файл не найден: {source_path}")

    text = load_text(source_path)
    blocks = split_paragraphs(text)

    print("Локальный чатбот готов. Введите вопрос (пустая строка — выход).")
    while True:
        try:
            query = input("\nВопрос: ").strip()
        except EOFError:
            print("\nВыход.")
            break
        if not query:
            print("Выход.")
            break
        results = rank_blocks(blocks, query, top_k=args.top)
        answer = format_answer(results)
        print("\nОтвет (черновик):")
        print(fill(answer, width=args.wrap, replace_whitespace=False))

if __name__ == "__main__":
    main()
