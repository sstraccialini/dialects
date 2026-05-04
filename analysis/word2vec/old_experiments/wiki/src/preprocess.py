import re
from typing import List

TOKEN_PATTERN = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)?", re.UNICODE)


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    text = normalize_text(text)
    return TOKEN_PATTERN.findall(text)
