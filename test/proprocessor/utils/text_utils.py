# utils/text_utils.py
"""Text utility functions used across preprocessing stages."""
import re
from difflib import SequenceMatcher
from typing import List

QUOTE_PATTERN = re.compile(r"[“\"]([^”\"]+)[”\"]")
ROLE_PATTERN = re.compile(
    r"^\s*(催收员|坐席|客户|债务人|agent|customer)\s*[：:]\s*(.*)$"
)

AGENT_NAMES = {"催收员", "坐席", "agent"}
CUSTOMER_NAMES = {"客户", "债务人", "customer"}


def extract_quoted_spans(text: str) -> List[str]:
    """Extract all quoted substrings using Chinese and English quotes."""
    return QUOTE_PATTERN.findall(text or "")


def fuzzy_contains(evidence: str, content: str, threshold: float = 0.6) -> bool:
    """Return True if `evidence` appears in `content`, allowing fuzzy match."""
    if not evidence or not content:
        return False
    if evidence in content:
        return True
    if len(evidence) < 4:
        return False
    if len(content) < len(evidence):
        return SequenceMatcher(None, evidence, content).ratio() > threshold
    for i in range(len(content) - len(evidence) + 1):
        window = content[i : i + len(evidence)]
        if SequenceMatcher(None, evidence, window).ratio() > threshold:
            return True
    return False