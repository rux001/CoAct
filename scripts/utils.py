"""
Shared utilities for the Selective DPO pipeline.

Provides:
- Cached regex compilation (get_compiled_regex)
- Dataset-specific answer extraction for GSM8K, MATH, WebInstruct
- Dispatch by dataset prefix (extract_answer)
- Question extraction from prompts (extract_question_from_prompt)
- Task instruction strings for preference-pair construction (get_task_instruction)
"""

import re
from typing import Optional

__all__ = [
    "get_compiled_regex",
    "normalize_math_answer",
    "extract_gsm8k_answer",
    "extract_math_answer",
    "extract_webinstruct_answer",
    "extract_answer",
    "extract_question_from_prompt",
    "get_task_instruction",
    "SUPPORTED_DATASETS",
]

SUPPORTED_DATASETS = ("gsm8k", "math", "webinstruct")

# ---------------------------------------------------------------------------
# Regex caching
# ---------------------------------------------------------------------------

_REGEX_CACHE: dict = {}


def get_compiled_regex(pattern: str, flags: int = 0) -> "re.Pattern[str]":
    """Compile and memoize a regex pattern."""
    key = (pattern, flags)
    compiled = _REGEX_CACHE.get(key)
    if compiled is None:
        compiled = re.compile(pattern, flags)
        _REGEX_CACHE[key] = compiled
    return compiled


# ---------------------------------------------------------------------------
# MATH answer extraction
# ---------------------------------------------------------------------------

def normalize_math_answer(answer: Optional[str]) -> Optional[str]:
    if not answer:
        return None
    answer = answer.strip().strip("$").strip()
    answer = get_compiled_regex(r"\\boxed\{([^}]*)\}").sub(r"\1", answer)
    answer = get_compiled_regex(r"\s+").sub(" ", answer)
    return answer.strip()


def _looks_like_math_expression(candidate: Optional[str]) -> bool:
    if not candidate:
        return False
    candidate = candidate.strip()
    if not candidate:
        return False
    if get_compiled_regex(r"[0-9]").search(candidate):
        return True
    if get_compiled_regex(r"\\(frac|sqrt|pi|times|cdot|pm|overline|underline|sum|prod|int)").search(candidate):
        return True
    if get_compiled_regex(r"[=+\-*/^]").search(candidate):
        return True
    return False


def _extract_last_numeric_token(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    matches = get_compiled_regex(
        r"(?:\\frac\{[^}]+\}\{[^}]+\}|\\sqrt\{[^}]+\}|[+-]?\d+(?:/\d+)?(?:\.\d+)?)"
    ).findall(text)
    if matches:
        return normalize_math_answer(matches[-1])
    return None


def extract_math_answer(text: str) -> Optional[str]:
    """Extract the final answer from a MATH response (LaTeX-boxed expression)."""
    if not text:
        return None

    # Primary: extract balanced \boxed{...}
    boxed_match = get_compiled_regex(r"\\boxed\{").search(text)
    if boxed_match:
        start_pos = boxed_match.end()
        brace_count = 1
        pos = start_pos
        while pos < len(text) and brace_count > 0:
            if text[pos] == "{":
                brace_count += 1
            elif text[pos] == "}":
                brace_count -= 1
            pos += 1
        if brace_count == 0:
            candidate = normalize_math_answer(text[start_pos:pos - 1])
            if _looks_like_math_expression(candidate):
                return candidate
            numeric_fallback = _extract_last_numeric_token(candidate)
            if numeric_fallback:
                return numeric_fallback

    # Secondary: simple boxed patterns without balanced braces
    simple_boxed = get_compiled_regex(r"\\boxed\s*\{([^}]*)").findall(text)
    if simple_boxed:
        candidate = normalize_math_answer(simple_boxed[-1])
        if _looks_like_math_expression(candidate):
            return candidate
        numeric_fallback = _extract_last_numeric_token(candidate)
        if numeric_fallback:
            return numeric_fallback

    # Tertiary: explicit answer statements
    match = get_compiled_regex(
        r"(?:final\s+answer|answer|the\s+answer\s+is)\s*[:\-]?\s*(.+)",
        re.IGNORECASE,
    ).search(text)
    if match:
        candidate = match.group(1).strip().split("\n", 1)[0]
        normalized = normalize_math_answer(candidate)
        if _looks_like_math_expression(normalized):
            return normalized
        numeric_fallback = _extract_last_numeric_token(candidate)
        if numeric_fallback:
            return numeric_fallback

    # Fallback: capture the last math token/number
    return _extract_last_numeric_token(text)


# ---------------------------------------------------------------------------
# GSM8K answer extraction
# ---------------------------------------------------------------------------

def extract_gsm8k_answer(text: str) -> Optional[str]:
    """Extract the final numeric answer from a GSM8K response."""
    if not text:
        return None

    first_final_answer_match = get_compiled_regex(
        r"final answer:\s*([^\n]*)", re.IGNORECASE
    ).search(text)
    if first_final_answer_match:
        answer_text = first_final_answer_match.group(1).strip()
        numbers = get_compiled_regex(r"([+-]?\d+(?:\.\d+)?)").findall(answer_text)
        if numbers:
            try:
                return str(float(numbers[0]))
            except ValueError:
                pass

    patterns = [
        r"answer\s*:\s*([+-]?\d+(?:\.\d+)?)",
        r"the answer is\s*([+-]?\d+(?:\.\d+)?)",
        r"final answer is\s*([+-]?\d+(?:\.\d+)?)",
        r"=\s*([+-]?\d+(?:\.\d+)?)",
    ]

    text_lower = text.lower().strip()
    for pattern in patterns:
        matches = get_compiled_regex(pattern, re.IGNORECASE | re.DOTALL).findall(text_lower)
        if matches:
            answer = get_compiled_regex(r"[^\d+-.]").sub("", matches[0].strip())
            if answer:
                try:
                    return str(float(answer))
                except ValueError:
                    continue

    numbers = get_compiled_regex(r"(\d+(?:\.\d+)?)").findall(text)
    for num_str in reversed(numbers):
        try:
            num = float(num_str)
            if num >= 1:
                return str(num)
        except ValueError:
            continue

    return None


# ---------------------------------------------------------------------------
# WebInstruct answer extraction
# ---------------------------------------------------------------------------

def _extract_webinstruct_value(text: str) -> Optional[str]:
    """Extract a numeric value from WebInstruct text, handling scientific notation,
    currency, comma-separated thousands, and plain decimals."""
    if not text:
        return None
    text = text.strip()

    # Scientific notation like "1.4 x 10 ^ 9"
    sci_match = get_compiled_regex(
        r"([+-]?\d+\.?\d*)\s*[x×*]\s*10\s*[\^**]\s*([+-]?\d+)", re.IGNORECASE
    ).search(text)
    if sci_match:
        try:
            coefficient = float(sci_match.group(1))
            exponent = int(sci_match.group(2))
            return str(coefficient * (10.0 ** exponent))
        except (OverflowError, ValueError):
            pass

    # Standard scientific notation like "1.4e9"
    sci_e_match = get_compiled_regex(r"([+-]?\d+\.?\d*)[eE]([+-]?\d+)").search(text)
    if sci_e_match:
        return str(float(text[sci_e_match.start():sci_e_match.end()]))

    # Currency like "$8,645.09"
    currency_match = get_compiled_regex(
        r"[\$€£¥]\s*([+-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?)"
    ).search(text)
    if currency_match:
        try:
            return str(float(currency_match.group(1).replace(",", "")))
        except ValueError:
            pass

    # Number with comma-separated thousands like "8,645.09"
    comma_match = get_compiled_regex(
        r"([+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?)"
    ).search(text)
    if comma_match:
        try:
            return str(float(comma_match.group(1).replace(",", "")))
        except ValueError:
            pass

    # Plain decimal number
    for num_str in get_compiled_regex(r"([+-]?\d+\.?\d*)").findall(text):
        if num_str in ("+", "-", "."):
            continue
        try:
            num = float(num_str)
            if abs(num) >= 0.0001 or num == 0:
                return str(num)
        except ValueError:
            continue

    return None


def extract_webinstruct_answer(text: str) -> Optional[str]:
    """Extract the numeric answer from a WebInstruct response."""
    if not text:
        return None

    match = get_compiled_regex(r"final answer:\s*([^\n]*)", re.IGNORECASE).search(text)
    if match:
        extracted = _extract_webinstruct_value(match.group(1).strip())
        if extracted:
            return extracted

    for pattern in (r"answer\s*:\s*([^\n]*)", r"the answer is\s*([^\n]*)", r"final answer is\s*([^\n]*)"):
        matches = get_compiled_regex(pattern, re.IGNORECASE | re.MULTILINE).findall(text)
        if matches:
            extracted = _extract_webinstruct_value(matches[0].strip())
            if extracted:
                return extracted

    for line in reversed(text.strip().split("\n")[-5:]):
        if line.strip():
            extracted = _extract_webinstruct_value(line)
            if extracted:
                return extracted

    return None


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def extract_answer(text: str, dataset_prefix: str) -> Optional[str]:
    """Extract the final answer based on dataset-specific format."""
    prefix = (dataset_prefix or "gsm8k").lower()
    if prefix == "math":
        return extract_math_answer(text)
    if prefix == "webinstruct":
        return extract_webinstruct_answer(text)
    return extract_gsm8k_answer(text)


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def extract_question_from_prompt(prompt: str) -> str:
    """Extract the question text from a prompt containing a 'Question:' line."""
    for line in prompt.split("\n"):
        if line.startswith("Question:"):
            return line.replace("Question:", "").strip()
    return prompt


_TASK_INSTRUCTIONS = {
    "math": (
        "You are given a competition-style mathematics problem. Provide a rigorous, step-by-step "
        "solution that clearly justifies each transformation. Present the final result inside a "
        "LaTeX boxed expression, i.e., write the answer as \\boxed{<answer>}.\n\n"
    ),
    "webinstruct": (
        "You are given a scientific or academic question that requires numerical reasoning. "
        "Your task is to carefully read the problem and provide a step-by-step solution.\n"
        "Provide a step-by-step reasoning process and then write the final numerical answer on a new line "
        "in the format: final answer: <answer>. Include units if applicable.\n\n"
    ),
    "gsm8k": (
        "You are given a grade school math word problem involving basic arithmetic, algebra, or geometry. "
        "Your task is to carefully read the problem and provide a step-by-step solution for it.\n"
        "Provide a step-by-step reasoning process and then write the final numerical answer on a new line "
        "in the format: final answer: <answer>.\n\n"
    ),
}


def get_task_instruction(dataset_prefix: str) -> str:
    """Return the task instruction string for the given dataset prefix."""
    return _TASK_INSTRUCTIONS.get((dataset_prefix or "gsm8k").lower(), _TASK_INSTRUCTIONS["gsm8k"])
