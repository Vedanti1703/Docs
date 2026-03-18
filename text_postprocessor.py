import re
from typing import List


def merge_broken_lines(text: str) -> str:
    lines = text.split("\n")
    merged: List[str] = []
    buffer = ""

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if buffer:
                merged.append(buffer)
                buffer = ""
            merged.append("")
            continue

        if buffer:
            ends_sentence = buffer.rstrip()[-1] in ".!?:;)" if buffer.rstrip() else False
            next_starts_lower = stripped[0].islower() if stripped else False

            if not ends_sentence and next_starts_lower:
                buffer = buffer.rstrip() + " " + stripped
            else:
                merged.append(buffer)
                buffer = stripped
        else:
            buffer = stripped

    if buffer:
        merged.append(buffer)

    result_lines: List[str] = []
    prev_empty = False
    for line in merged:
        if line == "":
            if not prev_empty:
                result_lines.append(line)
            prev_empty = True
        else:
            result_lines.append(line)
            prev_empty = False

    return "\n".join(result_lines)


def remove_noise_characters(text: str) -> str:
    text = re.sub(r"(?<!\w)[|\\~`^]{1,2}(?!\w)", "", text)
    text = re.sub(r" {2,}", " ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.strip()


def apply_spell_check(text: str) -> str:
    try:
        from spellchecker import SpellChecker
        spell = SpellChecker()
        words = text.split()
        corrected_words = []

        for word in words:
            prefix = ""
            suffix = ""

            while word and not word[0].isalpha():
                prefix += word[0]
                word = word[1:]

            while word and not word[-1].isalpha():
                suffix = word[-1] + suffix
                word = word[:-1]

            if not word:
                corrected_words.append(prefix + suffix)
                continue

            if word[0].isupper() or len(word) <= 2 or any(c.isdigit() for c in word):
                corrected_words.append(prefix + word + suffix)
                continue

            correction = spell.correction(word.lower())
            if correction and correction != word.lower():
                corrected_words.append(prefix + correction + suffix)
            else:
                corrected_words.append(prefix + word + suffix)

        return " ".join(corrected_words)

    except ImportError:
        return text


def postprocess_text(text: str, spell_check: bool = True) -> str:
    if not text or not text.strip():
        return ""
    text = remove_noise_characters(text)
    text = merge_broken_lines(text)
    if spell_check:
        text = apply_spell_check(text)
    return text.strip()