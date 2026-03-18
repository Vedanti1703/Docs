import re
from dataclasses import dataclass
from typing import List


@dataclass
class TextBlock:
    text: str
    block_type: str  # 'heading1' | 'heading2' | 'bullet' | 'numbered' | 'paragraph'


def _is_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 80:
        return False
    if stripped[-1] in ".!?,;":
        return False
    if re.match(r"^[-*•\d]", stripped):
        return False
    if len(stripped.split()) > 10:
        return False
    return True


def _heading_level(line: str) -> str:
    stripped = line.strip()
    if stripped.isupper() or len(stripped.split()) <= 3:
        return "heading1"
    return "heading2"


def _is_bullet(line: str) -> bool:
    return bool(re.match(r"^[-*•>–—]\s+\S", line.strip()))


def _is_numbered(line: str) -> bool:
    return bool(re.match(r"^(\d+[.)]\s+|\(\d+\)\s+)", line.strip()))


def _clean_bullet_prefix(line: str) -> str:
    line = re.sub(r"^[-*•>–—]\s+", "", line.strip())
    line = re.sub(r"^\d+[.)]\s+", "", line)
    line = re.sub(r"^\(\d+\)\s+", "", line)
    return line.strip()


def detect_structure(text: str) -> List[TextBlock]:
    lines = text.split("\n")
    blocks: List[TextBlock] = []
    paragraph_buffer: List[str] = []

    def flush_paragraph():
        if paragraph_buffer:
            para_text = " ".join(paragraph_buffer).strip()
            if para_text:
                blocks.append(TextBlock(text=para_text, block_type="paragraph"))
            paragraph_buffer.clear()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            flush_paragraph()
            continue

        if _is_bullet(stripped):
            flush_paragraph()
            blocks.append(TextBlock(text=_clean_bullet_prefix(stripped), block_type="bullet"))
        elif _is_numbered(stripped):
            flush_paragraph()
            blocks.append(TextBlock(text=_clean_bullet_prefix(stripped), block_type="numbered"))
        elif _is_heading(stripped):
            flush_paragraph()
            blocks.append(TextBlock(text=stripped, block_type=_heading_level(stripped)))
        else:
            paragraph_buffer.append(stripped)

    flush_paragraph()
    return blocks


def blocks_to_plain_text(blocks: List[TextBlock]) -> str:
    lines = []
    for block in blocks:
        if block.block_type == "heading1":
            lines.append(f"\n{block.text.upper()}\n")
        elif block.block_type == "heading2":
            lines.append(f"\n{block.text}\n")
        elif block.block_type == "bullet":
            lines.append(f"  • {block.text}")
        elif block.block_type == "numbered":
            lines.append(f"  {block.text}")
        else:
            lines.append(block.text)
    return "\n".join(lines)