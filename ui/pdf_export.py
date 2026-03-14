from __future__ import annotations

import re
from datetime import datetime
from io import BytesIO
from typing import List, Dict

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def _escape_text(text: str) -> str:
    if not text:
        return ""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _convert_markdown_bold_to_html(text: str) -> str:
    escaped = _escape_text(text)
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)


def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r"^\*\*\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n\s*\*\*\s*\n", "\n\n", text)
    text = re.sub(r"^\s*\*\*\s*", "", text)
    text = re.sub(r"\s*\*\*\s*$", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_content_blocks(text: str) -> List[Dict[str, object]]:
    text = _clean_text(text)
    if not text:
        return []

    lines = [line.rstrip() for line in text.splitlines()]
    blocks: List[Dict[str, object]] = []
    paragraph_buffer: List[str] = []
    bullet_buffer: List[str] = []

    def flush_paragraph():
        nonlocal paragraph_buffer
        if paragraph_buffer:
            paragraph_text = "\n".join(paragraph_buffer).strip()
            if paragraph_text:
                blocks.append({"type": "paragraph", "content": paragraph_text})
            paragraph_buffer = []

    def flush_bullets():
        nonlocal bullet_buffer
        if bullet_buffer:
            blocks.append({"type": "bullets", "content": bullet_buffer[:]})
            bullet_buffer = []

    for line in lines:
        stripped = line.strip()

        if not stripped:
            flush_paragraph()
            flush_bullets()
            continue

        bullet_match = re.match(r"^[-*•]\s+(.*)", stripped)
        if bullet_match:
            flush_paragraph()
            bullet_buffer.append(bullet_match.group(1).strip())
        else:
            flush_bullets()
            paragraph_buffer.append(stripped)

    flush_paragraph()
    flush_bullets()

    return blocks


def build_answer_pdf(
    question: str,
    sections: Dict[str, str],
    sources: List[Dict[str, str]],
    app_title: str = "Product Assistant for Sales & Marketing",
) -> bytes:
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="Chatbot Answer Export",
        author="Marketing RAG Chatbot",
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=27,
        textColor=colors.HexColor("#1D4ED8"),
        alignment=TA_LEFT,
        spaceAfter=10,
    )

    subtitle_style = ParagraphStyle(
        "SubtitleStyle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#475569"),
        spaceAfter=14,
    )

    section_heading_style = ParagraphStyle(
        "SectionHeadingStyle",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=6,
        spaceBefore=10,
    )

    body_style = ParagraphStyle(
        "BodyStyle",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10.5,
        leading=16,
        textColor=colors.HexColor("#111827"),
        spaceAfter=6,
    )

    bullet_style = ParagraphStyle(
        "BulletStyle",
        parent=body_style,
        leftIndent=0,
        firstLineIndent=0,
        spaceAfter=2,
    )

    small_style = ParagraphStyle(
        "SmallStyle",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#334155"),
    )

    story = []

    exported_at = datetime.now().strftime("%d %b %Y, %I:%M %p")

    story.append(Paragraph(_escape_text(app_title), title_style))
    story.append(Paragraph(f"Exported on {exported_at}", subtitle_style))

    story.append(Paragraph("Question", section_heading_style))
    story.append(Paragraph(_convert_markdown_bold_to_html(question).replace("\n", "<br/>"), body_style))
    story.append(Spacer(1, 4))

    section_map = [
        ("Direct Answer", sections.get("direct_answer", "")),
        ("Relevant Product Details", sections.get("relevant_details", "")),
        ("Gaps / Validation Needed", sections.get("gaps_validation", "")),
        ("Simple Explanation", sections.get("simple_explanation", "")),
    ]

    for heading, content in section_map:
        if content and _clean_text(content):
            story.append(Paragraph(_escape_text(heading), section_heading_style))

            blocks = _split_content_blocks(content)

            for block in blocks:
                if block["type"] == "paragraph":
                    paragraph_html = _convert_markdown_bold_to_html(str(block["content"])).replace("\n", "<br/>")
                    story.append(Paragraph(paragraph_html, body_style))
                    story.append(Spacer(1, 2))

                elif block["type"] == "bullets":
                    bullet_items = [
                        ListItem(
                            Paragraph(_convert_markdown_bold_to_html(item), bullet_style),
                            leftIndent=10,
                        )
                        for item in block["content"]
                    ]
                    story.append(
                        ListFlowable(
                            bullet_items,
                            bulletType="bullet",
                            start="circle",
                            leftIndent=12,
                        )
                    )
                    story.append(Spacer(1, 4))

    if sources:
        story.append(Spacer(1, 8))
        story.append(Paragraph("Sources Used", section_heading_style))

        table_data = [["#", "File", "Unit", "Page"]]
        for src in sources:
            table_data.append([
                str(src.get("source_number", "")),
                _escape_text(str(src.get("file_name", ""))),
                _escape_text(str(src.get("unit", ""))),
                _escape_text(str(src.get("page", ""))),
            ])

        col_widths = [12 * mm, 96 * mm, 28 * mm, 18 * mm]

        table = Table(table_data, colWidths=col_widths, repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E2E8F0")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                    ("LEADING", (0, 0), (-1, -1), 12),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("ALIGN", (0, 0), (0, -1), "CENTER"),
                    ("ALIGN", (3, 1), (3, -1), "CENTER"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 7),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ]
            )
        )
        story.append(table)

    story.append(Spacer(1, 10))
    story.append(
        Paragraph(
            "Generated by the Marketing RAG Chatbot. Please validate critical technical decisions against source documents.",
            small_style,
        )
    )

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes