"""Rescore saved R001-R008 outputs against the current review and render a PDF.

python3 scripts/create_asr_comparison_pdf.py
Requires numpy, reportlab and pypdf. No ASR inference or cache mutations.
"""

import argparse
import hashlib
import json
import math
import sys
from fractions import Fraction
from pathlib import Path
from xml.sax.saxutils import escape

import reportlab
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Flowable, LongTable, Paragraph, SimpleDocTemplate, Spacer, TableStyle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lecture_recognition.evaluation import edit_score, reference_cards  # noqa: E402

CASES = (
    ("Qwen", "qwen", "3893fc67b39a3f75"),
    ("GigaAM CTC", "gigaam-ctc", "d08a9d751d2c098d"),
    ("GigaAM RNNT", "gigaam-rnnt", "d1615af5a08d5697"),
)
CARD_IDS = [f"R{i:03d}" for i in range(1, 9)]
GREEN = colors.HexColor("#DEEFDF")
YELLOW = colors.HexColor("#FFF0CE")
RED = colors.HexColor("#F3D8D6")
INK = colors.HexColor("#172D3D")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def wer_color(wer):
    position = min(1.0, max(0.0, wer / 0.30))
    left, right = (GREEN, YELLOW) if position <= 0.5 else (YELLOW, RED)
    weight = position * 2 if position <= 0.5 else (position - 0.5) * 2
    return colors.Color(*[a + (b - a) * weight for a, b in zip(left.rgb(), right.rgb())])


def timestamp(seconds):
    seconds, ms = divmod(round(seconds * 1000), 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{ms:03d}"


def read_rows(run, review):
    reference = load(run / "reference.json")
    current_cards = reference_cards(review.read_text(encoding="utf-8"), CARD_IDS)
    cards = {name: card for group in reference["groups"] for name, card in group["cards"].items()}
    if set(cards) != set(CARD_IDS):
        raise ValueError("Expected exactly R001-R008 in the frozen reference")
    records = []
    for _, model, case_id in CASES:
        case = load(run / "cases" / case_id / "case.json")
        if not (
            case["status"] == "ok"
            and case["mode"] == "compare"
            and case["config"]["model"] == model
            and case["config"]["dictionary"] is None
            and case["repeat"] == 0
            and case["shift"] == 0
        ):
            raise ValueError(f"Incorrect baseline case: {case_id}")
        records.append(case)
    rows = []
    for name in CARD_IDS:
        card = cards[name]
        cells = []
        for (label, model, case_id), record in zip(CASES, records):
            score = record["score"]["cards"][name]
            if score["reference"] != card["text"]:
                raise ValueError(f"Reference mismatch for {name}: {model}")
            if score["words"] <= 0 or not math.isclose(score["wer"], score["errors"] / score["words"]):
                raise ValueError(f"Inconsistent saved WER for {name}: {model}")
            updated = edit_score(current_cards[name], score["hypothesis"])
            cells.append({"model": label, "case_id": case_id, "text": score["hypothesis"],
                          "wer": updated["wer"], "errors": updated["errors"], "words": updated["words"],
                          "score": updated, "previous_wer": score["wer"]})
        best = min(Fraction(c["errors"], c["words"]) for c in cells)
        for cell in cells:
            cell["best"] = Fraction(cell["errors"], cell["words"]) == best
            cell["background"] = wer_color(cell["wer"]).hexval()
        rows.append({"id": name, "window": card["window"], "reference": current_cards[name], "models": cells})
    return rows


class Legend(Flowable):
    def __init__(self, width):
        super().__init__()
        self.width, self.height = width, 38

    def draw(self):
        canvas = self.canv
        canvas.setFont("DejaVu", 8)
        canvas.setFillColor(INK)
        canvas.drawString(0, 28, "Меньше WER - ближе к эталону")
        width = 140
        for i in range(140):
            canvas.setFillColor(wer_color(i / 139 * 0.30))
            canvas.rect(i, 12, 1.1, 8, stroke=0, fill=1)
        canvas.setFillColor(INK)
        canvas.setFont("DejaVu", 7)
        canvas.drawString(0, 2, "0%")
        canvas.drawCentredString(width / 2, 2, "15%")
        canvas.drawRightString(width, 2, "30% и выше")
        canvas.setFont("DejaVuBold", 8)
        canvas.drawString(160, 15, "Жирным выделен лучший результат в строке.")
        canvas.setFont("DejaVu", 8)
        canvas.drawString(160, 3, "При равном WER выделены все лучшие варианты. Цветовая шкала общая.")


def make_pdf(rows, output, run, review):
    font_dir = Path("/usr/share/fonts/truetype/dejavu")
    for name, file in [("DejaVu", "DejaVuSans.ttf"), ("DejaVuBold", "DejaVuSans-Bold.ttf")]:
        pdfmetrics.registerFont(TTFont(name, str(font_dir / file)))
    pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVuBold")
    page_width, page_height = landscape(A4)
    margin = 12 * mm
    available = page_width - 2 * margin
    widths = [38, 84] + [(available - 122) / 4] * 4
    styles = {
        "body": ParagraphStyle("body", fontName="DejaVu", fontSize=9, leading=12,
                               textColor=INK, alignment=TA_LEFT, splitLongWords=True),
        "best": ParagraphStyle("best", fontName="DejaVuBold", fontSize=9, leading=12, textColor=INK),
        "small": ParagraphStyle("small", fontName="DejaVu", fontSize=7.5, leading=10, textColor=INK),
        "id": ParagraphStyle("id", fontName="DejaVuBold", fontSize=8, leading=11, textColor=INK),
        "header": ParagraphStyle("header", fontName="DejaVuBold", fontSize=9, leading=12,
                                 textColor=colors.white),
        "title": ParagraphStyle("title", fontName="DejaVuBold", fontSize=18, leading=23, textColor=INK),
    }

    def paragraph(text, style="body"):
        return Paragraph(escape(text), styles[style])

    header = ["ID", "Промежуток времени", "Qwen", "GigaAM CTC", "GigaAM RNNT", "Ground truth"]
    table_rows = [[paragraph(text, "header") for text in header]]
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#21485A")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor("#21485A")),
        ("INNERGRID", (0, 1), (-1, -1), 0.35, colors.HexColor("#C8D1D7")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#C8D1D7")),
    ]
    for index, row in enumerate(rows, 1):
        times = "<br/>-<br/>".join(escape(timestamp(t)) for t in row["window"])
        cells = [paragraph(row["id"], "id"), Paragraph(times, styles["small"])]
        for col, cell in enumerate(row["models"], 2):
            metric = f"WER {cell['wer']:.2%}".replace(".", ",")
            cells.append([paragraph(metric, "small"), Spacer(1, 5),
                          paragraph(cell["text"], "best" if cell["best"] else "body")])
            commands.append(("BACKGROUND", (col, index), (col, index), wer_color(cell["wer"])))
        cells.append(paragraph(row["reference"]))
        table_rows.append(cells)
        for col in (0, 1, 5):
            commands.append(("BACKGROUND", (col, index), (col, index), colors.HexColor("#F2F5F7")))
    table = LongTable(table_rows, colWidths=widths, repeatRows=1, splitByRow=1, splitInRow=0,
                      hAlign="LEFT", style=TableStyle(commands))
    story = [paragraph("Сравнение транскрипций", "title"), Spacer(1, 5),
             paragraph("R001-R008 | Лекция 25.09.2026 | Исходные модели без словаря", "small"),
             Spacer(1, 10), Legend(available), Spacer(1, 8), table, Spacer(1, 10)]
    notes = [
        "WER - доля замен, удалений и вставок относительно числа слов эталона. "
        "Показатели пересчитаны по текущему эталону; транскрипции моделей взяты из сохранённого запуска.",
        "Время соответствует окнам оценки исправленных фраз, а не первоначальным интервалам "
        "прослушивания. Карточки пересекаются: их показатели нельзя складывать. "
        "Пограничные слова зависят от модельного выравнивания.",
        "Нормализация игнорирует регистр, пунктуацию и е/ё, но не полностью учитывает падежные формы "
        "числительных (например, «двадцати» и 20). Лучший WER не гарантирует сохранность всех важных смыслов.",
        f"Источник: запуск {run.name}; reference.json и score.cards исходных случаев. "
        "Qwen: 3893fc67b39a3f75; CTC: d08a9d751d2c098d; RNNT: d1615af5a08d5697.",
        f"Эталон: {review.name}; SHA-256: {hashlib.sha256(review.read_bytes()).hexdigest()[:16]}. "
        "Окна оценки сохранены; повторное распознавание не выполнялось.",
    ]
    for note in notes:
        story.extend([paragraph(note, "small"), Spacer(1, 4)])

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#C8D1D7"))
        canvas.line(margin, 9 * mm, page_width - margin, 9 * mm)
        canvas.setFont("DejaVu", 7)
        canvas.setFillColor(INK)
        canvas.drawString(margin, 5.5 * mm, "R001-R008 | Qwen / GigaAM CTC / GigaAM RNNT")
        canvas.drawRightString(page_width - margin, 5.5 * mm, f"Страница {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(str(output), pagesize=(page_width, page_height),
                            leftMargin=margin, rightMargin=margin, topMargin=margin, bottomMargin=margin,
                            title="Сравнение транскрипций R001-R008", author="lecture-recognition")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def verify_text(output, rows):
    reader = PdfReader(output)
    text = "\n".join(page.extract_text() for page in reader.pages)
    compact = "".join(text.split())
    for row in rows:
        if row["id"] not in text:
            raise ValueError(f"Missing ID in PDF: {row['id']}")
        for value in [row["reference"], *[cell["text"] for cell in row["models"]]]:
            if "".join(value.split()) not in compact:
                raise ValueError(f"Missing or changed text in PDF: {row['id']}")
        for value in row["window"]:
            if timestamp(value) not in text:
                raise ValueError(f"Missing timestamp in PDF: {row['id']}")
    return len(reader.pages)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path,
                        default=ROOT / ".lecture-cache/model-benchmark/9915a85fd8f93d0c")
    parser.add_argument("--output", type=Path, default=ROOT / "output/pdf/r001-r008-asr-comparison.pdf")
    parser.add_argument("--review", type=Path, default=ROOT / "record/20260925_101716.review.md")
    args = parser.parse_args()
    rows = read_rows(args.run_dir, args.review)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.stem + ".tmp.pdf")
    make_pdf(rows, temporary, args.run_dir, args.review)
    pages = verify_text(temporary, rows)
    temporary.replace(args.output)
    sources = [args.run_dir / "reference.json"] + [args.run_dir / "cases" / case / "case.json"
                                                    for _, _, case in CASES]
    sources.extend([args.review, ROOT / "src/lecture_recognition/evaluation.py",
                    ROOT / "src/lecture_recognition/experiments.py", Path(__file__).resolve()])
    manifest = {"pdf": str(args.output), "pages": pages, "reportlab": reportlab.Version,
                "sources": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                "scale": {"0": GREEN.hexval(), "0.15": YELLOW.hexval(), "0.30": RED.hexval()},
                "rows": rows}
    args.output.with_suffix(".json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                               encoding="utf-8")
    print(f"Created {args.output} ({pages} pages); all 32 text cells and timestamps verified.")
    for row in rows:
        print(row["id"], " / ".join(cell["model"] for cell in row["models"] if cell["best"]))


if __name__ == "__main__":
    main()
