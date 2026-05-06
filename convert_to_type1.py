"""Re-render text in matplotlib-generated PDFs using PDF base Type 1 fonts.

Strategy:
1. Walk every text span in the page (PyMuPDF's "dict" extraction).
2. Add a redaction annotation per span; apply with graphics preservation
   so the radar lines / shapes stay intact while the Type 3 glyph paint is
   removed.
3. Re-insert the same string at the original baseline using "tiro"
   (Times-Roman) or "tibo" (Times-Bold) — both are PDF 14-base Type 1
   fonts (no embedding required, conference-compliant).
"""

import re
import sys
import fitz

LEGEND_RENAME = {
    "diverse_prompt": "Diverse Prompt",
    "min_p": "Min-p",
    "top_p": "Top-p",
    "top_k": "Top-k",
    "normal": "Normal",
}


def color_to_rgb01(c: int):
    return ((c >> 16) & 0xFF) / 255.0, ((c >> 8) & 0xFF) / 255.0, (c & 0xFF) / 255.0


def convert(in_path: str, out_path: str):
    doc = fitz.open(in_path)
    for page in doc:
        spans = []
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                for s in line["spans"]:
                    spans.append(s)

        # Slightly inflate redact rect so antialiased glyph edges are also wiped.
        pad = 0.5
        for s in spans:
            x0, y0, x1, y1 = s["bbox"]
            page.add_redact_annot((x0 - pad, y0 - pad, x1 + pad, y1 + pad))

        # graphics defaults to PDF_REDACT_LINE_ART_NONE (preserve drawings).
        page.apply_redactions()

        # Re-insert text first (this registers tiro/tibo in page resources).
        for s in spans:
            x0, y0, x1, y1 = s["bbox"]
            size = s["size"]
            is_bold = "Bold" in s["font"]
            fontname = "tibo" if is_bold else "tiro"
            color = color_to_rgb01(s["color"])
            # bbox spans ascender..descender. Baseline is roughly y1 - 0.22*size
            # for both DejaVuSans and Times — close enough for visual parity.
            baseline_y = y1 - 0.22 * size
            text = LEGEND_RENAME.get(s["text"], s["text"])
            page.insert_text(
                (x0, baseline_y),
                text,
                fontname=fontname,
                fontsize=size,
                color=color,
            )

        # Drop unused Type 3 font references from /Resources/Font.
        # After redaction the content stream no longer references them, but
        # the resource dict still lists them — pdffonts and conference
        # validators inspect that dict, so we must remove them explicitly.
        contents = page.read_contents().decode("latin-1", errors="replace")
        font_dict = doc.xref_get_key(page.xref, "Resources/Font")
        if font_dict and font_dict[0] == "dict":
            body = font_dict[1]
            entries = re.findall(r"/(\w+)\s+(\d+\s+\d+\s+R)", body)
            kept = [
                (name, ref) for name, ref in entries
                if re.search(rf"/{re.escape(name)}\b", contents)
            ]
            new_body = "<<" + "".join(f"/{n} {r}" for n, r in kept) + ">>"
            doc.xref_set_key(page.xref, "Resources/Font", new_body)

    doc.save(out_path, garbage=4, deflate=True, clean=True)
    doc.close()


if __name__ == "__main__":
    targets = [
        "/data1/toby/wjx/TAP_trado/quality_radar_llada_trado.pdf",
        "/data1/toby/wjx/TAP_trado/quality_radar_llada_trado-T0.8.pdf",
    ]
    for path in targets:
        src = path + ".bak"
        convert(src, path)
        print(f"Wrote {path}")
