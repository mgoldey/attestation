"""Generate tests/fixtures/research/one_page.pdf: a minimal hand-written one-page PDF.

Committed once so `test_pdf_text_extracts_and_tolerates_garbage` has a stable,
readable-by-pypdf fixture rather than depending on pypdf's own internals (the
brief's `_pdf_bytes` writer helper pokes at private `PdfWriter` attributes and
broke against this pypdf version). Run `uv run python
tests/fixtures/research/make_one_page_pdf.py` to regenerate.

The PDF is the smallest structure pypdf's `PdfReader.pages[i].extract_text()`
can read: a catalog, one page with a Helvetica font resource, and a content
stream drawing the text "Hello full text" with a `Tj` show-text operator.
"""

from pathlib import Path

TEXT = "Hello full text"


def build() -> bytes:
    """Assemble the PDF byte-for-byte, computing the xref offsets as we go."""
    content = f"BT /F1 12 Tf 10 100 Td ({TEXT}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200]"
        b" /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_offset = len(out)
    n = len(objects) + 1
    out += f"xref\n0 {n}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets[1:]:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {n} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF".encode()
    return bytes(out)


if __name__ == "__main__":
    dest = Path(__file__).parent / "one_page.pdf"
    dest.write_bytes(build())
    print(f"wrote {dest} ({dest.stat().st_size} bytes)")
