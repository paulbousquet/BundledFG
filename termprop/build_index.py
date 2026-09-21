"""Re-embed the dashboard's data into index.html with all adjustments stripped.

Reads ../data/ed_quadratic_fits.csv (the snapshot the paper's adjusted tables use), sets
every Modified column equal to its Base counterpart, blanks UpdatedAt, and rewrites only the
`const CSV_TEXT = "...";` line of index.html in this folder. Everything else in the file,
including its line endings, is left byte-for-byte as it was.

Run after refreshing data/ed_quadratic_fits.csv:

    python termprop/build_index.py
"""
from __future__ import annotations

import csv
import io
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "ed_quadratic_fits.csv"
INDEX = HERE / "index.html"

RESET_PAIRS = [
    ("Modified", "Base"),
    ("ModifiedFit", "BaseFit"),
    ("ModifiedCoefA", "BaseCoefA"),
    ("ModifiedCoefB", "BaseCoefB"),
    ("ModifiedCoefC", "BaseCoefC"),
]
REQUIRED = {"Date", "Horizon", "HorizonIndex", "SourceED2", "SourceValue", "UpdatedAt",
            *{c for pair in RESET_PAIRS for c in pair}}
# Only horizontal whitespace after the semicolon, so the line's own newline is never consumed;
# the lookahead tolerates a CR so this still matches if the file is checked out with CRLF.
CSV_LINE = re.compile(r'^([ \t]*const CSV_TEXT = )"(?:[^"\\]|\\.)*";[ \t]*(?=\r?$)', re.M)


def stripped_csv_text(path: Path) -> tuple[str, int]:
    if not path.is_file():
        raise SystemExit(f"data file not found: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    missing = sorted(REQUIRED - set(fieldnames))
    if missing:
        raise SystemExit(f"{path.name} is missing required columns: {', '.join(missing)}")
    if not rows:
        raise SystemExit(f"{path.name} has a header but no rows; refusing to embed an empty dataset")

    for row in rows:
        for target, source in RESET_PAIRS:
            row[target] = row[source]
        row["UpdatedAt"] = ""

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue(), len(rows)


def js_string_literal(text: str) -> str:
    """JSON-encode for a JS string, then make it safe inside an inline <script> block.

    json.dumps handles quotes, backslashes, and newlines, but leaves `<` alone, so a
    literal `</script>` in a CSV field would end the script element. Escaping every `<`
    as `\\u003c` is a no-op for the JS parser and closes that hole.
    """
    return json.dumps(text).replace("<", "\\u003c")


def main() -> None:
    text, n_rows = stripped_csv_text(DATA)
    with INDEX.open("r", encoding="utf-8", newline="") as handle:
        html = handle.read()
    matches = CSV_LINE.findall(html)
    match = CSV_LINE.search(html)
    if match is None or len(matches) != 1:
        raise SystemExit(f"expected exactly one `const CSV_TEXT = \"...\";` line in {INDEX.name}, found {len(matches)}")
    html = html[: match.start()] + f"{match.group(1)}{js_string_literal(text)};" + html[match.end():]
    with INDEX.open("w", encoding="utf-8", newline="") as handle:
        handle.write(html)
    dates = len({row.split(",", 1)[0] for row in text.splitlines()[1:]})
    print(f"embedded {n_rows} rows ({dates} announcements) from {DATA.name}; Modified == Base on every row")


if __name__ == "__main__":
    try:
        main()
    except SystemExit as exc:
        if exc.code and not isinstance(exc.code, int):
            print(f"build_index.py: {exc.code}", file=sys.stderr)
            sys.exit(1)
        raise
