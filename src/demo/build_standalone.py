"""
Step 3 of the ICU demonstration: bundle dashboard.html + demo_data.json
into a single, portable, self-contained HTML file.

WHY THIS EXISTS: dashboard.html normally fetches demo_data.json at
runtime, which requires a local server (`python -m http.server`) because
browsers block fetch() on file:// URLs. That extra step is one more thing
to remember — and one more thing that can go wrong — right before a viva
or when sharing the demo. This script embeds the data directly into the
page as a JSON <script> tag, so the output file opens by double-clicking
it, with no server and no separate data file to keep alongside it.

dashboard.html itself is untouched as a development template — it still
works with a local server for quick editing/iteration (see its init()
function, which checks for the inline tag first and falls back to
fetch()). Only the BUNDLED OUTPUT file is single-file/standalone.

Usage:
    python -m src.demo.build_standalone
    python -m src.demo.build_standalone --dashboard src/demo/dashboard.html --data outputs/demo_data.json --out outputs/demo/index.html
"""

import argparse
import json
from pathlib import Path

INLINE_MARKER = "</head>"


def build_standalone(dashboard_path: Path, data_path: Path, out_path: Path) -> None:
    html = dashboard_path.read_text(encoding="utf-8")

    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)  # parse-and-reserialise: catches malformed JSON early, not at demo time

    if 'id="demo-data-inline"' in html:
        raise ValueError(
            f"{dashboard_path} already contains an embedded data tag — "
            "run this against the original template, not a previous bundle."
        )
    if INLINE_MARKER not in html:
        raise ValueError(f"Could not find '{INLINE_MARKER}' in {dashboard_path} to inject data before.")

    # `</` inside the JSON payload (unlikely here, but cheap to guard)
    # would otherwise prematurely close the <script> tag if it ever spelled
    # out "</script"; escaping the forward slash neutralises that.
    embedded_json = json.dumps(data).replace("</", "<\\/")
    inline_tag = f'<script type="application/json" id="demo-data-inline">{embedded_json}</script>\n{INLINE_MARKER}'

    standalone_html = html.replace(INLINE_MARKER, inline_tag, 1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(standalone_html, encoding="utf-8")

    size_kb = out_path.stat().st_size / 1024
    print(f"Wrote standalone demo to {out_path} ({size_kb:.0f} KB) — open it directly, no server needed.")


def main():
    parser = argparse.ArgumentParser(description="Bundle the ICU demo into a single standalone HTML file.")
    parser.add_argument("--dashboard", type=Path, default=Path("src/demo/dashboard.html"))
    parser.add_argument("--data", type=Path, default=Path("outputs/demo_data.json"))
    parser.add_argument("--out", type=Path, default=Path("outputs/demo/index.html"))
    args = parser.parse_args()

    build_standalone(args.dashboard, args.data, args.out)


if __name__ == "__main__":
    main()
