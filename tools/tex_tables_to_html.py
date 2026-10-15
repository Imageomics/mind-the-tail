"""Regenerate the benchmark tables on the project page from the paper's LaTeX.

Reads tables/classification.tex and tables/multimodal_pretrain.tex (one level
above project_page/) and rewrites the block between the BEGIN/END
results-tables markers in index.html.

    python3 tools/tex_tables_to_html.py
"""
import re
from pathlib import Path

PAGE = Path(__file__).resolve().parents[1]
PAPER = PAGE.parent
RATES = ["10%", "20%", "30%", "50%", "70%"]

TABLES = [
    {
        "tex": PAPER / "tables" / "classification.tex",
        "title": "Image classification",
        "sub": "ResNet-50 trained on the selected subset",
        "open": False,
        "benches": [
            ("NABirds", "Accuracy"),
            ("Stanford Cars", "Accuracy"),
            ("Waterbirds", "Worst-group accuracy"),
            ("Spawrious", "Accuracy"),
        ],
        "caption": "Averaged over three runs. <b>Bold</b>: best; <u>underlined</u>: second best.",
    },
    {
        "tex": PAPER / "tables" / "multimodal_pretrain.tex",
        "title": "Multimodal pretraining",
        "sub": "CLIP ViT-B/16 trained on a CC12M subset",
        "open": False,
        "benches": [
            ("ImageNet-1K zero-shot", "Accuracy"),
            ("ImageNet-1K linear probe", "Accuracy"),
            ("Flickr30K image&rarr;text", "Recall@1"),
            ("Flickr30K text&rarr;image", "Recall@1"),
        ],
        "caption": "<b>Bold</b>: best; <u>underlined</u>: second best.",
    },
]


def clean_name(s):
    s = s.strip()
    s = re.sub(r"\\rowcolor\{[^}]*\}", "", s).strip()
    s = s.replace(r"$\mathbf{D^2}$", "D<sup>2</sup>")
    m = re.fullmatch(r"\\textbf\{(.*)\}", s)
    if m:
        s = m.group(1)
    return s.strip()


def parse_cell(s):
    s = s.strip().strip("$").strip()
    if s == "-":
        return None, ""
    if s.startswith(r"\mathbf{"):
        return s[len(r"\mathbf{"):-1], "best"
    if s.startswith(r"\underline{"):
        return s[len(r"\underline{"):-1], "second"
    return s, ""


def parse_rows(tex_path):
    text = tex_path.read_text()
    body = text.split(r"\midrule[1pt]", 1)[1].split(r"\bottomrule", 1)[0]
    body = re.sub(r"\\midrule\[[^\]]*\]", "", body)
    rows = []
    for raw in body.split(r"\\"):
        raw = " ".join(raw.split())
        if "&" not in raw:
            continue
        parts = raw.split("&")
        name = clean_name(parts[0])
        cells = [parse_cell(c) for c in parts[1:]]
        rows.append((name, cells))
    return rows


def bench_table(rows, b, name, metric):
    cols = slice(b * 6, b * 6 + 5)
    full = None
    out = [f'      <div class="bench">\n        <h4>{name} <span class="metric">{metric}</span></h4>',
           '        <div class="table-wrap"><table class="data-table">',
           "          <thead><tr><th>Method</th>" + "".join(f"<th>{r}</th>" for r in RATES) + "</tr></thead>",
           "          <tbody>"]
    for method, cells in rows:
        if method == "Random":
            full = cells[b * 6 + 5][0]
        cls = ' class="ours"' if method == "Ours" else (' class="ref"' if method == "Random" else "")
        tds = []
        for val, mark in cells[cols]:
            v = "&ndash;" if val is None else val
            tds.append(f'<td><span class="{mark}">{v}</span></td>' if mark else f"<td>{v}</td>")
        out.append(f"            <tr{cls}><td>{method}</td>{''.join(tds)}</tr>")
    out.append("          </tbody>\n        </table></div>")
    if full:
        out.append(f'        <p class="full-ref">Full dataset (100%): {full}</p>')
    out.append("      </div>")
    return "\n".join(out)


def build():
    blocks = []
    for t in TABLES:
        rows = parse_rows(t["tex"])
        benches = "\n".join(bench_table(rows, i, n, m) for i, (n, m) in enumerate(t["benches"]))
        blocks.append(
            f'    <details class="more-box"{" open" if t["open"] else ""}>\n'
            f'      <summary><i class="fas fa-table" aria-hidden="true"></i>{t["title"]} '
            f'<span class="sum-sub">{t["sub"]}</span><i class="fas fa-chevron-down chev" aria-hidden="true"></i></summary>\n'
            f'      <div class="more-body">\n'
            f'    <div class="bench-grid">\n{benches}\n    </div>\n'
            f'    <p class="table-caption">{t["caption"]}</p>\n'
            f"      </div>\n    </details>"
        )
    return "\n".join(blocks)


def main():
    index = PAGE / "index.html"
    html = index.read_text()
    begin = re.search(r"<!-- BEGIN:results-tables[^>]*-->\n", html)
    end = html.index("<!-- END:results-tables -->")
    html = html[: begin.end()] + build() + "\n" + html[end:]
    index.write_text(html)
    print("Updated", index)


if __name__ == "__main__":
    main()
