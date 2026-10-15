# Mind the Tail — project page

Static page (no build step). Open `index.html` through any static server:

```bash
python3 -m http.server 8765 --directory project_page
```

## Layout

- `index.html` — the page
- `static/css/style.css`, `static/js/main.js` — styles and small scripts (contents menu, copy BibTeX)
- `static/images/` — figures converted from `../figs/*.pdf` (WebP) and `og_teaser.jpg` for link previews
- `tools/tex_tables_to_html.py` — regenerates the benchmark tables from `../tables/*.tex`

## Deploy

This folder is the `gh-pages` branch of `Imageomics/mind-the-tail`; GitHub Pages serves the branch root.
Commit and `git push` to update the site.

## Before publishing

- Fill in the link buttons marked `data-todo="true"` in `index.html` (Paper, Code, arXiv, Models, Data)
- Add author homepage links
- Replace the BibTeX placeholder with the arXiv entry
- Make `og:image` / `twitter:image` absolute URLs once the domain is known

## Updating figures

```bash
inkscape --pdf-poppler ../figs/NAME.pdf --export-type=png --export-background=white -w 2400 -o /tmp/NAME.png
cwebp -q 88 -m 6 /tmp/NAME.png -o static/images/NAME.webp
```
