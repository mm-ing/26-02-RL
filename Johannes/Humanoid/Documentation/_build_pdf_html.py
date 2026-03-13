from pathlib import Path
import markdown

md_path = Path('Documentation/Documentation.md')
out_html = Path('Documentation/Documentation_compiled.html')
md = md_path.read_text(encoding='utf-8')
body = markdown.markdown(md, extensions=['extra', 'sane_lists', 'toc'])
css = ('body{font-family:Segoe UI,Arial,sans-serif;line-height:1.45;margin:28px;max-width:1100px}'
       ' img{max-width:100%;height:auto}'
       ' table{border-collapse:collapse;width:100%}'
       ' th,td{border:1px solid #bbb;padding:6px 8px;text-align:left}'
       ' code{background:#f4f4f4;padding:1px 3px}')
html = ('<!doctype html><html><head><meta charset="utf-8"><title>Humanoid Documentation</title>'
        f'<style>{css}</style></head><body>{body}</body></html>')
out_html.write_text(html, encoding='utf-8')
print(out_html)
