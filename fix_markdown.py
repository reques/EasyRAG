import sys, re
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

SRC = 'e:/pycode/all-in-rag/gradio_app.py'

with open(SRC, encoding='utf-8') as fh:
    lines = fh.readlines()

# Lines 248-251 (0-indexed 247-250) contain the broken gr.Markdown call.
# Replace them with a single-string version.
# Find the line index by content search.
for i, ln in enumerate(lines):
    if 'gr.Markdown(' in ln and i+1 < len(lines) and '\u4e0a\u4f20\u6587\u6863\u5230\u77e5\u8bc6\u5e93' in lines[i+1]:
        start = i
        break
else:
    print('ERROR: marker not found')
    sys.exit(1)

# Find the closing paren line
end = start
for j in range(start, min(start+10, len(lines))):
    if lines[j].rstrip().endswith(')'):
        end = j
        break

print(f'Replacing lines {start+1}-{end+1}')
for ln in lines[start:end+1]:
    print(repr(ln))

NEW_LINES = [
    '            gr.Markdown(\n',
    '                "### \u4e0a\u4f20\u6587\u6863\u5230\u77e5\u8bc6\u5e93\\n"\'\n',
    '                "\u652f\u6301\uff1a`.txt`  `.md`  `.pdf`  `.docx`\\n"\'\n',
    '                "> \u53ef\u4e00\u6b21\u9009\u62e9\u591a\u4e2a\u6587\u4ef6\u6279\u91cf\u4e0a\u4f20"\n',
    '            )\n',
]

lines[start:end+1] = NEW_LINES

with open(SRC, 'w', encoding='utf-8') as fh:
    fh.writelines(lines)
print('OK: written')

import ast
try:
    ast.parse(open(SRC, encoding='utf-8').read())
    print('OK: syntax valid')
except SyntaxError as e:
    print(f'SyntaxError at line {e.lineno}: {e.msg}')
    # show context
    with open(SRC, encoding='utf-8') as fh:
        all_lines = fh.readlines()
    lo = max(0, e.lineno-3)
    hi = min(len(all_lines), e.lineno+2)
    for idx, ln in enumerate(all_lines[lo:hi], lo+1):
        print(f'{idx:4}: {repr(ln)}')
