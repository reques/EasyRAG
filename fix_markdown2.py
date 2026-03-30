import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

SRC = 'e:/pycode/all-in-rag/gradio_app.py'

with open(SRC, encoding='utf-8') as fh:
    lines = fh.readlines()

# Find and replace the broken gr.Markdown block (lines 248-254, 1-indexed)
# We know it starts at line 248 (0-indexed: 247)
start = 247
end = 253  # inclusive, 0-indexed

print('Before:')
for i in range(start, end+1):
    print(f'{i+1:4}: {repr(lines[i])}')

# Build the replacement as a single joined string, then split into lines
replacement = '            gr.Markdown(\n                "### \u4e0a\u4f20\u6587\u6863\u5230\u77e5\u8bc6\u5e93\\n"\n                "\u652f\u6301\uff1a`.txt`  `.md`  `.pdf`  `.docx`\\n"\n                "> \u53ef\u4e00\u6b21\u9009\u62e9\u591a\u4e2a\u6587\u4ef6\u6279\u91cf\u4e0a\u4f20"\n            )\n'
replacement_lines = [l + '\n' for l in replacement.split('\n')]
# split gives trailing empty string, remove it
if replacement_lines[-1] == '\n':
    replacement_lines.pop()

print('\nReplacement lines:')
for ln in replacement_lines:
    print(repr(ln))

lines[start:end+1] = replacement_lines

with open(SRC, 'w', encoding='utf-8') as fh:
    fh.writelines(lines)
print('\nOK: written')

import ast
try:
    ast.parse(open(SRC, encoding='utf-8').read())
    print('OK: syntax valid')
except SyntaxError as e:
    print(f'SyntaxError at line {e.lineno}: {e.msg}')
    with open(SRC, encoding='utf-8') as fh:
        all_lines = fh.readlines()
    lo = max(0, e.lineno - 3)
    hi = min(len(all_lines), e.lineno + 3)
    for idx, ln in enumerate(all_lines[lo:hi], lo + 1):
        print(f'{idx:4}: {repr(ln)}')
