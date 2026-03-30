import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

SRC = 'e:/pycode/all-in-rag/gradio_app.py'

with open(SRC, encoding='utf-8') as fh:
    text = fh.read()

# The bad lines have literal newlines inside the f-string.
# Replace the two broken lines with correct ones.
bad = 'summary = f"\u5171\u5904\u7406 {len(file_objs)} \u4e2a\u6587\u4ef6\uff0c\u5408\u8ba1\u7d22\u5f15 {total_indexed} \u4e2a\u5757\u3002\n\n"\n    return summary + "\n".join(results)'
good = 'summary = f"\u5171\u5904\u7406 {len(file_objs)} \u4e2a\u6587\u4ef6\uff0c\u5408\u8ba1\u7d22\u5f15 {total_indexed} \u4e2a\u5757\u3002" + "\\n\\n"\n    return summary + "\\n".join(results)'

if bad in text:
    text = text.replace(bad, good, 1)
    print('OK: fixed summary line')
else:
    # fallback: find and fix by splitting on the literal-newline pattern
    import re
    # match: summary = f"...<literal newline><literal newline>"
    pattern = re.compile(
        r'summary = f"(.*?)\n\n"\n(    return summary \+ "\n"\.join\(results\))',
        re.DOTALL
    )
    def replacer(m):
        content = m.group(1).replace('\n', '')
        return f'summary = "{content}" + "\\n\\n"\n    return summary + "\\n".join(results)'
    if pattern.search(text):
        text = pattern.sub(replacer, text, count=1)
        print('OK: fixed summary line (regex fallback)')
    else:
        print('ERROR: pattern not found')
        sys.exit(1)

with open(SRC, 'w', encoding='utf-8') as fh:
    fh.write(text)
print('OK: written')
