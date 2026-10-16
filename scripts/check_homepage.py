"""Check the buildless homepage's local links and asset references (stdlib only)."""

from html.parser import HTMLParser
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.refs = []
        self.duplicates = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            if attrs["id"] in self.ids:
                self.duplicates.append(attrs["id"])
            self.ids.add(attrs["id"])
        for key in ("src", "href"):
            if key in attrs:
                self.refs.append(attrs[key])
        if "srcset" in attrs:
            self.refs.extend(part.strip().split()[0] for part in attrs["srcset"].split(","))
        if tag == "img" and "alt" not in attrs:
            raise ValueError("Image missing alt attribute")


def main():
    page = Page()
    page.feed((SITE / "index.html").read_text(encoding="utf-8"))
    errors = [f"Duplicate id: {item}" for item in page.duplicates]
    for ref in page.refs:
        url = urlsplit(ref)
        if url.scheme or url.netloc:
            continue
        if not url.path:
            if url.fragment and unquote(url.fragment) not in page.ids:
                errors.append(f"Missing anchor: {ref}")
        else:
            target = (SITE / unquote(url.path)).resolve()
            if not target.is_relative_to(SITE) or not target.is_file():
                errors.append(f"Missing or out-of-site reference: {ref}")
    # Also check assets referenced dynamically by JavaScript and the README.
    script = (SITE / "main.js").read_text(encoding="utf-8")
    for ref in re.findall(r"['\"](assets/[^'\"]+)['\"]", script):
        if not (SITE / ref).is_file():
            errors.append(f"Missing JavaScript asset: {ref}")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for ref in re.findall(r"(?:src=[\"']|\]\()(site/assets/[^\"')]+)", readme):
        if not (ROOT / ref).is_file():
            errors.append(f"Missing README asset: {ref}")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Homepage OK: {len(page.ids)} unique ids, {len(page.refs)} HTML references; JS and README assets exist.")


if __name__ == "__main__":
    main()
