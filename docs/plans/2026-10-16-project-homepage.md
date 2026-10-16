# EasyRAG Project Homepage Implementation Plan

**Goal:** Refine the supplied owl identity into a lively 3D mascot, build a Chinese GitHub Pages homepage, and embed the icon, animated overview and homepage entry in README.

**Architecture:** A buildless HTML/CSS/JavaScript site in `site/`, independent of the Vue application and backend. Local assets are shared with README; GitHub Actions deploys only `site/`. All capability descriptions come from repository documentation and implementation. Demonstrations are explicitly illustrative.

**Tech Stack:** Semantic HTML, responsive CSS, vanilla JavaScript, generated PNG/WebP brand assets, Pillow for GIF animation packaging, GitHub Pages Actions.

## Design

User selected a lively 3D mascot direction. Preserve the supplied white owl/document silhouette in the compact icon, with a friendly sculpted mascot for the hero. Midnight indigo, icy cyan and lavender form the palette. Use substantial typography, generous space, floating documents and gentle motion. Chinese is the primary language. Honor reduced-motion preferences and provide a static animation poster.

## Implementation

1. Save generated icon and transparent mascot to `site/assets/`; create optimized display sizes and a reproducible animated pipeline overview with a static poster. Record generation prompts in `site/README.md`.
2. Build `site/index.html`, `site/styles.css`, `site/main.js`: hero, interactive illustrative capability examples, process GIF, architecture and quick start. All navigation and CTAs lead to existing sections or repository resources. Copy buttons provide visible success/failure feedback.
3. Add a scoped GitHub Pages workflow in `.github/workflows/pages.yml`. Deploy only the static site, from default `main` on relevant pushes or manual dispatch. Document how to publish the current `home` branch via manual dispatch or merge to main. No automatic git commit/push or repository settings changes during implementation.
4. Add a branded README introduction, animated overview, key capabilities and homepage link. Preserve existing installation and architecture documentation.
5. Verify local assets/anchors, JavaScript syntax, desktop/mobile layout, tabs and keyboard operation, clipboard behavior, reduced motion, GIF frame count and size. Inspect screenshots and fix observed issues. Report publication state honestly.

## Verification commands

```powershell
python scripts/check_homepage.py
node --check site/main.js
python -m http.server 4173 --directory site --bind 127.0.0.1
```

Browser review uses a desktop viewport and narrow mobile viewport, including no-JavaScript and reduced-motion states. No backend services are required for the homepage.

## Verification results

- Local HTTP response: 200. HTML/JavaScript/README asset references and section anchors pass `scripts/check_homepage.py`; JavaScript syntax and whitespace checks pass.
- Browser widths 320, 390, 768, 1024 and 1440px show no horizontal document overflow. Fixed decorative orbit overflow found during the first mobile review.
- Scenario click switching and arrow-key selection update the question, answer, sources and ARIA state. Pause toggles to the static poster. Copy action returns its success feedback; the browser automation clipboard bridge did not expose the copied string, so cross-bridge clipboard contents were not independently verified.
- Browser console reported no warnings or errors during interaction checks. Default HTML content and reduced-motion CSS/picture fallback were inspected in source; disabled-JavaScript and OS motion emulation were not available in this preview interface.
- GIF confirmed at 1000×430, 64 encoded frames, 6.4-second loop, approximately 467 KiB. Mascot PNG alpha ranges from 0 to 255. Replaced a font-dependent check glyph with a drawn checkmark in the animation.
- Updated the Vue application's browser favicon to the same owl icon. `npm run build` succeeds, with the existing KnowledgeView bundle-size warning.
- Desktop and mobile review screenshots are local development outputs in `tmp/homepage-preview/`, outside the Pages publication directory.
- Initial implementation was prepared locally without publishing. In a follow-up, the user authorized committing only homepage/branding/README files, pushing to GitHub, and enabling GitHub Pages. The workflow now supports the current `home` branch as well as `main`; unrelated business-code changes are excluded.
