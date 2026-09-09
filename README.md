# fuzzy-funicular

Layout-faithful text extraction for scholarly and difficult documents, plus a private,
Apple-native web library (a paperless-ngx idea rebuilt around iCloud Drive, iPhone, iPad and
macOS instead of Linux). Everything runs on the Mac; nothing is scheduled or hosted elsewhere.

## What it does

| Input | Engines | Outputs |
|---|---|---|
| PDF with a text layer | poppler `pdftotext -layout` → physical layout text; pymupdf4llm + pymupdf-layout → reading-order Markdown (columns, headings, tables); PyMuPDF → plain text; pypdfium2 → independent character-count cross-check; Ghostscript → repair when a file will not open | `.layout.txt`, `.md`, `.txt`, `.json` |
| Scanned PDF, phone photo, HEIC | scan/photo detector (per page: text chars, image coverage, unmappable glyphs). OCR is **off by default**; with `--ocr auto` only flagged pages are OCR'd. Backend on macOS is Apple Vision (`ocrmac`), then a macOCR-style CLI, then tesseract | as above, plus `.ocr.pdf` (searchable copy) |
| DOCX / PPTX / XLSX / HTML / EPUB / web URL | docling | `.md`, `.txt`, `.json` |
| Audio (wav, mp3, m4a, …) / video | docling ASR (Whisper; MLX build auto-selected on Apple Silicon) | `.md`, `.txt`, `.json` |

The OCR path writes recognised text *into a copy of the PDF as a real text layer*, then runs
every engine on that copy. That is why an OCR'd scan comes out with the same `pdftotext -layout`
fidelity as a born-digital PDF, and why a searchable PDF is a by-product.

## Install (macOS, Apple Silicon)

```sh
brew bundle                 # uv, poppler, ghostscript, ffmpeg, tesseract (Brewfile)
uv python install 3.14      # uv-managed CPython; the project refuses any other interpreter
uv sync --all-extras        # core + Apple Vision OCR + docling/ASR + web
uv run funicular doctor     # every engine, version and what is missing
```

`pyproject.toml` pins `python-preference = "only-managed"`, `.python-version` = 3.14, and
resolves only for macOS arm64 and Linux x86_64. `uv.lock` is committed; on Linux, torch comes
from the CPU wheel index so a validation box never downloads CUDA. On macOS the PyPI wheel is
the Metal (MPS) build.

Extras: `ocr` (Apple Vision), `docling` (Office, HTML, audio, video), `web` (FastAPI app).
`uv sync` alone gives the PDF stack.

## Command line

```sh
uv run funicular extract paper.pdf scan.pdf notes.docx talk.m4a https://example.org/post -o out
uv run funicular extract scan.pdf --ocr auto                # OCR only pages that look scanned
uv run funicular extract photo.heic --ocr auto --ocr-lang en-US,de-DE
uv run funicular extract book.pdf --ocr force               # OCR every page (figures too)
uv run funicular detect paper.pdf                           # per-page scan / garble signals
uv run funicular doctor
```

Each run writes a `.json` report: engines used, per-page signals, warnings (for example
"2 pages look like scans and OCR is off", or "layout text has 900 chars but pdfium sees
3,100; an engine may have dropped content").

## Web app

A single-user (well, allowlisted-users) library with per-file progress bars, dark mode, a
native iOS look, full-text search, tags, and an iCloud Drive inbox/archive.

### 1. GitHub App

Create a GitHub App (Settings → Developer settings → GitHub Apps → New):

* Callback URL: `<FUNICULAR_PUBLIC_URL>/auth/callback`
* "Request user authorization (OAuth) during installation": on
* "Expire user authorization tokens": on (we discard the token immediately anyway)
* Permissions: none needed beyond the default account read
* Webhook: off

Copy the Client ID and generate a client secret. Find each allowed user's numeric ID:
`curl -s https://api.github.com/users/<login> | jq .id`. Logins can be renamed and reassigned;
numeric IDs cannot, which is why the allowlist is IDs.

### 2. Configure

```sh
cp .env.example .env    # fill in the four REQUIRED values
uv run funicular serve  # http://127.0.0.1:8787
```

### 3. Reach it from iPhone and iPad

The app binds to loopback and refuses to start on a public interface without a TLS proxy in
front of it. Two good options on the same Mac:

* **Tailscale Serve** (recommended): `tailscale serve --bg 8787`, then set
  `FUNICULAR_PUBLIC_URL=https://<mac>.<tailnet>.ts.net` and `FUNICULAR_TRUST_PROXY=1`. Only
  devices on the tailnet can reach it; GitHub sign-in is the second lock.
* **Caddy** with a real hostname: see `deploy/Caddyfile.example`.

Add the page to the Home Screen on iOS; it runs standalone with the system dark/light theme
(or force one from the footer).

### 4. Keep it running (local launchd, no cloud)

```sh
deploy/install-launchd.sh      # per-user LaunchAgent: uv run --frozen funicular serve --watch
tail -f ~/Library/Logs/funicular.log
deploy/uninstall-launchd.sh
```

`--watch` imports anything dropped into `iCloud Drive/Funicular/Inbox` (from any Apple
device, via the Files app or a Shortcut) and writes results to `iCloud Drive/Funicular/Archive/
<year>/<title>/` as plain files: layout text, Markdown, plain text, the original, and the
searchable PDF when OCR ran. iCloud placeholders (`.name.icloud`) are fetched with `brctl`.

### Security posture

* GitHub App OAuth with single-use `state`, numeric-ID allowlist, no token retention.
* Server-side sessions (hashed random IDs in SQLite), `HttpOnly`, `SameSite=Lax`,
  `__Host-` + `Secure` on https, idle and absolute expiry, allowlist re-checked per request.
* CSRF: per-session token on every POST (form field or `X-CSRF-Token`) plus Origin check.
* Headers: strict CSP (`default-src 'none'`, no inline script/style), HSTS on https,
  `frame-ancestors 'none'`, `nosniff`, `no-referrer`, `Cache-Control: no-store`.
* Uploads: magic-byte sniffing (a renamed executable is rejected), size cap enforced before
  and during the read, per-user rate limit, files stored under random IDs, uploaded HTML is
  always served as `text/plain` attachments, path traversal blocked on downloads.
* External binaries (poppler, gs, tesseract) are invoked with argv lists and timeouts;
  Ghostscript runs with `-dSAFER`.
* Rate limits on sign-in and callback; an audit table records sign-ins, denials, uploads,
  deletes and re-runs.
* Nothing leaves the Mac except the two GitHub calls during sign-in and model downloads on
  first use of docling/Whisper.

### Accessibility

Targets are 52 px or larger, nothing needs dragging or two hands (drag-and-drop is an
optional extra to the file button), there are no timed prompts or auto-dismissing toasts,
every action is a real form that works without JavaScript, progress is announced through
`aria-live` regions, focus rings are visible, and the theme follows the system by default
with a persistent manual override. Verified in this repo with Playwright at iPhone 14 Pro and
iPad mini viewports in light and dark mode, keyboard tab order, and a scan for targets under
44 px. Not yet verified with a screen reader or switch access on a real device.

## Tests

```sh
uv run pytest                        # local only; no GitHub Actions in this repo
FUNICULAR_TEST_ASR=1 uv run pytest   # also the Whisper smoke test (downloads ~72 MB once)
```

Fixtures are generated at test time with PyMuPDF (`tests/fixtures/make_fixtures.py`): a
two-column scholarly paper with an embedded Unicode font, footnotes and a table; an image-only
scan of it; a mixed file; a skewed JPEG "phone photo"; a truncated PDF; an HTML article; a
DOCX; and, when `espeak-ng` is installed, a synthetic speech WAV. Tests that need poppler,
ghostscript, tesseract or docling skip cleanly when those are absent.

## Layout

```
src/funicular/
  cli.py            funicular extract | detect | doctor | serve
  config.py         ExtractSettings / Settings (FUNICULAR_* env, .env)
  sniff.py          magic-byte type detection
  detect.py         scan / photo / garble detector
  pdf.py            PDF orchestration (poppler, pymupdf4llm, PyMuPDF, pypdfium2, gs, OCR)
  docling_backend.py
  pipeline.py       router + output writer
  ocr/              backends (Apple Vision, macOCR CLI, tesseract) + layout reconstruction
  tools/            poppler.py, ghostscript.py, binaries.py
  web/              FastAPI app, auth, store (SQLite + FTS5), jobs, importer, inbox watcher
deploy/             launchd plist + install scripts, Caddyfile example
tests/              pytest suite; fixtures generated on the fly
```
