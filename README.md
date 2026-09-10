# fuzzy-funicular

Layout-faithful text extraction for scholarly and difficult documents, plus a private,
Apple-native web library: the paperless-ngx idea rebuilt around iCloud Drive, iPhone, iPad and
macOS. Everything runs on the Mac, tests run locally, nothing is scheduled or hosted elsewhere.

## What it does

| Input | Engines | Outputs |
|---|---|---|
| PDF with a text layer | poppler `pdftotext -layout` (physical layout); pymupdf4llm + pymupdf-layout (reading-order Markdown with headings and tables); PyMuPDF plain text; pypdfium2 cross-check; Ghostscript repair for files that will not open | `.layout.txt`, `.md`, `.txt`, `.json` |
| Scanned PDF, phone photo, HEIC | per-page scan/garble detector. OCR is off unless you ask; Auto mode OCRs only flagged pages. Backend on macOS: Apple Vision (`ocrmac` or pyobjc directly), then a macOCR-style CLI, then tesseract | as above, plus `.ocr.pdf` (searchable copy) |
| DOCX / PPTX / XLSX / HTML / EPUB / web URL | docling (MPS on Apple Silicon) | `.md`, `.txt`, `.json` |
| Audio / video | docling ASR (Whisper; MLX build on Apple Silicon) | `.md`, `.txt`, `.json` |
| zip / tar / tar.gz / tgz, nested folders | safe extraction with bomb limits, then every compatible file above | one document per file, tagged by archive and folder |

After extraction every document is indexed for search, auto-tagged, and (PDFs) looked up
online for a verified bibliographic record. Optional: evidence-card summaries and grounded
Q&A through Claude, OpenAI-compatible endpoints or a local model.

## Install (macOS, Apple Silicon)

```sh
brew bundle                 # uv, poppler, ghostscript, ffmpeg, tesseract (Brewfile)
uv python install 3.14      # uv-managed CPython; the project refuses any other interpreter
uv sync --all-extras        # core + Apple Vision OCR + docling/ASR + web + search + scholar + mcp + llm
uv run funicular doctor     # every engine, model, and what is missing
uv run funicular models fetch   # docling layout/table models, Whisper, embeddings (once, needs network)
```

`pyproject.toml` pins `python-preference = "only-managed"`, `.python-version` = 3.14, and
resolves only for macOS arm64 and Linux x86_64. `uv.lock` is committed; Linux pulls CPU torch,
macOS gets the Metal build from PyPI.

Extras: `ocr` (Apple Vision), `docling`, `web`, `search` (model2vec embeddings without torch),
`scholar` (feeds), `mcp`, `llm` (Anthropic SDK). `uv sync` alone gives the PDF stack.

### Upgrading and inspecting

```sh
uv run funicular deps                 # locked dependency tree
uv run funicular deps --outdated      # what has newer releases
uv lock --upgrade && uv sync --all-extras && uv run pytest   # then commit pyproject.toml + uv.lock
```

## Command line

```sh
uv run funicular extract paper.pdf scan.pdf notes.docx talk.m4a https://example.org/post -o out
uv run funicular extract scan.pdf --ocr auto           # OCR only pages that look scanned
uv run funicular estimate *.pdf --ocr auto             # time / memory before doing the work
uv run funicular detect paper.pdf                      # per-page scan / garble signals
uv run funicular compress big.pdf --strength 70 --preview   # dry run; drop --preview to write
uv run funicular scholar meta paper.pdf                # DOI / record verified against the first page
uv run funicular scholar refs paper.pdf --resolve      # reference list, each entry resolved to a DOI
uv run funicular scholar rename *.pdf --apply          # Author - Year - Title (dry run without --apply)
uv run funicular pdf analyze form.pdf                  # forms, signatures, pagination, headers, outline
uv run funicular pdf pdfa paper.pdf                    # PDF/A-2b via Ghostscript (reports reverts honestly)
uv run funicular pdf fill form.pdf -o out.pdf --set name="B. Shore" --flatten
uv run funicular summarize paper.pdf                   # evidence card (needs an LLM provider)
uv run funicular ask "absolute effect of riluzole?"    # grounded answer from the library index
uv run funicular feeds add "cat:q-bio.NC AND all:motor neuron" --kind arxiv --auto-stage
uv run funicular zotero import --collection <key>
uv run funicular mcp                                   # MCP server on stdio for Claude Desktop / ChatGPT
uv run funicular serve                                 # the web app
uv run funicular stop                                  # clean shutdown (jobs cancelled, helpers killed)
```

## Web app

### Pages

* **Library** — upload (files, folders through the browser, zip/tar archives), per-file
  progress bars over SSE, selection + tar.gz export, batch imports, tags.
* **Folders** — browse Home, iCloud Drive and volumes on the Mac; import a file or a whole tree.
  Cloud-only iCloud files are downloaded first (`brctl`), optionally evicted after archiving.
* **Document** — layout / Markdown / plain views, evidence card, scholarly record (DOI,
  citations, Scite tallies, open-access link), references, PDF tools (unlock, PDF/A, forms,
  signatures, header/footer stripping, heading outline), compression with a dry-run preview,
  re-run with an estimate, cancel.
* **Search** — normalised full text, trigram jargon matching, phonetic expansion for dictated
  or misspelt terms, semantic vectors; results carry page numbers and snippets.
* **Ask** — grounded answers with `[doc p.N]` citations; buttons for Claude, ChatGPT, GitHub
  Models, Ollama, LM Studio depending on what is configured.
* **Feeds** — RSS/Atom, arXiv and PubMed subscriptions; stage new articles (open-access PDF
  fetched, extracted, indexed, record verified, summary written when an LLM is configured).
* **Zotero** — import PDFs from a collection through Zotero 7's local API.
* **Graph** — citation neighbourhood of your verified papers (about 40 nodes), similarity
  edges, keyboard/button pan-zoom, list fallback.
* **System** — CPU, memory, disk, acceleration, running jobs with cancel, clean stop.

### 1. GitHub App sign-in

Create a GitHub App (Settings → Developer settings → GitHub Apps → New):

* Callback URL: `<FUNICULAR_PUBLIC_URL>/auth/callback`
* "Request user authorization (OAuth) during installation": on
* "Expire user authorization tokens": on (the token is discarded immediately anyway)
* Permissions: none beyond the default; Webhook: off

Copy the Client ID and generate a client secret. Allowed users are numeric IDs:
`curl -s https://api.github.com/users/<login> | jq .id`.

### 2. Configure

```sh
cp .env.example .env    # fill in the four REQUIRED values
uv run funicular serve  # http://127.0.0.1:8787
```

### 3. Reach it from iPhone and iPad

The app binds to loopback and refuses to start on a public interface without a TLS proxy.

* **Tailscale Serve** (recommended): `tailscale serve --bg 8787`, then
  `FUNICULAR_PUBLIC_URL=https://<mac>.<tailnet>.ts.net` and `FUNICULAR_TRUST_PROXY=1`.
* **Cloudflare Tunnel or Caddy** later: see `deploy/Caddyfile.example`; keep the GitHub
  allowlist as the second lock.

Add to the Home Screen on iOS for a standalone app with system dark/light theme.

### 4. Keep it running (local launchd, no cloud)

```sh
deploy/install-launchd.sh      # per-user LaunchAgent: uv run --frozen funicular serve --watch
deploy/uninstall-launchd.sh
```

`--watch` imports anything dropped into `iCloud Drive/Funicular/Inbox` and writes results to
`iCloud Drive/Funicular/Archive/<year>/<title>/`.

### LLM providers

| Provider | Configure | Default model |
|---|---|---|
| Anthropic | `ANTHROPIC_API_KEY` or `ant auth login` | `claude-opus-5` (adaptive thinking, `FUNICULAR_LLM_EFFORT`, refusal fallbacks on) |
| OpenAI | `OPENAI_API_KEY` (+ `OPENAI_BASE_URL`) | `gpt-4.1` |
| GitHub Models | `GITHUB_TOKEN=ghp_…` | `openai/gpt-4.1` |
| Ollama / LM Studio | `FUNICULAR_LLM_PROVIDER=ollama|lmstudio` | local |

Only the passages needed for an answer (or the chunks of one document for a summary) are
sent; nothing else leaves the Mac.

### MCP for Claude Desktop / ChatGPT

Claude Desktop `claude_desktop_config.json`:

```json
{"mcpServers": {"funicular": {"command": "uv", "args": ["run", "--project", "/path/to/fuzzy-funicular", "funicular", "mcp"]}}}
```

Tools: `search`, `list_documents`, `get_document`, `get_summary`, `get_scholar`,
`get_references`, `ask`. For HTTP clients: `FUNICULAR_MCP_TOKEN=… funicular mcp --http 8788`
(loopback, Bearer token).

### Security posture

* GitHub App OAuth with single-use `state`, numeric-ID allowlist, no token retention.
* Server-side sessions (hashed random IDs), `HttpOnly`, `SameSite=Lax`, `__Host-` + `Secure`
  on https, idle and absolute expiry, allowlist re-checked per request.
* CSRF token on every POST plus Origin check; per-user rate limits on uploads and on every
  expensive operation (LLM, PDF/A, graph, online lookups, feeds).
* Strict CSP (no inline script/style), HSTS on https, `frame-ancestors 'none'`, `nosniff`,
  `no-referrer`, `noindex`, `Cache-Control: no-store`.
* Uploads sniffed by magic bytes; archives extracted with zip-slip, symlink, ratio, size and
  depth limits; uploaded HTML served only as plain-text attachments; folder browsing limited to
  allowed roots with symlink resolution.
* Every external binary runs from an argv list with rlimits (process count, address space on
  Linux, no core dumps) and belongs to a job whose tree the memory guard can kill.
* `/system` shows caps and last kill; `funicular stop` and the web "Stop server" exit cleanly.

### Accessibility

Targets are 52 px or larger, nothing needs dragging or two hands (drag-and-drop and the
compression slider both have button equivalents), no timed prompts, every action is a real
form that works without JavaScript, progress is announced through `aria-live`, the graph has
button/keyboard pan-zoom and a list view, and the theme follows the system with a manual
override. Verified with Playwright at iPhone 14 Pro and iPad mini sizes and by tab order; not
yet verified with a screen reader or switch access on a real device.

## Tests

```sh
uv run pytest                              # local only; no GitHub Actions in this repo
FUNICULAR_TEST_ASR=1 uv run pytest         # + Whisper smoke test
FUNICULAR_TEST_EMBED=1 uv run pytest       # + semantic search with a real embedding model
```

Fixtures are generated at test time; network services are replaced by `httpx.MockTransport`
in tests, so the suite runs offline. Google Scholar is not integrated: it has no API and
blocks automated access.

## Layout

```
src/funicular/
  cli.py            funicular extract|detect|doctor|estimate|compress|serve|stop|deps|mcp|…
  config.py         ExtractSettings / Settings (FUNICULAR_* env, .env)
  pipeline.py       router + output writer          pdf.py         PDF orchestration
  detect.py         scan / garble detector           ocr/           Apple Vision, macOCR CLI, tesseract
  archives.py       zip/tar handling                 browse.py      allowed-root folder browser
  icloud.py         dataless files, brctl            resources.py   monitor, memory guard, cancel, rlimits
  compress.py       PDF compression + preview        estimate.py    work estimator with recorded timings
  search.py         chunk index, phonetic, semantic  textnorm.py    normalisation
  embeddings.py     sentence-transformers / model2vec autotag.py    keyphrases + structural tags
  scholar/          ids, clients, metadata, refs, rename            graph.py    literature graph
  llm.py            providers                        summarize.py   evidence cards, grounded ask
  feeds.py          RSS/arXiv/PubMed + staging       zotero.py      Zotero local API
  pdftools.py       encryption, PDF/A, forms, signatures, pagination, headers, outline
  mcp_server.py     MCP tools                        models.py      model prefetch/status
  web/              FastAPI app, auth, store, jobs, importer, watcher, templates, static
deploy/             launchd plist + scripts, Caddyfile example
tests/              pytest suite; fixtures generated on the fly
```
