# PDF Assistant

**AI-Powered PDF Question Answering System** — a BCA final-year project.

Upload a PDF and ask questions about it in plain English. The application finds
the passages in your document that relate to your question, sends only those
passages to a language model, and shows the answer together with the page
numbers it came from — so you can check it yourself.

This is a **Retrieval-Augmented Generation (RAG)** application. Everything runs
on your own machine except the calls it makes to the Groq API for generating
text.

---

## Quick start

Run the setup script for your system from inside this folder
(`pdf-assistant`). It installs everything.

**Ubuntu / Linux / macOS**

```bash
bash setup.sh
```

**Windows** (PowerShell)

```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1
```

The script does three things: installs `uv` if it is missing, installs Python
and all 82 packages, and creates your `.env` file. It takes a few minutes the
first time (about 200 MB to download) and is safe to run again — it never
overwrites an existing `.env`, so it cannot destroy your API key.

Then **two manual steps** the script cannot do for you:

**1. Add your Groq API key.** Get a free one at
<https://console.groq.com/keys>, open `.env`, and change:

```env
GROQ_API_KEY=your_groq_api_key
```

to your real key. Copying the file is not enough — the key must be pasted in,
or the app will say *"Groq API key: missing"* and cannot answer questions.

**2. Start the app.**

```bash
uv run streamlit run app.py
```

It opens at <http://localhost:8501>.

> **Why is `uv` installed first?** It is the only thing you install by hand.
> Everything else — including Python itself, if the machine does not have it —
> is installed by `uv` from `uv.lock`. If the script says `uv` is not on PATH,
> close the terminal, open a new one, and run the script again.

The next sections explain each piece and what to do when something goes wrong.

---

## All commands

| What you want | Command |
|---|---|
| Install / update dependencies | `uv sync` |
| **Run the app** | `uv run streamlit run app.py` |
| Run the app on a different port | `uv run streamlit run app.py --server.port 8502` |
| Run all tests | `uv run pytest` |
| Run one test file | `uv run pytest tests/test_chat_flow.py -v` |
| Check code style | `uv run ruff check .` |
| Auto-fix formatting | `uv run ruff format .` |
| Check formatting without changing files | `uv run ruff format --check .` |
| Verify the RAG engine end to end (**uses API credits**) | `uv run python scripts/check_rag_pipeline.py` |
| Add a dependency | `uv add <package>` |
| Install on a slow or unreliable network | `UV_HTTP_TIMEOUT=1800 UV_CONCURRENT_DOWNLOADS=1 uv sync` |

Every command starts with `uv run` because that runs it inside this project's
own virtual environment. You never need to "activate" anything.

---

## What you need

| You install by hand | Everything else |
|---|---|
| **`uv`** — one command, once per machine | Installed automatically by `uv sync`: **Python 3.12** (if missing), and all **82 packages** |
| **A Groq API key** — free, from <https://console.groq.com/keys> | The **88 MB embedding model**, downloaded on your first PDF upload |

Budget about **2 GB of free disk** and an internet connection for the first
install.

Nothing else is required. No Git, no Docker, no database, no GPU, no paid API.

### If you prefer to run the steps yourself

The setup script just runs these in order. The order matters — `uv sync` cannot
work before `uv` exists.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh    # 1. Linux / macOS
uv sync                                            # 2. Python + 82 packages
cp .env.example .env                               # 3. create config
#    then open .env and paste your Groq key
uv run streamlit run app.py                        # 4. run
```

On Windows, step 1 is `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`
and step 3 is `Copy-Item .env.example .env`.

> **If the download keeps timing out,** make uv fetch one package at a time and
> wait longer before giving up:
>
> ```bash
> UV_HTTP_TIMEOUT=1800 UV_CONCURRENT_DOWNLOADS=1 uv sync
> ```
>
> uv does not resume a part-finished download, so many parallel downloads on a
> slow link keep restarting the large files. Completed packages are cached, so
> re-running `uv sync` after a failure is always safe. The setup script falls
> back to this automatically if the first attempt fails.

**Your `.env` stays on your machine.** Never share it, never put it in your
report, and never paste your key into a chat or an email.

**The first PDF you upload takes a few minutes** — the embedding model
downloads once and is then cached forever. The progress spinner says so. Later
uploads take seconds.

---

## Using the app

Upload a PDF in the sidebar, then work through the four sections.

| Section | What it does |
|---|---|
| **Overview** | File name, size, pages, character and chunk counts, processing status, and a **Generate summary** button that produces a summary plus the important topics and key points |
| **Ask PDF** | Chat. Ask a question, get an answer with a **Sources** panel showing the page numbers and the exact excerpts the answer came from. Supports follow-up questions and has a Clear chat button |
| **Understand Document** | Eight one-click explanations: what it is about, its purpose, main topics, important sections, key points, difficult terms in simple English, main conclusions, and how to present it to someone else |
| **Document Details** | Everything technical: filename, size, pages, characters, chunks, embedding model, language model, vector index status with its vector count, retrieval settings, a per-page extracted-text viewer, and a delete control |

Two deliberate behaviours worth knowing:

- **Summaries and explanations only run when you click.** They are not generated
  automatically on upload, because each one costs API credits and you may only
  want to browse the document. Once generated they are cached, so looking at
  them again is instant and free.
- **When the document does not contain the answer**, the app replies
  *"I could not find sufficient information about this topic in the uploaded
  PDF."* and does not call the model at all. It would rather say nothing than
  guess.

---

## Configuration

All settings live in `.env`. See `.env.example` for the documented template.

| Variable | Default | What it does |
|---|---|---|
| `GROQ_API_KEY` | — | Your key. Required for summaries, explanations and answers |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | The Groq chat model |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Runs locally; costs nothing |
| `CHUNK_SIZE` | `800` | Characters per chunk |
| `CHUNK_OVERLAP` | `150` | Characters repeated between neighbouring chunks |
| `TOP_K` | `4` | How many chunks are retrieved per question |
| `MIN_SIMILARITY` | `0.25` | Score floor. Below this the app abstains instead of guessing |
| `MAX_UPLOAD_SIZE_MB` | `20` | Largest PDF accepted |

**Tuning tips.** Raise `TOP_K` if answers miss information that is in the
document. Raise `MIN_SIMILARITY` if the app answers questions it should refuse;
lower it if it refuses questions it should answer.

---

## Technology

| Component | Technology |
|---|---|
| Language | Python 3.12 |
| Environment & dependencies | `uv` |
| User interface | Streamlit |
| PDF text extraction | PyMuPDF |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`), run locally |
| Vector search | FAISS (`IndexFlatIP`) |
| Answer generation | Groq API |
| Configuration | python-dotenv |
| Testing | pytest |
| Code quality | Ruff |

---

## How the RAG pipeline works

**Indexing — runs once per upload**

```text
PDF upload
  -> validation (type, size, password, extractable text)
  -> text extraction with PyMuPDF, page by page
  -> text cleaning
  -> chunking into overlapping pieces, each tagged with its page number
  -> local Sentence Transformer embeddings (L2-normalised)
  -> FAISS index (held in memory)
```

**Answering — runs per question**

```text
your question
  -> embedded with the same model used for the chunks
  -> FAISS similarity search
  -> top-k chunks retrieved, anything below MIN_SIMILARITY discarded
  -> if nothing is left: abstain, with no API call
  -> otherwise question + chunks sent to Groq
  -> answer displayed with page numbers and excerpts built from chunk metadata
```

### Five design decisions you should be able to explain

1. **Why chunk at all?** A whole PDF will not fit in the model's context window,
   and sending it all would be slow and expensive. Retrieval sends only the few
   passages that matter.
2. **Why do chunks overlap?** A chunk boundary is an arbitrary cut. A sentence
   sitting across it would be torn in half and neither half would resemble the
   question closely enough to be retrieved. Overlap means every sentence appears
   whole in at least one chunk.
3. **Why do chunks never cross a page boundary?** So every chunk has exactly one
   correct page number. A chunk spanning pages 4 and 5 could not honestly answer
   "which page is this from?", so its citation would be a guess.
4. **Why normalise the vectors?** For unit vectors, inner product equals cosine
   similarity — which is what lets FAISS's `IndexFlatIP` rank by meaning.
5. **Why are citations built in code?** The page numbers come from metadata
   stored with each chunk. The language model is never shown a page number and
   never asked for one, so it cannot invent a citation.

---

## Project structure

```text
pdf-assistant/
├── app.py                      # Streamlit entry point: upload pipeline and wiring
├── setup.sh                    # one-command setup for Linux / macOS
├── setup.ps1                   # one-command setup for Windows
├── pyproject.toml              # dependencies and tool configuration
├── uv.lock                     # exact locked dependency versions
├── .env.example                # configuration template (no real secrets)
├── .env                        # YOUR config, with your key - never share this
├── .python-version             # pins the interpreter so installs are reproducible
├── src/pdf_assistant/
│   ├── config.py               # the only module that reads environment variables
│   ├── models/schemas.py       # dataclasses passed between layers
│   ├── services/               # business logic, testable without Streamlit
│   │   ├── pdf_service.py          # validation and page-wise extraction
│   │   ├── chunking_service.py     # overlapping chunks with page metadata
│   │   ├── embedding_service.py    # local embeddings
│   │   ├── vector_store_service.py # FAISS index and search
│   │   ├── retrieval_service.py    # question -> relevant chunks
│   │   ├── llm_service.py          # Groq calls and error handling
│   │   └── summary_service.py      # summaries and explanations (map-reduce)
│   ├── prompts/                # the prompts that keep answers grounded
│   │   ├── qa_prompt.py
│   │   ├── summary_prompt.py
│   │   └── explanation_prompt.py
│   ├── ui/                     # Streamlit sections, one module each
│   │   ├── state.py                # session state; clears everything on delete
│   │   ├── errors.py               # the error tuple every UI call catches
│   │   ├── overview.py
│   │   ├── chat.py
│   │   ├── understand.py
│   │   └── document_details.py
│   └── utils/
│       ├── file_validation.py  # all upload checks
│       └── text_utils.py       # cleaning and formatting helpers
├── scripts/
│   └── check_rag_pipeline.py   # live end-to-end check (uses API credits)
├── data/                       # local working folder, not shared
├── docs/                       # project documentation
└── tests/                      # 274 tests
```

The layering rule: `ui/` modules read and write Streamlit state and draw
widgets; `services/`, `prompts/` and `utils/` are plain Python with no Streamlit
import, so they can be unit-tested directly.

---

## Testing

```bash
uv run pytest                  # all 274 tests
uv run pytest -v               # with each test name
uv run pytest tests/test_chat_flow.py -v
```

**No test costs money.** Groq is mocked everywhere and the embedding model is
never downloaded during testing, so the suite runs offline in about 7 seconds.
Test PDFs are built in memory with PyMuPDF, so there are no sample files to
manage.

The tests worth knowing about:

| File | What it protects |
|---|---|
| `test_pipeline_integration.py` | Page numbers survive extraction **through** chunking and retrieval — the guarantee every citation rests on |
| `test_chat_flow.py` | The answer/abstain decision, citations, and every error path |
| `test_state.py` | Deleting or replacing a PDF leaves nothing from the previous one |
| `test_app_smoke.py` | The real app renders, and a configured API key never appears on the page |
| `test_file_validation.py` | Every bad upload is rejected with a readable message |

### The live check

```bash
uv run python scripts/check_rag_pipeline.py
```

This one **does** use API credits (about 10 calls). It builds a test PDF, runs
the whole pipeline against the real Groq API, and verifies five things: answers
cite the correct pages, an unanswerable question gets the exact abstention
sentence, the model itself abstains when given irrelevant evidence, a prompt
injection hidden in the PDF is ignored, and the model never invents a page
number. It exits non-zero if any check fails.

---

## Troubleshooting

| Problem | What to do |
|---|---|
| `uv sync` times out | `UV_HTTP_TIMEOUT=1800 UV_CONCURRENT_DOWNLOADS=1 uv sync` — safe to re-run, downloads are cached |
| **"model not found"** from Groq | Groq retires models. Check <https://console.groq.com/docs/models> and update `GROQ_MODEL` in `.env` |
| "Groq API key: missing" in the sidebar | You have no `.env`, or the key line is still the placeholder. Copy `.env.example` to `.env` and paste your real key |
| "Rate limit reached" | The free tier has limits. Wait a minute and retry, or switch `GROQ_MODEL` to `openai/gpt-oss-20b` |
| First upload seems stuck | It is downloading the 90 MB embedding model. Only happens once |
| Scanned PDF rejected | Expected. There is no OCR in this version, so a PDF of page images has no text to read |
| `Port 8501 is already in use` | `uv run streamlit run app.py --server.port 8502` |
| An answer looks wrong | Open **Document Details → View extracted text**. Bad answers are usually bad extraction, not a bad model |

---

## Moving the project to another computer

**Do not copy the whole folder.** Two directories in it are large and machine-specific:
`.venv` (about 1.7 GB of installed packages, built for this exact OS and Python
build) and `.ruff_cache` / `.pytest_cache`.

Copy the source and let `uv` rebuild the environment on the new machine:

```bash
# on the old machine, from the pdf-assistant folder
cd ..
zip -r pdf-assistant.zip pdf-assistant \
    -x "pdf-assistant/.venv/*" \
    -x "pdf-assistant/.pytest_cache/*" \
    -x "pdf-assistant/.ruff_cache/*" \
    -x "*__pycache__*" \
    -x "pdf-assistant/.superpowers/*"
```

Then on the new machine:

```bash
unzip pdf-assistant.zip
cd pdf-assistant
bash setup.sh                    # or: powershell -ExecutionPolicy Bypass -File setup.ps1
#   then add their own Groq key to .env
uv run pytest                    # confirm: 274 passed
uv run streamlit run app.py
```

The setup script rebuilds `.venv` from `uv.lock`, so the new machine gets the
identical package versions rather than whatever is newest that day.

`uv.lock` is what makes this reliable: it records the exact version of every
package, so the new machine gets an identical environment rather than whatever
is newest that day.

**About `.env`:** the zip above includes it, which is convenient for your own
second machine but means your API key travels inside the file. If you are
sending the project to anyone else — a classmate, a teacher, an evaluator — add
`-x "pdf-assistant/.env"` to the zip command and let them add their own key.

The embedding model downloads again on the new machine (about 90 MB, once).

---

## Limitations

- **One PDF at a time.** Uploading a new document replaces the previous one.
- **No OCR.** Scanned PDFs made of page images are rejected with an explanation,
  because there is no text in them to extract.
- **Nothing persists between runs.** The vector index lives in memory, so
  restarting the app means re-uploading the document.
- **A chunk never crosses a page boundary,** so a fact split across a page break
  may retrieve less well. This is a deliberate trade for honest citations.
- **An answer can be wrong even with a correct citation.** The citation proves
  the passage exists in your document; it does not prove the model read it
  correctly. Always open the excerpt.
- **Groq free-tier rate limits apply** and may change without notice.
- Answer quality depends on the model set in `GROQ_MODEL`.

---

## Build status

| Phase | Scope | Status |
|---|---|---|
| 1 | Project setup, `uv` environment, structure, config, upload validation, UI shell | **Complete** |
| 2 | Page-wise text extraction, cleaning, chunking with page metadata | **Complete** |
| 3 | Local embeddings, FAISS index, retrieval, Groq integration, citations | **Complete** |
| 4 | Full UI integration of all four sections | **Complete** |
| 5 | Evaluation with 20–30 known-answer questions, error-path testing | Not started |
| 6 | Complete documentation in `docs/` and viva preparation | Not started |

The application is feature-complete and usable. Phase 5 measures how well it
actually answers questions on a real document; Phase 6 writes the submission
documents.

---

## Notes

This project is **local-only**: no Git repository, no cloud deployment, no
Docker. The full specification and the original project proposal are in the
parent folder.
