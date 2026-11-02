# Phase 6 — Documentation and Viva Preparation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce the ten documents in `docs/` that the college submission needs, plus a viva question bank, all consistent with the code that actually exists and with the reference proposal.

**Architecture:** Ten focused markdown files, each with one job, written for a reader who has never seen the project — an examiner. Content is drawn from the real codebase and the measured Phase 5 numbers, never invented. `README.md` stays the developer entry point; `docs/` is the submission.

**Tech Stack:** Markdown. No code changes except fixing anything the documentation pass reveals as wrong.

**Spec:** `/home/kalpesh_nts/Project/CH_P/prompt.md` (section 12, and the folder listing in section 7) and the reference proposal `.docx` §6 (submission checklist)

**Depends on:** Phase 5 complete — several documents cite its measured numbers.

## Global Constraints

- **No Git.** This plan replaces commit steps with verification checkpoints.
- **Only create or modify files inside `/home/kalpesh_nts/Project/CH_P/`.**
- **Simple English throughout.** The audience is an examiner, not a senior engineer. Short sentences, no unexplained jargon; define every term on first use.
- **Consistent with the reference `.docx`.** Where this project deliberately departs from it (single PDF not multi-PDF, in-memory index not SQLite, local embeddings not provider embeddings, `uv` not pip, no Git), say so explicitly and give the reason. Do not pretend the proposal and the build are identical.
- **Never invent a number.** Every figure comes from the Phase 5 results or from running a command. If something was not measured, say it was not measured.
- **No API key, no personal data, and no content from the sample PDF that is not the student's to share.**
- Delete `docs/.gitkeep` once real documents exist.
- Every document must be accurate as of the final code. Verify claims against the source before writing them.

## Review Focus

Five ways project documentation most often misleads an examiner.

1. **Claiming a feature that is not built.** OCR, multi-PDF, persistence, and deployment are all absent. Each must appear in Limitations, not in Features. → Task 3.
2. **Stale commands.** Every command in the setup guide must be run and seen to work on a clean checkout, not copied from memory. → Task 2.
3. **An architecture diagram that does not match the code.** The data-flow must name the real modules and the real function names. → Task 1.
4. **Unmeasured accuracy claims.** "The system answers accurately" with no number behind it. Only Phase 5's measured rates may be stated. → Task 3.
5. **Viva answers the student cannot defend.** An answer that recites a term without explaining it will fail under follow-up questioning. Every answer must be in the student's own vocabulary. → Task 4.

---

### Task 1: Project, requirements and architecture documents

**Files:**
- Create: `docs/project_overview.md` — introduction, problem statement, objectives, scope
- Create: `docs/requirements.md` — functional and non-functional requirements, technology stack
- Create: `docs/architecture.md` — system architecture, component diagram, module responsibilities
- Create: `docs/technical_workflow.md` — the two RAG workflows, step by step
- Create: `docs/folder_structure.md` — annotated tree

**Interfaces:**
- Consumes: the real source tree, `prompt.md` §3–§7, `.docx` §3.1–§3.7.

`docs/architecture.md` must contain an ASCII block diagram naming the actual modules (`pdf_service`, `chunking_service`, `embedding_service`, `vector_store_service`, `retrieval_service`, `llm_service`, `summary_service`, the four `ui/` modules, `state.py`, `config.py`) and state the layering rule: `ui/` may import `services/`; `services/` never imports Streamlit.

`docs/technical_workflow.md` must document both pipelines with the real function names, and must explain the three decisions an examiner will probe:
- why page numbers are converted to 1-based once, in `extract_pages`;
- why chunks never cross a page boundary;
- why citations are assembled in application code and the model is never asked for a page number.

`docs/requirements.md` must separate functional requirements (what the four sections do) from non-functional ones (runs locally, single user, no persistence, free-tier API limits, response time observed in Phase 5).

- [ ] **Step 1: Re-read the source tree**

Run: `find src tests scripts -name '*.py' | sort` and `uv run pytest -q --collect-only | tail -3` to get the real module list and test count. Write nothing from memory.

- [ ] **Step 2: Write the five documents**

- [ ] **Step 3: Check every module named in `architecture.md` exists — REVIEW FOCUS 3**

For each module named, confirm the file exists and the described responsibility matches its docstring. Correct the document, not the code, unless the code is genuinely wrong.

- [ ] **Step 4: Verification checkpoint (no Git)**

Confirm no document claims a file or function that does not exist.

---

### Task 2: Setup and user guide

**Files:**
- Create: `docs/setup_guide.md` — local installation, configuration, how to run
- Create: `docs/user_guide.md` — how to use each of the four sections, with the expected screenshots listed

**Interfaces:**
- Consumes: `README.md`, `.env.example`, the real commands.

`docs/setup_guide.md` must cover: prerequisites (Python 3.12, `uv`, a Groq key); `uv sync`; copying `.env.example` to `.env` and adding the key; `uv run streamlit run app.py`; the slow-network `UV_CONCURRENT_DOWNLOADS=1` fallback; the first-run embedding-model download of about 90 MB; and what to do when Groq reports a model as not found. Give both PowerShell and bash forms of the copy command, since the spec was written for PowerShell.

`docs/user_guide.md` walks each section in order with the exact on-screen labels, and lists the screenshots the report needs: empty state, uploaded document overview, a generated summary, an answered question with its Sources expander open, an abstention reply, all eight Understand buttons, Document Details, and the extracted-text viewer.

- [ ] **Step 1: Verify every command on a clean environment — REVIEW FOCUS 2**

Run in order and record the real output: `uv sync`, `uv run pytest -q`, `uv run ruff check .`, `uv run streamlit run app.py`. Any command that does not work as written is a documentation bug; fix the document.

- [ ] **Step 2: Write both documents using the verified commands**

- [ ] **Step 3: Capture the screenshots listed in the user guide**

Save them under `docs/screenshots/`. These go in the final report.

- [ ] **Step 4: Verification checkpoint (no Git)**

Have someone else (or a fresh terminal with no environment set up) follow `setup_guide.md` literally and reach a running app.

---

### Task 3: Testing, limitations and future enhancements

**Files:**
- Modify: `docs/testing_strategy.md` (started in Phase 5)
- Create: `docs/future_enhancements.md`
- Modify: `docs/project_overview.md` (add the Limitations section)

**Interfaces:**
- Consumes: Phase 5's `docs/evaluation_results.md` and `docs/manual_test_log.md`.

`docs/testing_strategy.md` must state the unit-test approach (what is tested, that Groq is mocked so tests cost nothing, the real test count), the evaluation method, and the three measured rates from Phase 5 — retrieval hit rate, answer correctness, citation accuracy — each with the number of questions behind it.

The Limitations section must list, without softening (REVIEW FOCUS 1 and 4): one PDF at a time; no OCR, so scanned documents are rejected; nothing persists between runs; chunks do not cross page boundaries, so a fact split across a page break may retrieve poorly; an answer can be wrong even with a correct citation; Groq free-tier rate limits apply and may change; answer quality depends on the chosen model; and the measured accuracy figures come from one sample document and do not generalise.

`docs/future_enhancements.md` draws from `.docx` §3.8: multi-PDF collections with per-document attribution, OCR for scanned PDFs, SQLite persistence for metadata and history, an in-app evaluation page, comparing two embedding models or chunk sizes, reranking retrieved chunks, and deployment. Mark each with rough difficulty.

- [ ] **Step 1: Write the three documents from the Phase 5 artefacts**

- [ ] **Step 2: Cross-check every accuracy claim — REVIEW FOCUS 4**

Every number must trace to a row in `docs/evaluation_results.md`. Delete any claim that cannot.

- [ ] **Step 3: Cross-check the feature list against Limitations — REVIEW FOCUS 1**

Confirm nothing appears as a feature anywhere in `docs/` that is actually listed as a limitation.

- [ ] **Step 4: Verification checkpoint (no Git)**

---

### Task 4: Viva preparation

**Files:**
- Create: `docs/viva_preparation.md`

**Interfaces:**
- Consumes: every other document, plus the code.

At least 40 questions with answers, grouped: project basics; RAG and why it is used; PDF processing; chunking; embeddings; vector search; prompting and grounding; citations; error handling; testing; limitations; and code walkthrough.

The questions the examiner is most likely to ask, which must each have a strong answer:
- What is RAG and why not just send the whole PDF to the model?
- What is an embedding? What does a 384-dimensional vector mean?
- What is cosine similarity, and why normalise the vectors before using inner product?
- Why FAISS and not a plain Python loop over all chunks?
- Why do chunks overlap? What breaks without overlap?
- Why do chunks never cross a page boundary?
- How do you know a citation is real and not invented by the model?
- What happens when the document does not contain the answer, and how is that guaranteed?
- What stops instructions hidden inside a PDF from hijacking the model?
- Why local embeddings but a hosted LLM?
- Why `uv` instead of pip?
- Show me where the page number is set and prove it is the reader's page number.
- What are this project's three biggest weaknesses?

Include a short "code walkthrough script": the five files to open, in order, and the one sentence to say about each — for the live demo.

- [ ] **Step 1: Write the question bank**

Answer in the student's own words, in simple English (REVIEW FOCUS 5). Each answer is 2–5 sentences. Where an answer points at code, give the real `file.py:function` reference.

- [ ] **Step 2: Verify every code reference**

Open each referenced file and confirm the function exists and does what the answer says.

- [ ] **Step 3: Rehearse**

Read every question aloud and answer it without looking. Any answer that cannot be given from memory is too complicated — rewrite it simpler.

- [ ] **Step 4: Final verification checkpoint (no Git)**

Run: `uv run pytest -q && uv run ruff check . && uv run ruff format --check .`
Confirm `docs/` contains all ten spec-named documents, delete `docs/.gitkeep`, and mark Phase 6 complete in `CLAUDE.md` and `README.md`.
Confirm no `.git` directory exists anywhere: `find /home/kalpesh_nts/Project/CH_P -name .git | wc -l` must print `0`.

---

## Self-Review

**Spec coverage.** §12's seventeen required topics map as: introduction, problem statement, objectives, scope → `project_overview.md` (Task 1); functional and non-functional requirements, technology stack → `requirements.md` (Task 1); system architecture → `architecture.md`, RAG workflow → `technical_workflow.md`, folder structure → `folder_structure.md` (Task 1); local installation and how to run → `setup_guide.md`, user guide → `user_guide.md` (Task 2); testing approach → `testing_strategy.md`, limitations → `project_overview.md`, future enhancements → `future_enhancements.md` (Task 3); viva questions and answers → `viva_preparation.md` (Task 4). All ten filenames from §7's `docs/` listing are produced.

**Step scan.** Steps are verification-heavy on purpose, because the failure mode for documentation is not absence but inaccuracy. Each task ends with a cross-check against the code or a recorded number rather than a self-assessment.

**Type consistency.** Module and function names referenced here match the Phase 3 and Phase 4 plans and the existing tree. The three measured rates have one definition, set in Phase 5 Task 2, and are cited identically in Task 3.

**Review Focus.** 1→Task 3, 2→Task 2, 3→Task 1, 4→Task 3, 5→Task 4. All five are checks against reality rather than new content, which is what this phase needs.

**Proportion.** Four tasks for ten documents, grouped by what gets verified together. Short relative to its output, because the content is prose the writer must compose from the finished system, not decisions a plan can make in advance.
