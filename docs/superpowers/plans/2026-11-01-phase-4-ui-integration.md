# Phase 4 — UI Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect the Phase 3 RAG services to the four Streamlit sections, so a user can upload a PDF and actually get summaries, grounded answers with page citations, and guided explanations.

**Architecture:** The services stay Streamlit-free; this phase adds only the wiring. The upload pipeline in `app.py` gains two more stages (embed, index) and reaches `READY`. `ui/state.py` remains the single gatekeeper for session state, so the search index and generated text are cleared with the document. Every generated result is cached in session state, because Streamlit re-runs the whole script on each interaction and regenerating a summary on every click would burn API credits.

**Tech Stack:** Streamlit 1.x (`AppTest` for testing), plus the Phase 3 services.

**Spec:** `/home/kalpesh_nts/Project/CH_P/prompt.md` (sections 5, 6, 10, 13 Phase 4, 14) and `/home/kalpesh_nts/Project/CH_P/CLAUDE.md`

**Depends on:** `2026-11-01-phase-3-rag-core.md` must be complete. Do not start until `scripts/check_rag_pipeline.py` has passed.

## Global Constraints

- **No Git.** This plan replaces commit steps with verification checkpoints.
- **Only create or modify files inside `/home/kalpesh_nts/Project/CH_P/`.**
- `uv` only. No new dependencies.
- **Keep UI logic separate from business logic.** `ui/` modules may import from `services/`; `services/` must never import Streamlit. Verify with `grep -rl "import streamlit" src/pdf_assistant/services/` returning nothing.
- **Every new document-derived session key goes in `_DOCUMENT_STATE_DEFAULTS`, and every new document-scoped widget key in `_DOCUMENT_WIDGET_KEYS`** (`ui/state.py`). A widget value outlives its widget and leaks across documents if missed.
- **Citations are rendered from `RetrievedChunk.chunk.page_number`**, never from model output.
- The abstention string is `qa_prompt.INSUFFICIENT_CONTEXT_MESSAGE`; import it, never retype it.
- `GROQ_API_KEY` must never appear in the UI, including in error messages.
- Every API-backed action shows a `st.spinner` while it runs and a readable `st.error` on `LlmError`.
- Line length 100, type hints, docstrings. `uv run ruff check .` and `uv run ruff format --check .` must pass.

## Review Focus

Five failure modes a real user will hit that the happy path does not cover.

1. **Clicking a generate button twice.** The second click must not fire a second API call when a result is already cached. → Task 2.
2. **An API failure must not corrupt the chat.** If answering raises `LlmError`, the user's question must not be left in history with no reply, and the next question must still work. → Task 3.
3. **Replacing a PDF mid-conversation.** The new document must show no summary, no chat history and no index from the old one. → Task 1.
4. **Asking a question before indexing finished / after it failed.** The chat input must be unavailable or explain itself rather than searching an absent index. → Task 3.
5. **A document that indexes but answers nothing.** Every one of the eight Understand options must handle `LlmError` individually, so one failing option does not blank the whole section. → Task 4.

---

### Task 1: Extend the upload pipeline through embedding and indexing

**Files:**
- Modify: `app.py` (`_process_upload`)
- Modify: `src/pdf_assistant/ui/state.py` (index accessors, new state keys)
- Modify: `tests/test_state.py`
- Create: `tests/test_indexing_pipeline.py`

**Interfaces:**
- Consumes: `embedding_service.load_model`, `embed_chunks`; `vector_store_service.build_index`.
- Produces:
  - `state.get_search_index()` / `state.set_search_index(index)` — stored under the existing `vector_index` key.
  - `state.get_embedding_model()` / `state.set_embedding_model(model)` — new key `embedding_model`.
  - Pipeline reaches `ProcessingStatus.READY`, so the existing `state.is_ready()` becomes `True`.

The model is held in session state as well as being `lru_cache`d in the service, so the UI can pass the same instance to `retrieve` without re-resolving it.

After chunking succeeds, `_process_upload` continues: set `EMBEDDING`, load the model and embed the chunks under a spinner (first run downloads ~90 MB — the spinner text must say so, or the user will think it has hung); set `INDEXING`, build the index; set `READY`. Any `EmbeddingError` or `VectorStoreError` sets `FAILED` with the message and leaves the document at text-only, so Document Details still shows the extracted text.

- [ ] **Step 1: Write the failing tests**

In `tests/test_state.py` (dictionary-backed session state, as the existing tests do):
```python
def test_clear_removes_the_search_index()
def test_clear_removes_the_embedding_model()
def test_set_and_get_search_index_round_trip()
```

In `tests/test_indexing_pipeline.py` — test the pipeline as plain functions, not through Streamlit, using a `FakeEmbedder`:
```python
def test_pipeline_builds_an_index_covering_every_chunk()   # index.ntotal == len(chunks)
def test_index_row_order_matches_chunk_id_order()          # REVIEW FOCUS: misalignment = wrong citations
def test_pipeline_on_a_blank_pdf_produces_no_index()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_state.py tests/test_indexing_pipeline.py -v`
Expected: FAIL — `get_search_index` undefined.

- [ ] **Step 3: Add the state accessors, add `embedding_model` to `_DOCUMENT_STATE_DEFAULTS`, then extend `_process_upload`**

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_state.py tests/test_indexing_pipeline.py -v`
Expected: PASS

- [ ] **Step 5: Manual check — REVIEW FOCUS 3**

Run: `uv run streamlit run app.py`. Upload a PDF, wait for `Ready`, ask nothing yet. Upload a *different* PDF. Confirm: Overview shows the new filename and page count, no stale summary, Document Details shows the new text, and Ask PDF has an empty history.

- [ ] **Step 6: Verification checkpoint (no Git)**

Run: `uv run pytest -q && uv run ruff check . && uv run ruff format --check .`

---

### Task 2: Overview — summary, topics and key points

**Files:**
- Modify: `src/pdf_assistant/ui/overview.py`
- Modify: `src/pdf_assistant/ui/state.py` (summary cache accessors)
- Modify: `tests/test_app_smoke.py`

**Interfaces:**
- Consumes: `summary_service.summarise_document`, `state.get_chunks()`.
- Produces: `state.get_summary()` / `state.set_summary(text)` over the existing `summary` key.

Spec §5 Section 1 asks for a summary *and* important topics *and* key points. Generate all three in **one** request by asking `summarise_document` for a response with three labelled markdown sections, rather than making three separate calls — the free tier is rate-limited and three calls per upload is wasteful. Render the returned markdown directly.

Generation is **explicit, not automatic**: show a "Generate summary" button. Auto-generating on upload would spend credits on a document the user may only want to browse. Once generated, the text is cached in session state and the button becomes "Regenerate".

- [ ] **Step 1: Write the failing tests**

Extend `tests/test_app_smoke.py`, seeding `AppTest.session_state` with a ready document and fake chunks, and monkeypatching `summarise_document` with a counting fake:
```python
def test_overview_shows_a_generate_summary_button_when_ready()
def test_overview_renders_a_cached_summary_without_calling_the_model()  # REVIEW FOCUS 1: call count == 0
def test_overview_shows_an_error_when_generation_fails()                # fake raises LlmError
def test_overview_hides_generation_when_no_api_key_is_configured()
def test_overview_shows_character_and_chunk_counts_when_ready()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_app_smoke.py -v`
Expected: FAIL — no summary button rendered.

- [ ] **Step 3: Implement the summary panel in `overview.py` and the state accessors**

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_app_smoke.py -v`
Expected: PASS

- [ ] **Step 5: Verification checkpoint (no Git)**

Run: `uv run pytest -q && uv run ruff check . && uv run ruff format --check .`

---

### Task 3: Ask PDF — grounded answers with citations

**Files:**
- Modify: `src/pdf_assistant/ui/chat.py`
- Modify: `tests/test_app_smoke.py`
- Create: `tests/test_chat_flow.py`

**Interfaces:**
- Consumes: `retrieval_service.retrieve`, `has_sufficient_context`; `llm_service.answer_question`; `qa_prompt.INSUFFICIENT_CONTEXT_MESSAGE`.
- Produces: no new public functions; `_handle_new_question` is replaced.

The new `_handle_new_question` flow:
1. Append the user's `ChatMessage`.
2. Retrieve with `settings.top_k` and `settings.min_similarity`.
3. If `not has_sufficient_context(...)`, append an assistant message whose content is exactly `INSUFFICIENT_CONTEXT_MESSAGE` with `sources=[]` — **do not call Groq**. This saves a request and guarantees the exact wording regardless of what a model might say.
4. Otherwise call `answer_question(...)` under a spinner and append the reply with `sources=retrieved`.
5. On `LlmError`, append an assistant message containing the error text and `sources=[]`, so the conversation stays coherent and the user can retry.

`render_sources` already exists and reads page numbers from chunk metadata — leave it alone.

The chat input must be disabled unless `state.is_ready()`, with a message explaining why (still extracting / indexing failed).

- [ ] **Step 1: Write the failing tests**

`tests/test_chat_flow.py` tests the decision logic as a plain function with fakes, which is far easier than driving `st.chat_input`. Extract the logic as `build_assistant_reply(settings, question, chunks, index, model, history) -> ChatMessage` in `chat.py` and test that:
```python
def test_reply_is_the_exact_abstention_string_when_context_is_weak()
def test_abstention_path_never_calls_the_model()               # REVIEW FOCUS: call count == 0
def test_abstention_reply_has_no_sources()
def test_reply_carries_the_retrieved_chunks_as_sources()
def test_reply_sources_hold_the_original_page_numbers()
def test_llm_error_becomes_a_readable_assistant_message()      # REVIEW FOCUS 2
def test_llm_error_reply_has_no_sources()
def test_api_key_never_appears_in_an_error_reply()
def test_history_is_passed_to_the_model_for_follow_ups()
```

In `tests/test_app_smoke.py`:
```python
def test_chat_input_is_unavailable_until_the_document_is_ready()   # REVIEW FOCUS 4
def test_chat_replays_history_with_its_sources()
def test_clear_chat_empties_the_conversation()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_chat_flow.py -v`
Expected: FAIL — `build_assistant_reply` undefined.

- [ ] **Step 3: Implement `build_assistant_reply` and rewire `_handle_new_question` to use it**

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_chat_flow.py tests/test_app_smoke.py -v`
Expected: PASS

- [ ] **Step 5: Manual check**

Run the app, upload a real PDF, ask three questions whose answers are in it and one that is not. Confirm each answer shows a Sources expander with page numbers, that the page numbers are correct when you open the PDF at that page, and that the absent-answer question returns the exact abstention sentence. Then ask a follow-up using a pronoun ("what about its cost?") and confirm context carries.

- [ ] **Step 6: Verification checkpoint (no Git)**

Run: `uv run pytest -q && uv run ruff check . && uv run ruff format --check .`

---

### Task 4: Understand Document — wire the eight options

**Files:**
- Modify: `src/pdf_assistant/ui/understand.py`
- Modify: `src/pdf_assistant/ui/state.py` (per-option result cache)
- Modify: `tests/test_app_smoke.py`

**Interfaces:**
- Consumes: `summary_service.explain_document`; the existing `UNDERSTAND_OPTIONS` tuple (keys: `about`, `purpose`, `topics`, `sections`, `key_points`, `terms`, `conclusions`, `presentation`).
- Produces:
  - `state.get_explanation(key: str) -> str`
  - `state.set_explanation(key: str, text: str) -> None`

Results are cached per option under a single dict in session state (key `explanations`), so re-selecting an option the user already ran costs nothing. Add `explanations` to `_DOCUMENT_STATE_DEFAULTS` with default `{}` — note `clear_document_state` currently special-cases `list`; extend it to copy `dict` defaults too, or the same dict object would be shared across documents.

Each button calls `explain_document(settings, chunks, option.instruction)` in its own `try`/`except LlmError`, so one failing option leaves the other seven usable. Render the result under the button grid with the option's label as a heading.

- [ ] **Step 1: Write the failing tests**

In `tests/test_state.py`:
```python
def test_clear_removes_cached_explanations()
def test_explanation_cache_round_trip()
def test_explanation_dicts_are_not_shared_between_documents()   # the mutable-default bug
```

In `tests/test_app_smoke.py`:
```python
def test_understand_renders_all_eight_option_buttons()
def test_understand_options_are_disabled_until_ready()
def test_cached_explanation_is_shown_without_calling_the_model()
def test_one_failing_option_does_not_hide_the_others()           # REVIEW FOCUS 5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_state.py tests/test_app_smoke.py -v`
Expected: FAIL — `get_explanation` undefined.

- [ ] **Step 3: Extend `clear_document_state` to handle dict defaults, add the accessors, then wire the buttons**

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_state.py tests/test_app_smoke.py -v`
Expected: PASS

- [ ] **Step 5: Manual check**

Run the app and click all eight options against a real PDF. Confirm each returns something relevant in simple English, and that clicking an already-run option is instant (cached).

- [ ] **Step 6: Verification checkpoint (no Git)**

Run: `uv run pytest -q && uv run ruff check . && uv run ruff format --check .`

---

### Task 5: Document Details — index status and final polish

**Files:**
- Modify: `src/pdf_assistant/ui/document_details.py`
- Modify: `tests/test_app_smoke.py`

**Interfaces:**
- Consumes: `state.get_search_index()`.

The table already has an "Embedding model", "Language model", "Vector index status" and "Processing status" row. Two changes: report the index row as `Built (N vectors)` using `index.ntotal` rather than a bare "Built", and add a "Chunk size / overlap / top-k" row so the retrieval settings in use are visible — spec §5 Section 4 asks for the embedding model and index status, and showing the tuning values makes the Phase 5 evaluation reproducible.

- [ ] **Step 1: Write the failing tests**

```python
def test_details_reports_the_vector_count_when_an_index_exists()
def test_details_reports_not_built_without_an_index()
def test_details_shows_the_retrieval_settings()
def test_details_never_shows_the_api_key()
def test_delete_clears_everything_and_returns_to_the_empty_state()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_app_smoke.py -v`
Expected: FAIL — vector count not rendered.

- [ ] **Step 3: Update the table**

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_app_smoke.py -v`
Expected: PASS

- [ ] **Step 5: Full manual pass over all four sections**

Upload → Overview summary → three questions + one unanswerable → all eight Understand options → Document Details → view extracted text → delete. Confirm the empty state returns everywhere.

- [ ] **Step 6: Verification checkpoint (no Git)**

Run: `uv run pytest -q && uv run ruff check . && uv run ruff format --check .`
Then confirm the layering rule holds: `grep -rl "import streamlit" src/pdf_assistant/services/` must print nothing.
Then mark Phase 4 complete in `CLAUDE.md` and `README.md` with the real test count.

---

## Self-Review

**Spec coverage.** §5 Section 1 (summary, topics, key points, status, counts) → Task 2. §5 Section 2 (chat, retrieval, citations, follow-ups, history, clear-chat, exact abstention wording) → Task 3. §5 Section 3 (eight options, chunked processing for large PDFs) → Task 4. §5 Section 4 (all eight detail rows, view text, delete, clear on delete) → Task 5 plus the Phase 2 viewer. §6 "clear previous index and chat on new upload" → Task 1. §10 errors 8–13 → Tasks 1–4 error paths. §13 Phase 4 bullets (connect services, four sections, summary and explanation, conversation history, replace and delete) → Tasks 1–5 one-to-one.

**Step scan.** Code steps name the file and the behaviour; the two non-obvious decisions are called out because an implementer would plausibly get them wrong: `clear_document_state` currently copies only `list` defaults and will share a `dict` across documents (Task 4), and the abstention path must short-circuit before calling Groq (Task 3).

**Type consistency.** `ChatMessage(role, content, sources)` matches `schemas.py:104`. `RetrievedChunk.chunk.page_number` matches `schemas.py:74`. `state.is_ready()` already exists and starts returning `True` only after Task 1 sets `READY`. `UNDERSTAND_OPTIONS` keys are quoted from the current `understand.py`. `summarise_document` / `explain_document` signatures match Phase 3 Task 6; `answer_question` matches Phase 3 Task 5.

**Review Focus.** 1→Task 2, 2→Task 3, 3→Task 1 (manual, because `AppTest` cannot drive `st.file_uploader`), 4→Task 3, 5→Task 4. Items 3 and the end-to-end citation correctness check are manual by necessity: `AppTest` has no file-upload widget support, so a real browser pass is the only honest verification.

**Proportion.** Five tasks modifying five existing files plus two new test files. Mostly test names and wiring descriptions; no function bodies, since the services they call are already specified in the Phase 3 plan.
