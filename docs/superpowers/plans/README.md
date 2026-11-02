# Implementation Plans

Phase-by-phase plans for the remaining work on PDF Assistant. Written with the
`superpowers:writing-plans` skill.

Phases 1 and 2 are already built and verified, so they have no plan here. The
specification both phases were built against is `../../../../prompt.md`.

| Plan | Phase | Builds | Status |
|---|---|---|---|
| [2026-11-01-phase-3-rag-core.md](2026-11-01-phase-3-rag-core.md) | 3 | Embeddings, FAISS, retrieval, Groq, prompts, summaries | Not started |
| [2026-11-01-phase-4-ui-integration.md](2026-11-01-phase-4-ui-integration.md) | 4 | Wiring the services into the four Streamlit sections | Not started |
| [2026-11-01-phase-5-evaluation.md](2026-11-01-phase-5-evaluation.md) | 5 | 20–30 question evaluation, measured accuracy, error-path testing | Not started |
| [2026-11-01-phase-6-documentation.md](2026-11-01-phase-6-documentation.md) | 6 | The ten `docs/` submission documents and the viva question bank | Not started |

Run them in order. Each plan names its dependency, and Phases 4, 5 and 6 each
rely on the one before it being finished and verified.

## Two deviations from the standard plan format

Both come from this project's own rules, which override the skill's defaults.

1. **No Git.** The skill's task template ends every task with `git add` and
   `git commit`. `prompt.md` §2 forbids Git entirely, so each task instead ends
   with a verification checkpoint that runs the test suite and the linter.
2. **Plans live here, not in the repository root.** `docs/` is otherwise
   reserved for the ten submission documents produced in Phase 6, so planning
   artefacts are kept in this subfolder to avoid mixing them with the material
   that goes to the examiner.

## Where credits start being spent

Phases 1 and 2 cost nothing — no test contacts Groq, and the embedding model is
never downloaded during testing.

That changes in Phase 3. Its unit tests are all mocked and still free, but
**Task 7 makes real API calls**, and the first embedding run downloads about
90 MB. Phase 5's evaluation runner is the most expensive step, which is why it
supports `--limit N` for a cheap smoke run first.
