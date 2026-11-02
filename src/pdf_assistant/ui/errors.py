"""Shared error types that the UI must catch when calling services.

Every UI call into a service that might make an API call or process data must
catch exactly these exception types. This module defines them in one place so
later tasks can import the tuple rather than repeating it.

WHY NOT JUST CATCH RuntimeError: All five of these errors subclass RuntimeError,
so ``except RuntimeError`` would catch them all. But that would ALSO catch genuine
bugs like TypeError or AttributeError in our own code - those should surface as
Streamlit tracebacks so the developer sees them, not be swallowed and shown as
"something went wrong" to the user.

By catching only the known error types, we ensure that service failures turn into
readable error messages while genuine bugs stay loud.
"""

from __future__ import annotations

from pdf_assistant.services.embedding_service import EmbeddingError
from pdf_assistant.services.llm_service import LlmError
from pdf_assistant.services.retrieval_service import RetrievalError
from pdf_assistant.services.summary_service import SummaryError
from pdf_assistant.services.vector_store_service import VectorStoreError

# Every UI error handler that calls into services that might fail must catch
# exactly this tuple. Import this rather than hardcoding the list in each place.
GENERATION_ERRORS = (LlmError, RetrievalError, SummaryError, EmbeddingError, VectorStoreError)
