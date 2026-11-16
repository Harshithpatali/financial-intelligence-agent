from types import SimpleNamespace

import pytest

from backend.rag.llm import extract_answer


def test_extract_answer_returns_trimmed_content():
    response = SimpleNamespace(
        id="resp_123",
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="  Revenue was ₹100 crore.  "),
            finish_reason="stop",
        )],
    )
    assert extract_answer(response) == "Revenue was ₹100 crore."


def test_extract_answer_reports_empty_completion_metadata():
    response = SimpleNamespace(
        id="resp_empty",
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="  "),
            finish_reason="stop",
        )],
    )
    with pytest.raises(RuntimeError, match=r"response_id=resp_empty, finish_reason=stop"):
        extract_answer(response)


def test_extract_answer_reports_missing_choices():
    response = SimpleNamespace(id="resp_no_choices", choices=[])
    with pytest.raises(RuntimeError, match="no completion choices"):
        extract_answer(response)
