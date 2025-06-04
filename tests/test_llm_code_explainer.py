import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import builtins
import types
from unittest import mock
import pytest

import llm_code_explainer as explainer


def test_count_tokens_basic():
    text = "print('hello world')"
    expected = len(explainer.nlp(text))
    assert explainer.count_tokens(text) == expected


def test_get_code_explanation_returns_text():
    fake_resp = types.SimpleNamespace(choices=[types.SimpleNamespace(text=" explanation ")])
    with mock.patch.object(explainer, "OpenAI") as mock_openai:
        instance = mock_openai.return_value
        instance.completions.create.return_value = fake_resp
        result = explainer.get_code_explanation("print('hi')")
        assert result == "explanation"
        instance.completions.create.assert_called_once()


def test_main_exits_without_repo_url(tmp_path):
    repo_path = tmp_path / "does_not_exist"
    args = ["prog", "--repo_path", str(repo_path), "--output_base_path", str(tmp_path / "out")]
    with mock.patch.object(builtins, "print") as mock_print:
        with mock.patch.object(sys, "argv", args):
            with mock.patch.object(explainer, "sys", sys):
                with pytest.raises(SystemExit):
                    explainer.main()
    mock_print.assert_called()
