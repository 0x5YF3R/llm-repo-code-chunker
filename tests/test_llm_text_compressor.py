import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from unittest import mock
import types
import tiktoken
import pytest

import llm_text_compressor as tc


def test_calculate_prompt_tokens():
    encoding = tiktoken.get_encoding('cl100k_base')
    system_message = 'sys'
    prompts = {'a': 'hello {target_word_count} {chunk_string}'}
    # Build expected using same formula
    expected = len(encoding.encode(system_message + prompts['a'].format(target_word_count=0, chunk_string="")))
    assert tc.calculate_prompt_tokens(system_message, prompts, encoding) == expected


def test_split_text_into_chunks_simple():
    encoding = tiktoken.get_encoding('cl100k_base')
    text = 'a b c d'
    chunks, counts = tc.split_text_into_chunks(text, 2, encoding)
    assert chunks and counts
    assert sum(counts) == len(encoding.encode(text))
    assert all(c <= 2 for c in counts)


def test_get_prompts_json_flag():
    system_message, prompts = tc.get_prompts(True)
    assert 'format the output as JSON' in prompts['auto_detect']
    assert isinstance(system_message, str)


def test_compute_target_word_count_per_chunk():
    result = tc.compute_target_word_count_per_chunk(50, 100, 60, 2)
    assert result == max(int(((0.5*60)/2)/1.3), 1)


def test_compress_chunk_uses_openai():
    fake_resp = types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content='short'))])
    with mock.patch.object(tc, 'OpenAI') as mock_openai:
        instance = mock_openai.return_value
        instance.chat.completions.create.return_value = fake_resp
        out = tc.compress_chunk('txt', 'auto_detect', 10, {'auto_detect': 'hey {target_word_count} {chunk_string}'}, False, 'sys', 'model')
        assert out == 'short'
        instance.chat.completions.create.assert_called_once()
