import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import os
import types
from unittest import mock
import tiktoken

import llm_codebase_compressor as cc


def test_clone_repo_if_needed_local():
    path = '/tmp/repo'
    assert cc.clone_repo_if_needed(path) == path


def test_clone_repo_if_needed_url(monkeypatch, tmp_path):
    dummy_dir = tmp_path / 'clone'
    monkeypatch.setattr(cc.tempfile, 'mkdtemp', lambda: str(dummy_dir))
    with mock.patch.object(cc.git.Repo, 'clone_from') as clone:
        result = cc.clone_repo_if_needed('https://example.com/repo.git')
        clone.assert_called_once_with('https://example.com/repo.git', str(dummy_dir))
        assert result == str(dummy_dir)


def test_gather_python_files_exclude(tmp_path):
    (tmp_path / 'a.py').write_text('print(1)')
    (tmp_path / 'b.txt').write_text('hi')
    sub = tmp_path / 'sub'
    sub.mkdir()
    (sub / 'c.py').write_text('print(2)')
    files = cc.gather_python_files(str(tmp_path), ['sub/*'])
    assert os.path.join(str(tmp_path), 'a.py') in files
    assert all('sub' not in f for f in files)


def test_build_simple_call_graph(tmp_path):
    code = 'def foo():\n    bar()\n\ndef bar():\n    pass\n'
    file_path = tmp_path / 'code.py'
    file_path.write_text(code)
    graph = cc.build_simple_call_graph(str(file_path))
    assert graph['foo'] == ['bar']
    assert 'bar' in graph


def test_count_tokens():
    enc = tiktoken.get_encoding('cl100k_base')
    assert cc.count_tokens('hello', enc) == len(enc.encode('hello'))


def test_summarise_file_mock_openai():
    fake_resp = types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content='sum'))])
    with mock.patch.object(cc, 'OpenAI') as mock_openai:
        instance = mock_openai.return_value
        instance.chat.completions.create.return_value = fake_resp
        out = cc.summarise_file('code', {}, 'auto_detect', 'gpt-4o-mini')
        assert out == 'sum'
        instance.chat.completions.create.assert_called_once()


def test_compress_repository_integration(tmp_path, monkeypatch):
    f1 = tmp_path / 'a.py'
    f1.write_text('def a():\n    pass')
    f2 = tmp_path / 'b.py'
    f2.write_text('def b():\n    pass')
    monkeypatch.setattr(cc, 'summarise_file', lambda code, graph, mode, model: 'summary')
    result = cc.compress_repository(str(tmp_path), 1000, 'auto_detect', 'gpt-4o-mini', 'markdown', False, [])
    assert 'a.py' in result and 'b.py' in result
