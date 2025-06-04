# This program helps compress an entire code repository into a short summary.
# It is written with very detailed comments so someone new to Python can follow
# along. The program can clone a remote repository or use a local folder,
# analyse the files, and ask OpenAI's API to create short explanations of each
# file. All of the summaries are combined into a single output file that tries
# to stay under a certain number of tokens.

import os  # Lets us work with the file system
import argparse  # Handles the command line options
import tempfile  # Creates temporary folders when cloning repositories
import glob  # Finds files matching patterns
import ast  # Helps us look at Python code structure
from typing import List, Dict

from openai import OpenAI  # Talk to the OpenAI models
import git  # Used to clone repositories from the internet
import tiktoken  # Counts tokens so we do not go over limits
from rich.progress import Progress  # Shows progress bars in the terminal


# ----- Helper functions -----

def clone_repo_if_needed(repo_path: str) -> str:
    """Clone the repo if repo_path looks like a URL."""
    if repo_path.startswith("http://") or repo_path.startswith("https://"):
        tmp_dir = tempfile.mkdtemp()
        print(f"Cloning repository to temporary folder: {tmp_dir}")
        git.Repo.clone_from(repo_path, tmp_dir)
        return tmp_dir
    return repo_path


def gather_python_files(repo_path: str, exclude: List[str]) -> List[str]:
    """Return a list of Python files under repo_path while respecting exclusions."""
    all_files = glob.glob(os.path.join(repo_path, "**", "*.py"), recursive=True)
    filtered: List[str] = []
    for file in all_files:
        relative = os.path.relpath(file, repo_path)
        if any(glob.fnmatch.fnmatch(relative, pattern) for pattern in exclude):
            continue
        filtered.append(file)
    return filtered


def build_simple_call_graph(file_path: str) -> Dict[str, List[str]]:
    """Create a very small call graph for the given Python file."""
    graph: Dict[str, List[str]] = {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
    except Exception:
        return graph

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            graph[node.name] = []
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    if isinstance(child.func, ast.Name):
                        graph[node.name].append(child.func.id)
                    elif isinstance(child.func, ast.Attribute):
                        graph[node.name].append(child.func.attr)
    return graph


def count_tokens(text: str, encoding) -> int:
    """Return how many tokens the text would take."""
    return len(encoding.encode(text))


def summarise_file(code: str, call_graph: Dict[str, List[str]], mode: str, model: str) -> str:
    """Ask OpenAI to create a short summary of a code file."""
    client = OpenAI()

    system_message = (
        "You are a code compression assistant. Given a code file and a call "
        "graph, produce a short summary following the requested style."
    )

    # Short description of the call graph to pass to the model
    graph_lines = []
    for func, calls in call_graph.items():
        if calls:
            calls_str = ", ".join(calls)
            graph_lines.append(f"Function {func} calls: {calls_str}")
    call_graph_text = "\n".join(graph_lines)

    prompt_styles = {
        "auto_detect": "Summarise the following code:",
        "architecture_outline": "Describe the architecture and important classes:",
        "function_summaries": "Give a one line summary for each function:",
        "algorithm_sketches": "Write short pseudocode for complex parts:",
        "dependency_map": "Explain the imports and side effects:",
        "boilerplate_collapse": "Only keep unique logic, collapsing boilerplate:",
    }
    prompt = (
        f"{prompt_styles.get(mode, prompt_styles['auto_detect'])}\n\n"
        f"Call graph:\n{call_graph_text}\n\nCode:\n{code}"
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]

    try:
        response = client.chat.completions.create(model=model, messages=messages)
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error while contacting OpenAI: {e}")
        return ""


# ----- Main compression logic -----

def compress_repository(
    repo_path: str,
    token_target: int,
    mode: str,
    model_name: str,
    output_format: str,
    include_code_fragments: bool,
    exclude: List[str],
) -> str:
    """Process each file in the repository and build a combined summary."""

    encoding = tiktoken.encoding_for_model(model_name)
    output_chunks: List[str] = []
    current_tokens = 0

    files = gather_python_files(repo_path, exclude)
    if not files:
        print("No Python files found to process.")
        return ""

    with Progress() as progress:
        task = progress.add_task("Compressing files", total=len(files))

        for file_path in files:
            progress.update(task, description=f"{os.path.basename(file_path)}")
            with open(file_path, "r", encoding="utf-8") as f:
                code = f.read()

            call_graph = build_simple_call_graph(file_path)
            summary = summarise_file(code, call_graph, mode, model_name)
            if include_code_fragments:
                snippet = "\n".join(code.splitlines()[:10])
                summary = f"```python\n{snippet}\n``""\n{summary}"

            chunk_tokens = count_tokens(summary, encoding)
            if current_tokens + chunk_tokens > token_target:
                print("Token target reached. Stopping early.")
                break

            current_tokens += chunk_tokens
            output_chunks.append(f"### {os.path.relpath(file_path, repo_path)}\n\n{summary}\n")
            progress.advance(task)

    if output_format == "json":
        import json

        return json.dumps({"files": output_chunks}, indent=2)

    # Markdown or pseudo both return simple concatenation
    return "\n".join(output_chunks)


# ----- Command line interface -----

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compress an entire codebase into a small summary using OpenAI."
    )
    parser.add_argument("--repo_path", required=True, help="Path or Git URL of the repository")
    parser.add_argument("--token_target", type=int, required=True, help="Maximum tokens in output")
    parser.add_argument(
        "--mode",
        default="auto_detect",
        choices=[
            "auto_detect",
            "architecture_outline",
            "function_summaries",
            "algorithm_sketches",
            "dependency_map",
            "boilerplate_collapse",
        ],
        help="Compression mode",
    )
    parser.add_argument("--model_name", default="gpt-4o-mini", help="OpenAI model name")
    parser.add_argument(
        "--output_format",
        default="markdown",
        choices=["markdown", "pseudo", "json"],
        help="Format of the final summary",
    )
    parser.add_argument(
        "--include_code_fragments",
        action="store_true",
        help="Include short code snippets before each summary",
    )
    parser.add_argument(
        "--exclude",
        default="",
        help="Comma separated glob patterns to ignore, e.g. 'tests/*,vendor/*'",
    )

    args = parser.parse_args()

    repo_path = clone_repo_if_needed(args.repo_path)
    exclude_patterns = [pattern.strip() for pattern in args.exclude.split(",") if pattern.strip()]

    compressed = compress_repository(
        repo_path=repo_path,
        token_target=args.token_target,
        mode=args.mode,
        model_name=args.model_name,
        output_format=args.output_format,
        include_code_fragments=args.include_code_fragments,
        exclude=exclude_patterns,
    )

    if not compressed:
        print("No output produced.")
        return

    output_file = f"compressed.{ 'json' if args.output_format == 'json' else 'md'}"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(compressed)
    print(f"Compression complete. Output saved to {output_file}")


if __name__ == "__main__":
    main()
