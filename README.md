
# LLM Codebase Compressor

A Python tool that ingests an entire software repository and distills it into a compact, high‑signal representation that fits inside a single large‑language‑model (LLM) context window. It tracks every token against a configurable budget, stripping away boilerplate and duplications while inserting terse, model‑generated commentary so essential architecture, algorithms, and logic remain clear.

---

## Table of Contents

1. Overview  
2. Features  
3. Requirements  
4. Installation  
5. Usage  
   • Example Usage  
6. Compression Modes  
7. Example Repository  
8. Customization  
9. Known Issues  
10. License  

---

## Overview

`llm_codebase_compressor.py` walks a project’s directory tree, maps call graphs and imports, and streams each source file through an OpenAI model. The model returns semantics‑preserving annotations—a one‑sentence summary for helper functions, a concise algorithmic sketch for core routines, or notes on side effects and external dependencies. Repeated patterns such as CLI scaffolding or logging stubs collapse into single semantic tags, and duplicated constructs across modules are referenced rather than replayed. The pipeline dynamically shortens or omits secondary details until the entire narrative meets the token target.

The final artifact (Markdown briefing or pseudo‑code file) interleaves surviving code fragments with generated summaries and preserves original line numbers, enabling round‑trip navigation back to full source.

---

## Features

- **Repository‑Wide Compression**: Accepts local paths or remote Git URLs and processes every file under version control.  
- **Token Budget Enforcement**: Define an exact token target and watch the compressor iteratively tighten summaries to hit it.  
- **Semantic Deduplication**: Collapses boilerplate, repeated idioms, and copy‑pasted functions into single tags or references.  
- **Line‑Number Mapping**: Maintains original line numbers beside summaries so you can jump back to the full code with ease.  
- **Multiple Output Formats**: Choose Markdown, lightweight pseudo‑code, or raw JSON for downstream tooling.  
- **Configurable Aggression Factor**: Automatically ratchets compression strength if the target budget isn’t reached.  
- **Selective Inclusion**: CLI switches to include or exclude tests, vendor folders, generated code, or specific file globs.  

---

## Requirements

- Python 3.8+  
- OpenAI API key  
- Packages:  
  `openai`, `tiktoken`, `argparse`, `rich` (for progress and colorized CLI output)  

---

## Installation

```bash
#Clone this repository:
git clone https://github.com/yourusername/llm-codebase-compressor.git
#Move to the working directory:
cd llm-codebase-compressor
#Install the required Python packages:
pip install -r requirements.txt
#Set up your OpenAI API key as an environment variable
export OPENAI_API_KEY="your_openai_api_key"
```

## Usage

```bash
python llm_codebase_compressor.py \
    --repo_path <path_or_git_url> \
    --token_target <token_count> \
    --mode <compression_mode> \
    [--model_name gpt-4o-mini] \
    [--output_format markdown|pseudo|json] \
    [--include_code_fragments] \
    [--exclude "tests/*,vendor/*"]
```

    --repo_path: Local directory or Git URL to compress.
    --token_target: Desired maximum tokens in the final artifact.
    --mode: Compression mode (see below). Default is auto_detect.
    --model_name: OpenAI model; defaults to gpt-4o-mini.
    --output_format: markdown, pseudo, or json.
    --include_code_fragments: If set, interleave selected original code snippets with summaries.
    --exclude: Comma‑separated glob patterns to ignore.

### Example Usage

```bash
python llm_codebase_compressor.py \
    --repo_path https://github.com/pallets/flask.git \
    --token_target 16000 \
    --mode architecture_outline \
    --output_format markdown \
    --exclude "tests/*,examples/*"
```

The command produces a flask_compressed.md file containing a 16 k‑token architectural briefing of Flask—core modules, call graph highlights, and summarized algorithms—while skipping tests and examples.
Compression Modes

    auto_detect: Scans repository size, language mix, and complexity to choose an appropriate strategy.
    architecture_outline: High‑level overview of directories, core classes, and cross‑module interactions.
    function_summaries: One‑liners for every function or method, sorted by file.
    algorithm_sketches: Compact pseudocode sketches for computationally intense routines.
    dependency_map: Emphasizes import trees, external libraries, and side‑effect notes.
    boilerplate_collapse: Extreme deduplication of logging, CLI, and configuration scaffolding.

Example Repository

For quick testing, clone a small open‑source project such as requests and run the compressor with a 4 k‑token target to observe aggressive summarization and deduplication in action.
Customization

Adjust compression behavior by modifying:
    Token Target: Directly sets the output size ceiling.
    Aggression Factor: Percentage by which the compressor tightens summaries each iteration (default = 10 %).
    File Filters: Include or exclude patterns at the CLI or inside config.yaml.
    Summary Templates: Jinja‑style templates in templates/ govern wording of function and class summaries.

Known Issues

    Iteration Count: Deep repositories may require several passes before hitting token target.
    Language Support: Primary focus is Python; non‑Python files receive coarse summaries.
    Tokenization Variance: Exact token counts can differ slightly from targets because of model tokenization quirks.
    Circular Imports: Extremely tangled import graphs can inflate summaries beyond expectation.


## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
