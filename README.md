
# LLM Text Compressor

A Python script designed to compress large text files, making them more manageable for use within limited LLM context windows. Large texts can quickly consume valuable token space, limiting the space available for additional instructions, context, or responses. This script uses various compression techniques (e.g., bullet points, key points, paraphrasing) to reduce the text to a target token size, allowing you to include it as a file or within a prompt without the risk of unintentionally truncating the end of a long document. By optimizing the token limit, this tool helps ensure more effective use of context for improved results.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Example Usage](#example-usage)
- [Compression Types](#compression-types)
- [Example Input File](#example-input-file)
- [Customization](#customization)
- [Known Issues](#known-issues)
- [License](#license)

## Overview

This script reads a large text file, splits it into manageable chunks, and compresses each chunk using OpenAI's language model to reduce the text to a specified target token size. It iteratively reduces the size of the text until it reaches the target token count, adjusting compression aggressiveness if necessary to prevent endless loops.

## Features

- **Multiple Compression Techniques**: Choose from options like summarizing, outlining, paraphrasing, extracting keywords, and more.
- **Target Token Count**: Define the exact token count you want for the output.
- **JSON Support**: Output in JSON format if desired.
- **Automatic Loop Prevention**: The script automatically increases compression aggression after each iteration to prevent getting stuck in a loop.
- **Customizable Parameters**: Modify the type of compression and output format through command-line arguments.

## Requirements

- Python 3.7+
- OpenAI API key
- The following Python packages:
  - `tiktoken`
  - `openai`
  - `argparse`

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/llm-text-compressor.git
    cd llm-text-compressor
    ```

2. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up your OpenAI API key as an environment variable:
    ```bash
    export OPENAI_API_KEY="your_openai_api_key"
    ```

## Usage

```bash
python llm_text_compressor.py --large_text <path_to_text_file> --token_target <token_count> --compressor_type <type> [--json] [--return_str]
```

- `--large_text`: Path to the large text file you want to compress (e.g., `us_constitution.txt`).
- `--token_target`: Target number of tokens for the final output.
- `--compressor_type`: Optional. Type of compression. Options include:
  - `narrative_summary` (Default)
  - `bullet_points`
  - `glossary_terms`
  - `outline`
  - `critical_analysis`
  - `facts_database`
  - `keywords_keyphrases`
- `--model_name`: Optional. This is `gpt-4o-mini` by default.
- `--json`: Optional. Use this flag to output in JSON format.
- `--return_str`: Optional. Use this flag to return the result as a string.

### Example Usage

```bash
python llm_text_compressor.py --large_text sample_files/us_constitution.txt --token_target 2000 --compressor_type facts_database --model_name gpt-4o-mini --json --return_str
```

This command will take the `us_constitution.txt` file, compress it to approximately 2000 tokens, and extract keywords in JSON format using the model `gpt-4o-mini`.

## Compression Types

The `compressor_type` parameter lets you choose the method of compression. Here’s a breakdown of the options:

- **narrative_summary**: Provides a readable, story-like summary that conveys the main ideas in prose format.
- **bullet_points**: Summarizes the text using bullet points for clarity and quick reading.
- **glossary_terms**: Extracts and defines key terms as a glossary.
- **outline**: Structures the summary with headings and subheadings to capture the flow and main points.
- **critical_analysis**: Analyzes the main points, discussing strengths, weaknesses, and underlying themes.
- **facts_database**: Extracts factual statements, key details, and verifiable information from the text.
- **keywords_keyphrases**: Distills the text into a list of keywords or phrases that represent the core content, ideal for quick reference.

## Example Input File

You can use a sample text file, like the U.S. Constitution, for testing. Save it as `us_constitution.txt` and place it in the project directory.

## Customization

The script's compression algorithm can be fine-tuned by adjusting the `token_target` and compression type, as well as the aggression factor, which increases by 10% with each iteration if the target token count is not met.

### Modifying the Aggression Factor
By default, the script adjusts the compression aggression by 10% each iteration to avoid endless loops. You can modify this factor within the script if needed.

## Known Issues

1. **Loop Prevention**: Although an aggression factor is applied, in some cases, it may take multiple iterations to reach the target token count.
2. **Accuracy of Compression**: Compression accuracy varies by text complexity and chosen compression type.
3. **Token Count Limitations**: The actual output may not match the exact `token_target` due to model and tokenization limitations.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
