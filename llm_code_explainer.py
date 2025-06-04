import os
import sys
import argparse
import glob
from bs4 import BeautifulSoup
import git
import spacy
from openai import OpenAI

nlp = spacy.load("en_core_web_sm")


def count_tokens(text: str) -> int:
    """Return the number of tokens in the given text using spaCy."""
    doc = nlp(text)
    return len(doc)


def get_code_explanation(code: str, engine: str = "text-davinci-002") -> str:
    """Return a short explanation of a line of Python code using OpenAI."""
    client = OpenAI()
    try:
        response = client.completions.create(
            model=engine,
            prompt=f"Explain the following Python code:\n\n{code}\n",
            max_tokens=100,
            n=1,
            temperature=0.5,
        )
        return response.choices[0].text.strip()
    except Exception as e:
        print(f"Error while generating explanation: {e}")
        return ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate line-by-line code explanations for a repository."
    )
    parser.add_argument(
        "--repo_path",
        required=True,
        help="Local path to the repository. If missing and --repo_url is provided, the repo will be cloned there.",
    )
    parser.add_argument(
        "--repo_url",
        help="Git URL of the repository to clone if repo_path does not exist.",
    )
    parser.add_argument(
        "--output_base_path",
        required=True,
        help="Directory where explained files will be written.",
    )
    parser.add_argument(
        "--engine",
        default="text-davinci-002",
        help="OpenAI model engine used to generate explanations.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.repo_path):
        if not args.repo_url:
            print("repo_path does not exist and --repo_url was not provided.")
            sys.exit(1)
        os.makedirs(args.repo_path, exist_ok=True)
        git.Repo.clone_from(args.repo_url, args.repo_path)

    if os.path.exists(args.output_base_path):
        import shutil

        shutil.rmtree(args.output_base_path)

    python_files = glob.glob(f"{args.repo_path}/**/*.py", recursive=True)

    for file_path in python_files:
        print(f"Processing file: {file_path}\n")
        with open(file_path, "r", encoding="utf-8") as file:
            code_lines = file.readlines()

        output_file_path = (
            file_path.replace(args.repo_path, args.output_base_path).rstrip(".py")
            + "_explained.py"
        )

        output_dir = os.path.dirname(output_file_path)
        os.makedirs(output_dir, exist_ok=True)

        with open(output_file_path, "a", encoding="utf-8") as output_file:
            for line_number, line in enumerate(code_lines, start=1):
                stripped = line.strip()
                if stripped:  # Skip empty lines
                    soup = BeautifulSoup(stripped, "html.parser")
                    text_line = soup.get_text()
                    explanation = get_code_explanation(text_line, engine=args.engine)
                    for explanation_line in explanation.split("\n"):
                        output_file.write(f"# Explanation: {explanation_line}\n")
                    output_file.write(f"# Line {line_number}: {text_line}\n")
                output_file.write(line)
            output_file.write("\n")

        print(f"Appended explanations to: {output_file_path}\n")
        print("====================================\n")


if __name__ == "__main__":
    main()
