# This program explains Python code line by line using a smart computer program (OpenAI).
# It can download a code project, read each Python file, and write a new file with explanations for each line.
# Let's go through the code step by step!

import os  # Lets us work with files and folders
import sys  # Lets us stop the program if something goes wrong
import argparse  # Helps us get information from the person running the program
import glob  # Helps us find files that match a pattern
from bs4 import BeautifulSoup  # Helps clean up text from code lines
import git  # Lets us download code from the internet (GitHub)
import spacy  # Helps us count words and tokens in text
from openai import OpenAI  # Lets us talk to the OpenAI computer program

nlp = spacy.load("en_core_web_sm")  # Loads a language model to help with text

# This function counts how many tokens (pieces of words) are in the text
# It uses spaCy to split the text into tokens


def count_tokens(text: str) -> int:
    """Return the number of tokens in the given text using spaCy."""
    doc = nlp(text)
    return len(doc)

# This function asks OpenAI to explain a line of Python code
# It sends the code to OpenAI and gets back a short explanation


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

# This is the main function where everything starts
# It gets information from the user, finds the code files, and writes the explanations


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

    # If the code folder doesn't exist, and a URL is given, download the code from the internet
    if not os.path.exists(args.repo_path):
        if not args.repo_url:
            print("repo_path does not exist and --repo_url was not provided.")
            sys.exit(1)
        os.makedirs(args.repo_path, exist_ok=True)
        git.Repo.clone_from(args.repo_url, args.repo_path)

    # If the output folder already exists, delete it so we start fresh
    if os.path.exists(args.output_base_path):
        import shutil

        shutil.rmtree(args.output_base_path)

    # Find all Python files in the code folder
    python_files = glob.glob(f"{args.repo_path}/**/*.py", recursive=True)

    # Go through each Python file one by one
    for file_path in python_files:
        print(f"Processing file: {file_path}\n")
        with open(file_path, "r", encoding="utf-8") as file:
            code_lines = file.readlines()

        # Make a new file name for the explained version
        output_file_path = (
            file_path.replace(args.repo_path, args.output_base_path).rstrip(".py")
            + "_explained.py"
        )

        output_dir = os.path.dirname(output_file_path)
        os.makedirs(output_dir, exist_ok=True)

        # Open the new file and write explanations for each line
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


# This makes sure the main function runs if we start the program from the command line
if __name__ == "__main__":
    main()
