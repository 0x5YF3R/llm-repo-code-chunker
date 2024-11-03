import os
import argparse
import tiktoken
from openai import OpenAI
import sys

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Compress large text using OpenAI API.')
    parser.add_argument('--large_text', type=str, required=True, help='Path to the large text file.')
    parser.add_argument('--token_target', type=int, required=True, help='Target number of tokens for the final output.')
    parser.add_argument('--compressor_type', type=str, required=True, choices=['general', 'bullet_points', 'key_points', 'paraphrase', 'outline', 'keywords'], help='Type of compression to perform.')
    parser.add_argument('--json', action='store_true', help='Output JSON format if set.')
    args = parser.parse_args()

    # Check if OpenAI API key is set
    if not os.getenv('OPENAI_API_KEY'):
        print("Please set the OPENAI_API_KEY environment variable.")
        sys.exit(1)

    # Read the large text file
    with open(args.large_text, 'r', encoding='utf-8') as f:
        large_text = f.read()

    # Initialize tiktoken encoding
    try:
        encoding = tiktoken.encoding_for_model('gpt-4o-mini')
    except KeyError:
        encoding = tiktoken.get_encoding('cl100k_base')

    # Count tokens in the large text
    total_tokens = len(encoding.encode(large_text))

    # Define the prompts
    prompts = get_prompts(args.json)

    # Compress the text iteratively until it's under token_target
    compressed_text = large_text
    current_tokens = total_tokens

    while current_tokens > args.token_target:
        print(f"Current token count: {current_tokens}, target: {args.token_target}. Compressing...")
        chunks, chunk_token_counts = split_text_into_chunks(compressed_text, args.token_target, encoding)
        total_chunk_tokens = sum(chunk_token_counts)
        compressed_chunks = []
        for idx, (chunk, chunk_tokens_len) in enumerate(zip(chunks, chunk_token_counts)):
            target_word_count = compute_target_word_count_per_chunk(chunk_tokens_len, total_chunk_tokens, args.token_target)
            print(f"Compressing chunk {idx + 1}/{len(chunks)} with target word count {target_word_count}...")
            compressed_chunk = compress_chunk(chunk, args.compressor_type, target_word_count, prompts, args.json)
            compressed_chunks.append(compressed_chunk)
        compressed_text = '\n'.join(compressed_chunks)
        current_tokens = len(encoding.encode(compressed_text))

    # Output the result
    output_file = 'output.json' if args.json else 'output.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(compressed_text)
    print(f"Compression complete. Output saved to {output_file}.")

def split_text_into_chunks(text, token_target, encoding):
    # Split the text into chunks of at most token_target tokens
    tokens = encoding.encode(text)
    chunks = []
    chunk_token_counts = []
    max_chunk_size = token_target

    start = 0
    while start < len(tokens):
        end = min(start + max_chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
        chunk_token_counts.append(len(chunk_tokens))
        start = end
    return chunks, chunk_token_counts

def get_prompts(is_json):
    # Define the prompts
    prompts = {
        'general': "Please summarize the following text to approximately {target_word_count} words{json_spec}:\n\n{chunk_string}",
        'bullet_points': "Please create a bullet point summary of the following text, aiming for approximately {target_word_count} words{json_spec}:\n\n{chunk_string}",
        'key_points': "Please extract the key points from the following text, aiming for approximately {target_word_count} words{json_spec}:\n\n{chunk_string}",
        'paraphrase': "Please paraphrase the following text, aiming for approximately {target_word_count} words{json_spec}:\n\n{chunk_string}",
        'outline': "Please create a concise outline with headlines and subheadings for the following text, aiming for approximately {target_word_count} words{json_spec}:\n\n{chunk_string}",
        'keywords': "Distill the following text into a list of keywords or keyphrases, aiming for approximately {target_word_count} words{json_spec}:\n\n{chunk_string}",
    }

    if is_json:
        for key in prompts:
            prompts[key] = prompts[key].replace("{json_spec}", " and provide the output in JSON format")
    else:
        for key in prompts:
            prompts[key] = prompts[key].replace("{json_spec}", "")
    return prompts

def compute_target_word_count_per_chunk(chunk_tokens_len, total_tokens_len, token_target, aggression_factor=1.2):
    # Compute the proportion of the chunk relative to the entire text
    chunk_proportion = chunk_tokens_len / total_tokens_len
    
    # Compute the target total output tokens per chunk, applying the aggression factor
    target_tokens_per_chunk = (chunk_proportion * token_target) / aggression_factor
    
    # Convert tokens to words, assuming average tokens per word is 1.3
    target_words_per_chunk = int(target_tokens_per_chunk / 1.3)
    
    # Return the calculated target word count without a minimum constraint
    return target_words_per_chunk


def compress_chunk(chunk, compressor_type, target_word_count, prompts, is_json):
    # Prepare the prompt
    prompt = prompts[compressor_type].format(
        target_word_count=target_word_count,
        chunk_string=chunk
    )
    # Call OpenAI API
    client = OpenAI()
    try:
        response = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[
                {
                    'role': 'user',
                    'content': prompt
                }
            ]
        )
        compressed_chunk = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"An error occurred while processing the chunk: {e}")
        compressed_chunk = ""
    return compressed_chunk

if __name__ == '__main__':
    main()
