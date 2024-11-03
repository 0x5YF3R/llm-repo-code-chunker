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
    parser.add_argument('--compressor_type', type=str, required=True, choices=['general', 'bullet_points', 'key_points', 'paraphrase', 'outline', 'keywords'],
                        help=(
                            'Type of compression to perform:\n'
                            '  general: Provides a concise summary of the main ideas in prose format.\n'
                            '  bullet_points: Summarizes text using bullet points for clarity and quick reading.\n'
                            '  key_points: Extracts only the essential points or highlights.\n'
                            '  paraphrase: Restates the content with similar meaning but fewer words.\n'
                            '  outline: Creates an outline with headlines and subheadings to capture structure.\n'
                            '  keywords: Extracts key terms and phrases representing the core content.'
                        ))
    parser.add_argument('--json', action='store_true', help='Output JSON format if set.')
    parser.add_argument('--model_name', type=str, default='gpt-4o-mini', help='OpenAI model to use for compression.')

    args = parser.parse_args()

    # Check if OpenAI API key is set
    if not os.getenv('OPENAI_API_KEY'):
        print("Please set the OPENAI_API_KEY environment variable.")
        sys.exit(1)

    # Read the large text file
    try:
        with open(args.large_text, 'r', encoding='utf-8') as f:
            large_text = f.read()
    except FileNotFoundError:
        print(f"The file {args.large_text} does not exist.")
        sys.exit(1)

    # Define model token limits
    MODEL_TOKEN_LIMITS = {
        'gpt-4o': 128000,
        'gpt-4o-mini': 128000,
        'gpt-4-turbo': 128000,
        'gpt-4': 8192,
        'gpt-3.5-turbo': 16385,
        'gpt-3.5-turbo-instruct': 4096,
        'o1-preview': 128000,
        'o1-mini': 128000
    }

    model_name = args.model_name

    # Default to 4096 if model not found. 
    # Lower token limit to avoid "maximum context length" errors.
    model_max_tokens = MODEL_TOKEN_LIMITS.get(model_name, 4096) - 1000
    
    # Unpack system message and prompts
    system_message, prompts = get_prompts(args.json)

    # Initialize tiktoken encoding
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding('cl100k_base')

    # Calculate tokens reserved for prompt
    tokens_reserved_for_prompt = calculate_prompt_tokens(system_message, prompts, encoding)
    max_chunk_size = model_max_tokens - tokens_reserved_for_prompt

    # Count tokens in the large text
    total_tokens = len(encoding.encode(large_text))

    # Compress the text iteratively until it's under token_target
    compressed_text = large_text
    current_tokens = total_tokens
    iteration = 0
    max_iterations = 100
    initial_aggression_factor = 1.2
    aggression_factor = initial_aggression_factor

    while current_tokens > args.token_target and iteration < max_iterations:
        print(f"Iteration {iteration + 1}: Current token count: {current_tokens}, target: {args.token_target}. Compressing...")
        chunks, chunk_token_counts = split_text_into_chunks(compressed_text, max_chunk_size, encoding)
        total_chunk_tokens = sum(chunk_token_counts)
        compressed_chunks = []
        for idx, (chunk, chunk_tokens_len) in enumerate(zip(chunks, chunk_token_counts)):
            target_word_count = compute_target_word_count_per_chunk(chunk_tokens_len, total_chunk_tokens, args.token_target, aggression_factor)
            print(f"Compressing chunk {idx + 1}/{len(chunks)} with target word count {target_word_count}...")
            compressed_chunk = compress_chunk(chunk, args.compressor_type, target_word_count, prompts, args.json, system_message, model_name)
            compressed_chunks.append(compressed_chunk)
        compressed_text = '\n'.join(compressed_chunks)
        current_tokens = len(encoding.encode(compressed_text))
        # Increase aggression_factor by 10% for the next iteration
        aggression_factor *= 1.1
        iteration += 1

    if iteration >= max_iterations and current_tokens > args.token_target:
        print("Maximum iterations reached. Compression may not have reached the target token count.")
    else:
        print("Compression successful.")

    # Output the result
    output_file = 'compressed_output.json' if args.json else 'compressed_output.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(compressed_text)
    print(f"Compression complete. Output saved to {output_file}.")


def calculate_prompt_tokens(system_message, prompts, encoding):
    max_prompt_length = 0
    for prompt in prompts.values():
        full_prompt = system_message + prompt.format(target_word_count=0, chunk_string="")
        prompt_tokens = len(encoding.encode(full_prompt))
        if prompt_tokens > max_prompt_length:
            max_prompt_length = prompt_tokens
    #  
    return max_prompt_length


def split_text_into_chunks(text, max_chunk_size, encoding):
    # Split the text into chunks of at most max_chunk_size tokens
    tokens = encoding.encode(text)
    chunks = []
    chunk_token_counts = []
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

    # Define a system message to establish context and ensure consistency
    system_message = (
        "You are a compression assistant. Your task is to compress text to a specified word count or summary format. "
        "Follow the specified compression style, using concise language while retaining essential details."
    )

    # Define the prompts
    prompts = {
        'general': "Summarize the following text to around {target_word_count} words{json_spec}. Ensure the summary captures all main ideas in a coherent narrative:\n\n{chunk_string}",
        'bullet_points': "Summarize the following text into bullet points, with each point capturing an essential idea. Aim for approximately {target_word_count} words{json_spec}:\n\n{chunk_string}",
        'key_points': "Extract the core ideas or highlights from the following text. Focus only on key points, aiming for around {target_word_count} words{json_spec}:\n\n{chunk_string}",
        'paraphrase': "Rewrite the following text in a more concise manner, keeping the meaning but reducing length to around {target_word_count} words{json_spec}:\n\n{chunk_string}",
        'outline': "Create an outline of the following text with clear headings and subheadings. Aim for a structured summary of approximately {target_word_count} words{json_spec}:\n\n{chunk_string}",
        'keywords': "List key terms and phrases that represent the main ideas of the following text. Limit the list to approximately {target_word_count} words{json_spec}:\n\n{chunk_string}",
    }

    if is_json:
        for key in prompts:
            prompts[key] = prompts[key].replace("{json_spec}", " and format the output as JSON")
    else:
        for key in prompts:
            prompts[key] = prompts[key].replace("{json_spec}", "")
            
    return system_message, prompts

def compute_target_word_count_per_chunk(chunk_tokens_len, total_tokens_len, token_target, aggression_factor):
    # Compute the proportion of the chunk relative to the entire text
    chunk_proportion = chunk_tokens_len / total_tokens_len if total_tokens_len > 0 else 0

    # Compute the target total output tokens per chunk, applying the aggression factor
    target_tokens_per_chunk = (chunk_proportion * token_target) / aggression_factor if aggression_factor != 0 else 0

    # Convert tokens to words, assuming average tokens per word is 1.3
    target_words_per_chunk = max(int(target_tokens_per_chunk / 1.3), 1)

    return target_words_per_chunk

def compress_chunk(chunk, compressor_type, target_word_count, prompts, is_json, system_message, model_name):
    # Prepare the prompt
    prompt = prompts[compressor_type].format(
        target_word_count=target_word_count,
        chunk_string=chunk
    )
    
    # Call OpenAI API with system message included
    client = OpenAI()
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    'role': 'system',
                    'content': system_message
                },
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
