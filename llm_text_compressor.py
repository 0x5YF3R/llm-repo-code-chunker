import os  # Lets us work with the computer's files and environment settings
import argparse  # Helps us get information from the person running the program
import tiktoken  # Counts how many tokens (pieces of words) are in the text
from openai import OpenAI  # Lets us talk to the OpenAI computer program
import sys  # Lets us stop the program if something goes wrong

# This program helps make big pieces of text much shorter using a smart computer program called OpenAI's GPT models.
# It can turn a long text into bullet points, a glossary, an outline, or just a shorter version, depending on what you want.
# You can choose how short you want the text to be, and the program will keep making it shorter until it fits your goal.
# Let's go through the code step by step!

# This is the main function where everything starts
# It gets the information from the user, checks the files, and runs the compression

def main():
    # Parse command-line arguments
    # This part sets up the rules for what information the user needs to give the program
    parser = argparse.ArgumentParser(description='Compress large text using OpenAI API.')
    parser.add_argument('--large_text', type=str, required=True, help='Path to the large text file.')
    parser.add_argument('--token_target', type=int, required=True, help='Target number of tokens for the final output.')
    parser.add_argument(
        '--compressor_type', type=str, required=False, default='auto_detect',
        choices=[
            'auto_detect', 'bullet_points', 'glossary_terms', 'outline',
            'critical_analysis', 'facts_database', 'keywords_keyphrases'
        ],
        help=(
            "Type of compression to perform:\n"
            "  auto_detect: Analysizes the text and chooses a suitable compression type. This is the default.\n"
            "  bullet_points: Summarizes text using bullet points.\n"
            "  glossary_terms: Extracts and defines key terms as a glossary.\n"
            "  outline: Structures the summary with headings and subheadings.\n"
            "  critical_analysis: Offers analysis on strengths and weaknesses.\n"
            "  facts_database: Extracts factual information only.\n"
            "  keywords_keyphrases: Lists essential terms and phrases."
        )
    )
    parser.add_argument('--model_name', type=str, default='gpt-4o-mini', help='OpenAI model to use for compression.')
    parser.add_argument('--json', action='store_true', help='Output JSON format if set.')
    parser.add_argument('--return_str', action='store_true', help='Return compressed text as a string instead of saving to a file.')
    
    args = parser.parse_args()

    # Check if OpenAI API key is set
    # The program needs a secret key to talk to OpenAI. If it's missing, we stop.
    if not os.getenv('OPENAI_API_KEY'):
        print("Please set the OPENAI_API_KEY environment variable.")
        sys.exit(1)

    # Read the large text file
    # Try to open the file the user gave us. If it doesn't exist, stop and tell the user.
    try:
        with open(args.large_text, 'r', encoding='utf-8') as f:
            large_text = f.read()
    except FileNotFoundError:
        print(f"The file {args.large_text} does not exist.")
        sys.exit(1)

    # Define model token limits
    # Each OpenAI model can only handle a certain amount of text at once. We set those limits here.
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
    # Get the instructions and message to send to the OpenAI model
    system_message, prompts = get_prompts(args.json)

    # Initialize tiktoken encoding
    # This helps us count how many tokens are in our text
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding('cl100k_base')

    # Calculate tokens reserved for prompt
    # We need to save some space for the instructions we send to the model
    tokens_reserved_for_prompt = calculate_prompt_tokens(system_message, prompts, encoding)
    max_chunk_size = model_max_tokens - tokens_reserved_for_prompt

    # Count tokens in the large text
    total_tokens = len(encoding.encode(large_text))

    # Compress the text iteratively until it's under token_target
    # We keep making the text shorter until it fits the size the user wants
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
    # If the user wants the result as a string, print it. Otherwise, save it to a file.
    if args.return_str:
        print(compressed_text)  # Outputs compressed text directly as a string
    else:
        output_file = 'compressed_output.json' if args.json else 'compressed_output.txt'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(compressed_text)
        print(f"Compression complete. Output saved to {output_file}.")

# This function counts how many tokens are in the instructions we send to the model
# We want to make sure we don't go over the model's limit

def calculate_prompt_tokens(system_message, prompts, encoding):
    max_prompt_length = 0
    for prompt in prompts.values():
        full_prompt = system_message + prompt.format(target_word_count=0, chunk_string="")
        prompt_tokens = len(encoding.encode(full_prompt))
        if prompt_tokens > max_prompt_length:
            max_prompt_length = prompt_tokens
    #  
    return max_prompt_length

# This function splits the big text into smaller pieces so the model can handle them
# Each chunk is small enough to fit in the model's memory

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

# This function gives us the instructions and system message for the model
# The instructions change depending on what kind of summary the user wants

def get_prompts(is_json):

    # Define a system message to establish context and ensure consistency
    system_message = (
        "You are a compression assistant. Your task is to compress text to a specified word count or summary format. "
        "Follow the specified compression style, using concise language while retaining essential details."
    )

    # Define the prompts
    prompts = {
        'auto_detect': "Review the following text and choose a mode of compression that best suits the type of content provided, aiming for around {target_word_count} words{json_spec}. Here is the text to analysize and compress:\n\n{chunk_string}",
        'bullet_points': "Summarize the following text into clear bullet points, with each point capturing an essential idea. Aim for approximately {target_word_count} words{json_spec}:\n\n{chunk_string}",
        'glossary_terms': "Extract and define key terms and concepts from the following text, presenting them as a glossary list. Aim for around {target_word_count} words{json_spec}:\n\n{chunk_string}",
        'outline': "Create a structured outline with headings and subheadings, capturing the primary structure and flow of the text. Aim for around {target_word_count} words{json_spec}. Use hierarchical headings to emphasize key points and their relationships:\n\n{chunk_string}",
        'critical_analysis': "Provide a brief analysis of the main points, discussing strengths, weaknesses, or important themes present in the text. Aim for around {target_word_count} words{json_spec}:\n\n{chunk_string}",
        'facts_database': "Extract factual statements from the following text, summarizing key details, statistics, and verifiable information. Aim for around {target_word_count} words{json_spec}:\n\n{chunk_string}",
        'keywords_keyphrases': "List key terms and phrases that represent the main ideas of the following text. Limit the list to approximately {target_word_count} words{json_spec}:\n\n{chunk_string}"
    }

    if is_json:
        for key in prompts:
            prompts[key] = prompts[key].replace("{json_spec}", " and format the output as JSON")
    else:
        for key in prompts:
            prompts[key] = prompts[key].replace("{json_spec}", "")
            
    return system_message, prompts

# This function figures out how many words each chunk should have after compression
# It tries to make each chunk smaller, but not too small, so the final result fits the user's goal

def compute_target_word_count_per_chunk(chunk_tokens_len, total_tokens_len, token_target, aggression_factor):
    # Compute the proportion of the chunk relative to the entire text
    chunk_proportion = chunk_tokens_len / total_tokens_len if total_tokens_len > 0 else 0

    # Compute the target total output tokens per chunk, applying the aggression factor
    target_tokens_per_chunk = (chunk_proportion * token_target) / aggression_factor if aggression_factor != 0 else 0

    # Convert tokens to words, assuming average tokens per word is 1.3
    target_words_per_chunk = max(int(target_tokens_per_chunk / 1.3), 1)

    return target_words_per_chunk

# This function sends a chunk of text to the OpenAI model and gets back the compressed version
# It builds the prompt, sends it, and returns the answer

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

# This makes sure the main function runs if we start the program from the command line
if __name__ == '__main__':
    main()
