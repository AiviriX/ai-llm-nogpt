import re

def separate_sentences(input_file_path, output_file_path):
    # Read text from input file
    with open(input_file_path, 'r') as file:
        text = file.read()

    # Separate sentences using regular expression
    sentences = re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])', text)

    # Write separated sentences to the output file
    with open(output_file_path, 'w') as output_file:
        for sentence in sentences:
            output_file.write(sentence.strip() + '\n')

input_file_path = 'content/facts.txt'  # Path to the input text file
output_file_path = 'separated_sentences.txt'  # Path to the output text file
separate_sentences(input_file_path, output_file_path)