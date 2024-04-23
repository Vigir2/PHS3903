import os

def join_text_files_in_directory(input_directory, output_file):
    with open(output_file, 'w') as outfile:
        for filename in os.listdir(input_directory):
            if filename.endswith('.xyz'):
                with open(os.path.join(input_directory, filename), 'r') as infile:
                    outfile.write(infile.read())

if __name__ == "__main__":
    input_directory = "."  # Directory containing text files
    output_file = "joined_file.xyz"  # Name of the output file

    join_text_files_in_directory(input_directory, output_file)
    print("Text files joined successfully!")
