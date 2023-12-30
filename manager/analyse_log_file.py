import re

# Define the substrings to filter by
steps = ['Training', 'Validation', 'Testing']
substring1 = ' Results!'
substring2 = 'Avg:'

# Function to extract Avg value from a log line
def extract_avg(line):
    match = re.search(r'Avg:(\s*\d+\.\d+)', line)
    return float(match.group(1)) if match else 0.0

# Function to process log lines
def process_log(input_file, output_file, substr1):
    # Read all lines from the input file
    with open(input_file, 'r') as infile:
        lines = infile.readlines()

    # Filter lines containing both substrings
    filtered_lines = [line for line in lines if substr1 in line and substring2 in line]

    # Sort lines based on Avg value
    sorted_lines = sorted(filtered_lines, key=extract_avg)

    # Write the remaining lines to the output file
    with open(output_file, 'w') as outfile:
        outfile.writelines(sorted_lines)

def main():
    input_filename = 'app.log'
    for step in steps:
        output_filename = step + '_' + 'results.log'
        substr1 = step + substring1
        process_log(input_filename, output_filename, substr1)

if __name__ == "__main__":
    main()
