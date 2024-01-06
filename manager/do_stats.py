import csv, re, ast
from datetime import datetime
from collections import namedtuple

# Define the structure of the Trainer Config
TrainerConfig = namedtuple('TrainerConfig', [
    'learning_rate', 'weight_decay', 'epochs', 'optimizer', 'reduction',
    'criterion', 'device', 'batch_size', 'clip_value', 'shuffle'
])

run_pattern = r"Run: (\d+)"
config_pattern = r"Trainer Config: Trainer_Config\((.*?)\)"

def remove_training_steps(line):
    starting_substring = ", training_steps=["
    ending_substring = "],"

    start_index = line.find(starting_substring)
    end_index = line.find(ending_substring) + len(ending_substring)

    return line[:start_index] + "," + line[end_index:]

# Function to parse the log line and extract Trainer Config
def parse_log_line(line):
    line = remove_training_steps(line)
    line = "{\'" + line + "}"
    line = line.replace('=', '\':')
    line = line.replace(', ', ', \'')
    config_dict = ast.literal_eval(line)
    print(config_dict)
    return config_dict

def get_run_number(line):
    match = re.search(run_pattern, line)
    if match:
        # Extract the number from the match
        run_number = int(match.group(1))
        return run_number
    else:
        raise Exception(f"No Run Number in line {line}")

def get_config(line):
    match = re.search(config_pattern, line)

    if match:
        # Extract the substring from the match
        substr = match.group(1)
        return substr
    else:
        raise Exception(f"No Config in line {line}")

# Read the log file and write to CSV
def process_config(filename = 'run_config.log'):
    with open(filename, 'r') as log_file, open('stats.csv', 'w', newline='') as csv_file:
        #log_reader = csv.reader(log_file, delimiter=' ')
        csv_writer = csv.writer(csv_file)

        # Write header to CSV
        csv_writer.writerow(['timestamp', 'run_number', 'learning_rate', 'weight_decay',
                            'epochs', 'optimizer', 'reduction', 'criterion', 'device',
                            'batch_size', 'clip_value', 'shuffle'])

        # Process each line in the log file
        for line in log_file:
            if 'Trainer Config: Trainer_Config' in line:
                timestamp_str = line[:19]
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                run_number = get_run_number(line)
                config_str = get_config(line)
                config = parse_log_line(config_str)
                csv_writer.writerow([
                    timestamp, run_number, config['learning_rate'], config['weight_decay'],
                    config['epochs'], config['optimizer'], config['reduction'], config['criterion'],
                    config['device'], config['batch_size'], config['clip_value'], config['shuffle']
                ])

def main():
    process_config()
if __name__ == "__main__":
    main()