import csv, re, ast
from datetime import datetime
from collections import namedtuple
import pandas as pd

run_pattern = r"Run: (\d+)"
config_pattern = r"Trainer Config: Trainer_Config\((.*?)\)"
avg_pattern = r"Run: (\d+).*Avg: (\d+\.\d+)"
columns_to_remove = ['conv_config', 'dl_config', 'net_config']

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
    for column in columns_to_remove:
        config_dict.pop(column, None)
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
    
def get_avg(line):
    match = re.search(avg_pattern, line)
    if match:
        run_number = int(match.group(1))
        average = float(match.group(2))
        return run_number, average
    else:
        raise Exception(f"No Average in {line}")

def add_averages_from_file(df, column, filename):
    with open(filename, 'r') as log_file:
        for line in log_file:
            run_number, average = get_avg(line)
            df.loc[df['run_number'] == run_number, column] = average

def add_averages(df):
    add_averages_from_file(df, "train_results", "Training_results.log")
    add_averages_from_file(df, "val_results", "Validation_results.log")
    add_averages_from_file(df, "test_results", "Testing_results.log") 
    return df
 
# Read the log file and write to CSV
def process_config(filename = 'run_config.log'):
    df = None
    with open(filename, 'r') as log_file:
        first_line = True
        for line in log_file:
            if 'Trainer Config: Trainer_Config' in line:
                timestamp_str = line[:19]
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                run_number = get_run_number(line)
                config_str = get_config(line)
                config = parse_log_line(config_str)
                config["run_number"] = run_number + 1
                config["timestamp"] = timestamp
                config["train_results"] = -1.
                config["val_results"] = -1.
                config["test_results"] = -1.
                epochs = config["epochs"]
                if first_line:
                    first_line = False
                    df = pd.DataFrame([config])
                else:
                    df.loc[len(df)] = config
                for index in range (epochs-1):
                    config["run_number"] = run_number + 2 + index
                    df.loc[len(df)] = config
    add_averages(df)
    #correlation_matrix = df.corr()
    #impact_on_output = correlation_matrix['output_column_name'].abs().sort_values(ascending=False)
    #print(impact_on_output)
    df.to_csv("stats.csv", index=False, sep=';', decimal=',')

def main():
    process_config()
if __name__ == "__main__":
    main()