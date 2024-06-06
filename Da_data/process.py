import csv
import sys

name = sys.argv[1]
version = sys.argv[2]
try: 
    suffix = sys.argv[3]
except IndexError: 
    suffix = ""


# Define the input and output file paths
input_file = f"v{version}_output/{name}{suffix}.txt"
output_file = f"v{version}_output/{name}{suffix}.csv"

# Open the input and output files
with open(input_file, "r") as file_in, open(output_file, "w", newline="") as file_out:
    # Create a CSV writer object
    csv_writer = csv.writer(file_out)
    
    # Write the headers to the output CSV file
    csv_writer.writerow(["d", "sigma", "theta", "eta"])
    
    # Iterate through each line in the input file
    for line in file_in:
        tokens = line.split()
        d, sigma, theta, eta = 0, 0, 0, 0
        for i in range(len(tokens)): 
            if tokens[i] == "theta:": 
                theta = tokens[i + 1]
            elif tokens[i] == "d:":
                d = tokens[i + 1]
            elif tokens[i] == "sigma:": 
                sigma = tokens[i + 1]
            elif tokens[i] == "eta:": 
                eta = tokens[i + 1]
        csv_writer.writerow([d, sigma, theta, eta])


print("CSV file has been created successfully.")
