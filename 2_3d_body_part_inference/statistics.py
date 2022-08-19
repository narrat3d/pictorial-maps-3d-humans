import os
import csv
import numpy as np

output_folder_directory = "./experiments"

output_folders = os.listdir(output_folder_directory)

runs = {}

for output_folder in output_folders:    
    output_folder_parts = output_folder.split("_")
    run_name = "_".join(output_folder_parts[:-1])

    results = runs.setdefault(run_name, [])
    
    with open(os.path.join(output_folder_directory, output_folder, output_folder + ".csv")) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=";")
        
        min_loss = 1
        
        for row in reader:      
            loss = row['val_loss']
            min_loss = min(min_loss, float(loss))
            
        results.append(min_loss)


for run_name, results in runs.items():
    avg_result = np.average(results)
    print (run_name, round(avg_result, 5), "(from %s runs)" % len(results))