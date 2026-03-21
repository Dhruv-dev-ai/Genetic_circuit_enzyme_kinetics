import os
import glob
import math
import pandas as pd

# --- Configuration ---
# Name of the column containing the docking scores
# Change this if your CSV uses a different header for the affinity score
AFFINITY_COLUMN = 'Affinity (kcal/mol)'

# Constants for thermodynamic Kd calculation
R = 0.001987  # Ideal gas constant in kcal/(mol*K)
T = 310.15    # Temperature in Kelvin (37 degrees Celsius)

# Output filename
OUTPUT_FILENAME = 'Kd_Results_Summary.txt'

def calculate_kd():
    # Get the directory where the script is located to use as the base directory
    base_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    
    # We will store all processed data here before writing to the file
    results = []

    print(f"Scanning directory: {base_dir}...\n")

    # 1. Iterate through all items in the base directory
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        
        # Check if the item is a directory (which implies it's a ligand folder)
        if os.path.isdir(item_path):
            ligand_name = item
            
            # Find all CSV files in this specific ligand folder
            csv_files = glob.glob(os.path.join(item_path, '*.csv'))
            
            # Provide a warning if the folder does not contain any CSV files
            if not csv_files:
                print(f"Warning: No CSV files found in folder '{ligand_name}'. Skipping.")
                continue

            best_scores = []
            
            # 2. Iterate through each CSV found in the folder
            for csv_file in csv_files:
                try:
                    # Read the CSV data using pandas
                    df = pd.read_csv(csv_file)
                    
                    # 3. Handle misnamed/missing column scenarios safely
                    if AFFINITY_COLUMN not in df.columns:
                        print(f"Error: Column '{AFFINITY_COLUMN}' not found in {os.path.basename(csv_file)}. Skipping this file.")
                        continue
                        
                    # Find the lowest (most negative) score (the best docking pose)
                    best_score = df[AFFINITY_COLUMN].min()
                    
                    # Handle cases where the column is empty or contains non-numeric invalid data
                    if pd.isna(best_score):
                        print(f"Error: No valid numeric data in column '{AFFINITY_COLUMN}' for {os.path.basename(csv_file)}. Skipping.")
                        continue
                        
                    best_scores.append(best_score)
                    
                except pd.errors.EmptyDataError:
                    print(f"Error: The file {os.path.basename(csv_file)} is empty.")
                except Exception as e:
                    print(f"Error processing file {os.path.basename(csv_file)}: {e}")
            
            # Proceed to calculations only if we have collected some scores
            if best_scores:
                # 4. Average those 3 best scores together
                avg_delta_g = sum(best_scores) / len(best_scores)
                
                # Report if we found fewer or more than 3 replicate CSVs
                if len(best_scores) != 3:
                    print(f"Notice: Expected 3 replicates for '{ligand_name}', but processed {len(best_scores)}.")
                
                try:
                    # 5. Calculate theoretical Kd in Molar (M): Kd = exp(Delta_G / (R * T))
                    kd_molar = math.exp(avg_delta_g / (R * T))
                    
                    # 6. Convert the Kd from Molar (M) to micromolar (uM)
                    kd_umolar = kd_molar * 1e6
                    
                    # Store the final calculated data
                    results.append({
                        'Ligand': ligand_name,
                        'Average_Delta_G': avg_delta_g,
                        'Kd_Molar': kd_molar,
                        'Kd_uMolar': kd_umolar,
                        'Replicates': len(best_scores)
                    })
                except OverflowError:
                    # Prevent mathematical crashes if Delta_G is extremely large
                    print(f"Error: Delta G value ({avg_delta_g}) for '{ligand_name}' is too large for math.exp calculation.")

    # Sort results by Average Delta G for a nicer presentation in the file
    results.sort(key=lambda x: x['Average_Delta_G'])

    # 7 & 8. Output all the final data into a cleanly formatted text file
    output_path = os.path.join(base_dir, OUTPUT_FILENAME)
    
    try:
        with open(output_path, 'w') as f:
            # Write a clear, descriptive header
            f.write("Theoretical Kd Results Summary\n")
            f.write("="*85 + "\n")
            f.write(f"Thermodynamic parameters used: R = {R} kcal/(mol*K), T = {T} K\n")
            f.write("="*85 + "\n\n")
            
            # Print a neat columnar layout
            # Ligand name | Average Delta G | Kd (Molar) | Kd (uM) | Files Processed
            header = f"{'Ligand Name':<20} | {'Avg Delta G (kcal/mol)':<22} | {'Kd (Molar)':<15} | {'Kd (uM)':<15} | {'Replicates'}\n"
            f.write(header)
            f.write("-" * len(header) + "\n")
            
            # Format and list each row of specific data
            for res in results:
                # '.4f' means floating point to 4 decimal places
                # '.4e' means exponential/scientific notation to 4 decimal places
                formatted_string = (
                    f"{res['Ligand']:<20} | "
                    f"{res['Average_Delta_G']:<22.4f} | "
                    f"{res['Kd_Molar']:<15.4e} | "
                    f"{res['Kd_uMolar']:<15.4f} | "
                    f"     {res['Replicates']}\n"
                )
                f.write(formatted_string)
                
        print(f"\nSuccess! Analyzed data and wrote cleanly formatted results to:\n{output_path}")
        
    except IOError as e:
        print(f"Error writing to output file {output_path}: {e}")

if __name__ == "__main__":
    calculate_kd()
