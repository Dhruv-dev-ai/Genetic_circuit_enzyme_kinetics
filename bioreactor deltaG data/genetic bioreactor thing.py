import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import os

def calculate_kd(delta_G, temp_C=37):
    """Converts a docking score (Delta G) to a Kd value."""
    R = 0.001987  # Ideal gas constant in kcal/(mol*K)
    T = temp_C + 273.15  # Convert Celsius to Kelvin
    
    Kd_M = np.exp(delta_G / (R * T))
    Kd_uM = Kd_M * 1000000 # Convert to micromolar (uM)
    return Kd_M, Kd_uM

def select_files_and_calculate():
    # 1. Get the column name from the text box
    col_name = col_entry.get().strip()
    if not col_name:
        messagebox.showerror("Input Error", "Please enter the exact name of the score column.")
        return

    # 2. Open the file explorer window (allows selecting multiple files)
    file_paths = filedialog.askopenfilenames(
        title="Select VdockS CSV Files",
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
    )
    
    if not file_paths:
        return # User closed the window without selecting anything

    best_scores = []
    
    # Clear the results box for a new run
    result_text.delete(1.0, tk.END) 
    result_text.insert(tk.END, f"Processing {len(file_paths)} files...\n\n")

    # 3. Process each selected file
    for file in file_paths:
        try:
            df = pd.read_csv(file)
            if col_name not in df.columns:
                result_text.insert(tk.END, f"❌ Error: Column '{col_name}' not found in {os.path.basename(file)}\n")
                continue
            
            # Find the lowest (most negative) score
            best_score = df[col_name].min()
            best_scores.append(best_score)
            result_text.insert(tk.END, f"📄 {os.path.basename(file)} -> Best Score: {best_score} kcal/mol\n")
            
        except Exception as e:
            result_text.insert(tk.END, f"❌ Error reading {os.path.basename(file)}: {e}\n")

    # 4. Calculate the average and Kd if we successfully got scores
    if best_scores:
        avg_delta_G = np.mean(best_scores)
        kd_molar, kd_micromolar = calculate_kd(avg_delta_G)
        
        summary = (
            f"\n{'='*40}\n"
            f"🎯 FINAL RESULTS:\n"
            f"Average Delta G : {avg_delta_G:.4f} kcal/mol\n"
            f"Theoretical Kd  : {kd_micromolar:.4f} uM\n"
            f"                  ({kd_molar:.2e} M)\n"
            f"{'='*40}\n"
        )
        result_text.insert(tk.END, summary)

# ==========================================
# UI DESIGN (The Window Layout)
# ==========================================
root = tk.Tk()
root.title("VdockS Kd Calculator")
root.geometry("500x450")
root.configure(padx=20, pady=20)

# Input for the Column Name
tk.Label(root, text="Exact CSV Column Name for Affinity/Score:", font=("Arial", 10, "bold")).pack(anchor="w")
col_entry = tk.Entry(root, width=30, font=("Arial", 10))
col_entry.insert(0, "Affinity") # Default placeholder
col_entry.pack(anchor="w", pady=(0, 20))

# The Big Run Button
calc_btn = tk.Button(root, text="📁 Select CSV Files & Calculate", command=select_files_and_calculate, 
                     bg="#2b8cbe", fg="white", font=("Arial", 12, "bold"), pady=10)
calc_btn.pack(fill="x", pady=(0, 20))

# The Output Screen
tk.Label(root, text="Analysis Output:", font=("Arial", 10, "bold")).pack(anchor="w")
result_text = tk.Text(root, height=12, width=50, bg="#1e1e1e", fg="#00ff00", font=("Consolas", 10), padx=10, pady=10)
result_text.pack(fill="both", expand=True)

# Start the App
root.mainloop()
