import tkinter as tk
from tkinter import ttk, filedialog
import sv_ttk
import matplotlib
import ctypes
import pandas as pd
import json
import numpy as np
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Blurriness fix for high DPI screens
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass

# Tkinter setup
root = tk.Tk()
root.title("ML/DL Data Visualization with Inputs")
sv_ttk.set_theme("dark")
root.geometry("1200x800")
root.minsize(800, 600)

num_graphs = 5  # Reduced to 5 for better visualization

# Frame for inputs & file selection
input_frame = ttk.Frame(root)
input_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

# Parameter Labels
parameters = ["Mean", "Std Dev", "Data Points", "Multiplier"]
entries = {}

# Store input values using StringVar
input_vars = {param: tk.StringVar(value="1.0") for param in parameters}

# Create entry fields for parameters
for param in parameters:
    label = ttk.Label(input_frame, text=f"{param}:")
    label.pack(side=tk.LEFT, padx=5)
    
    entry = ttk.Entry(input_frame, textvariable=input_vars[param], width=10)
    entry.pack(side=tk.LEFT, padx=5)
    
    entries[param] = entry

# Notebook for graphs
notebook = ttk.Notebook(root)
notebook.pack(fill=tk.BOTH, expand=True)

# Dictionaries for graph storage
graph_canvases = {}
graph_axes = {}
stats_labels = {}

# Function to load data from CSV or JSON
def load_data(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def load_json(file_path):
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return None

def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv"), ("JSON Files", "*.json")])
    if file_path:
        if file_path.endswith('.csv'):
            data = load_data(file_path)
        elif file_path.endswith('.json'):
            data = load_json(file_path)
        update_graphs(data)

def update_graphs(data=None):
    try:
        if data is None:  # Use user inputs if no file is loaded
            mean_value = float(input_vars["Mean"].get())
            std_dev_value = float(input_vars["Std Dev"].get())
            num_points = int(input_vars["Data Points"].get())
            multiplier = float(input_vars["Multiplier"].get())

        for i in range
