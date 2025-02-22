import tkinter as tk
from tkinter import ttk
import sv_ttk
import matplotlib
import ctypes
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np

# Blurriness fix
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass

font = "Courier New"
fullfont = ("courier new", 12, "regular")

# Input parameters
num_graphs = 10

# comparison between machine learning and deep learning page

# variables
    # data = 
    # tp = 
    # fp = 
    # tn = 
    # fn = 

# derived quantities
    # accuracy = (tp*tn)/(tp*tn*fp*fn)
    # precision = tp/(tp+fp)
    # recall = tp/(tp+fn)
    # specificity = tn/(tn+fp)
    # f1_score = (2 * precision * recall) / (precision + recall) 
    # confusion_matrix =
    # mcc = 
    # fpr = fp / (fp + tn)
    # fnr = fn / (fn + tp)
    # balanced_accuracy = (recall + specificity) / 2
    
# Create the root window
root = tk.Tk()
root.title("Sample Graphs with Statistics")

# Dark theme
sv_ttk.set_theme("dark")

# Make the window resizable
root.geometry("1200x800")
root.minsize(800, 600)

# Create a frame for inputs at the top
input_frame = ttk.Frame(root)
input_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

# Create at least 4 Entry widgets
parameters = ["parameter 1", "parameter 2", "parameter 3", "parameter 4"]
entries = []
for i in range(4):
    label = ttk.Label(input_frame, text=f"{parameters[i]}:")
    label.pack(side=tk.LEFT, padx=5)
    entry = ttk.Entry(input_frame)
    entry.pack(side=tk.LEFT, padx=5)
    entries.append(entry)

# Create a Notebook widget for page selection
notebook = ttk.Notebook(root)
notebook.pack(fill=tk.BOTH, expand=True)

# Loop over the number of graphs to create pages
for i in range(num_graphs):
    # Create a new frame for each page
    page = ttk.Frame(notebook)
    notebook.add(page, text=f"Page {i+1}")
    
    # Generate random data
    data = np.random.randn(100)
    
    # Compute statistics
    mean = np.mean(data)
    std = np.std(data)
    
    # Create a Figure
    fig = Figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.plot(data)
    ax.set_title(f"Sample Graph {i+1}")
    
    # Create a FigureCanvasTkAgg object
    canvas_fig = FigureCanvasTkAgg(fig, master=page)
    canvas_fig.draw()
    
    # Place the canvas in the page
    canvas_fig.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Create a frame for statistics on the right
    stats_frame = ttk.Frame(page)
    stats_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Display the statistics
    stats_text = f"Mean: {mean:.2f}\nStd Dev: {std:.2f}"
    stats_label = ttk.Label(stats_frame, text=stats_text, anchor='center', font=(font, 12))
    stats_label.pack(expand=True)

# Bind the resize event to dynamically adjust widget sizes
def on_resize(event):
    notebook.update_idletasks()

root.bind("<Configure>", on_resize)

# Start the Tkinter event loop
root.mainloop()
