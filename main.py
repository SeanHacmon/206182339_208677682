import sklearn
import pandas
import numpy
import matplotlib

import tkinter as tk
from tkinter import filedialog


def build_model():
    path = path_entry.get()
    bins = bins_entry.get()

    # TODO: Add code to build the model using the provided path and bins


def classify():
    return
    # TODO: Add code for the classification function

# Create the main window
window = tk.Tk()
window.title("Naive Bayes Classifier")
window.geometry("300x200")
# Function to browse and select a path
def browse_path():
    folder_path = filedialog.askdirectory()
    path_entry.delete(0, tk.END)
    path_entry.insert(tk.END, folder_path)


# Create and pack the components
path_label = tk.Label(window, text="Path:")
path_label.pack()

path_entry = tk.Entry(window)
path_entry.pack()

browse_button = tk.Button(window, text="Browse", command=browse_path)
browse_button.pack()

bins_label = tk.Label(window, text="Discretization Bins")
bins_label.pack()

bins_entry = tk.Entry(window)
bins_entry.pack()

build_button = tk.Button(window, text="Build", command=build_model)
build_button.pack()

classify_button = tk.Button(window, text="Classify", command=classify)
classify_button.pack()

# Start the main event loop
window.mainloop()

# Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
