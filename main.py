import csv
import sklearn
import pandas
import numpy
import matplotlib
from collections import Counter

import tkinter as tk
from tkinter import filedialog


def build_model():
    path = path_entry.get()
    bins = bins_entry.get()

    # Read the structure file
    structure_file = path + "/Structure.txt"
    with open(structure_file, 'r') as file:
        structure = file.read()
    # TODO: Process the structure and build the model

    # Load train.csv file
    train_file = path + "/train.csv"
    instances = []
    with open(train_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            instances.append(row)

    # Fill missing values
    filled_instances = fill_missing_values(instances)

    # Discretize continuous numerical attributes using equal-width partitioning
    discretized_instances = discretize_numeric_attributes(instances, bins)

    # TODO: Pass structure and instances to the classifier class for construction


def discretize_numeric_attributes(instances, bins):
    # Find the indices of numeric attributes
    numeric_indices = []
    for i, attribute in enumerate(instances[0]):
        if attribute == "NUMERIC":
            numeric_indices.append(i)

    # Discretize numeric attributes using equal-width partitioning
    for i in numeric_indices:
        column_values = [float(row[i]) for row in instances[1:]]  # Exclude header row
        min_value = min(column_values)
        max_value = max(column_values)
        width = (max_value - min_value) / bins

        for j in range(1, len(instances)):
            if instances[j][i] != '':
                value = float(instances[j][i])
                bin_index = int((value - min_value) / width) + 1
                instances[j][i] = str(bin_index)

    return instances

def fill_missing_values(instances):
    # Check if a column is numeric or categorical
    def is_numeric(column_values):
        try:
            [float(value) for value in column_values if value != '']
            return True
        except ValueError:
            return False

    # Calculate column averages for numeric values
    column_averages = {}
    for i in range(0, len(instances[0]) - 1):  # Start from index 1 to skip the class attribute
        column_values = [row[i] for row in instances if row[i] != '']
        if is_numeric(column_values):
            values = [float(value) for value in column_values]
            column_average = sum(values) / len(values)
            column_averages[i] = column_average

    # Replace missing numeric values with column averages of the same class
    for i in range(len(instances)):
        for j in range(0, len(instances[i]) - 1):  # Start from index 1 to skip the class attribute
            if instances[i][j] == '':
                class_value = instances[i][-1]  # Assuming class attribute is the last column
                if j in column_averages:
                    instances[i][j] = str(column_averages[j]) if class_value == 'Y' else ''

    # Replace missing categorical values with the most common value
    for i in range(len(instances)):
        for j in range(0, len(instances[i]) - 1):  # Start from index 1 to skip the class attribute
            if instances[i][j] == '':
                column_values = [row[j] for row in instances if row[j] != '']
                if not is_numeric(column_values):
                    most_common_value = Counter(column_values).most_common(1)[0][0]
                    instances[i][j] = most_common_value

    return instances


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
