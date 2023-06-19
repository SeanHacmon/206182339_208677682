import csv
import os
from collections import Counter
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import numpy as np

global classifier


class Classifier:
    def __init__(self, X_train, y_train):
        self.classes = None
        self.class_prior_probabilities = None
        self.feature_probabilities = None
        self.X_train = X_train
        self.y_train = y_train

        self.fit(X_train, y_train)

    def fit(self, X, y):
        self.classes = np.unique(y)
        num_classes = len(self.classes)
        num_features = X.shape[1]

        self.class_prior_probabilities = np.zeros(num_classes)
        self.feature_probabilities = np.zeros((num_classes, num_features))

        for i, c in enumerate(self.classes):
            class_count = np.sum(y == c)
            total_count = len(y)

            # Calculate class prior probability using 3-estimator
            self.class_prior_probabilities[i] = (class_count + 3) / (total_count + 3 * num_classes)

            for feature in range(num_features):
                feature_values = np.unique(X[:, feature])
                num_values = len(feature_values)

                feature_count = np.sum((X[:, feature] == feature_values[0]) & (y == c))

                # Calculate feature probability using 3-estimator
                self.feature_probabilities[i, feature] = (feature_count + 1) / (class_count + 3 * num_values)

    def predict(self, X):
        predictions = []

        for x in X:
            class_scores = []

            for i, c in enumerate(self.classes):
                class_score = self.class_prior_probabilities[i]

                for feature, feature_value in enumerate(x):
                    feature_values = np.unique(self.X_train[:, feature])
                    num_values = len(feature_values)

                    feature_count = np.sum((self.X_train[:, feature] == feature_value) & (self.y_train == c))

                    # Calculate feature probability using 3-estimator
                    feature_prob = (feature_count + 1) / (np.sum(self.y_train == c) + 3 * num_values)

                    class_score *= feature_prob

                class_scores.append(class_score)

            predicted_class = self.classes[np.argmax(class_scores)]
            predictions.append(predicted_class)

        return predictions

def validate_input():
    path = path_entry.get()
    bins = bins_entry.get()

    if not path or not os.path.exists(path):
        messagebox.showerror("Error", "Invalid path. Please select a valid path.")
        return False

    if not bins.isdigit() or int(bins) <= 0:
        messagebox.showerror("Error", "Invalid number of bins. Please enter a positive integer.")
        return False

    return True

def build_model():
    if not validate_input():
        return

    path = path_entry.get()
    bins = bins_entry.get()

    # Load train.csv file
    train_file = path + "/train.csv"
    instances = []
    with open(train_file, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)  # Read the header
        for row in reader:
            instances.append(row)

    # Fill missing values
    filled_instances = fill_missing_values(instances)

    # Discretize continuous numerical attributes using equal-width partitioning
    discretized_instances = discretize_numeric_attributes(filled_instances, bins)
    X_train, y_train = prepare_data(discretized_instances)

    global classifier
    classifier = Classifier(X_train, y_train)

    messagebox.showinfo("Dialog", "building classifier using train-set is done")

def prepare_data(data_array):
    feature_values = []
    class_labels = []

    for row in data_array:
        feature_values.append(row[:-1])  # Extract feature values (except last element)
        class_labels.append(row[-1])  # Extract class label (last element)

    feature_values = np.array(feature_values)
    class_labels = np.array(class_labels)

    # Parse the structure file to obtain feature names and possible values
    feature_names, feature_possible_values = parse_structure_file()

    # Convert feature values to binary representation if needed
    for i in range(feature_values.shape[1]):
        possible_values = feature_possible_values[i]
        if len(possible_values) > 2 and not np.all(np.isin(feature_values[:, i], [0, 1])):  # Convert to binary if more than 2 possible values and not already binary
            for j, value in enumerate(possible_values):
                binary_values = np.where(feature_values[:, i] == value, 1, 0)
                feature_values[:, i] = binary_values

    return feature_values, class_labels


def parse_structure_file():
    feature_names = []
    feature_possible_values = []
    path = path_entry.get()
    structure_file = path + "/Structure.txt"

    with open(structure_file, 'r') as file:
        lines = file.readlines()

        for line in lines:
            line = line.strip()

            if line.startswith('@ATTRIBUTE'):
                parts = line.split(' ')
                feature_name = parts[1]  # Extract the feature name

                # Extract the possible values for the feature (inside curly braces {})
                possible_values = parts[2][1:-1].split(',')

                feature_names.append(feature_name)
                feature_possible_values.append(possible_values)

    return feature_names, feature_possible_values


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
    path = path_entry.get()
    # Load test.csv file
    test_file = path + "/test.csv"
    instances = []
    with open(test_file, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        for row in reader:
            instances.append(row)
    instances_without_labels = [instance[:-1] for instance in instances]
    global classifier
    predicted_labels = classifier.predict(instances_without_labels)

    output_file = path + "/output.txt"
    with open(output_file, 'w') as file:
        for i, label in enumerate(predicted_labels):
            file.write(f"{i+1} {label}\n")

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

build_button = tk.Button(window, text="Build", command=build_model, state="disabled")
build_button.pack()



path_entry.bind("<KeyRelease>", lambda event: enable_build_button())

bins_entry.bind("<KeyRelease>", lambda event: enable_build_button())

def enable_build_button():
    path = path_entry.get()
    bins = bins_entry.get()

    if validate_input():
        build_button.config(state="normal")
    else:
        build_button.config(state="disabled")
        messagebox.showerror("Error", "Please provide a valid path and a valid number of bins.")

    # Enable the "Build" button if all conditions are met
    build_button.config(state="normal")
classify_button = tk.Button(window, text="Classify", command=classify)
classify_button.pack()


# Start the main event loop
window.mainloop()
# Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
