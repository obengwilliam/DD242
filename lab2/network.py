import pickle
import os
from numpy import ndarray, number
import numpy as np


from typing import Tuple, Dict
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=200)

np.random.seed(10)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

CIDAR_LABELS = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def unpickle(filename: str) -> dict:
    """Copy from the dataset website."""
    with open(os.path.join(f"{ROOT_DIR}/dataset/", filename), "rb") as file:
        dict = pickle.load(file, encoding="bytes")
    return dict


def load_batch_cidar(batch_num: int) -> Tuple[ndarray, ndarray]:
    data = unpickle(f"data_batch_{batch_num}")
    input_features = data[b"data"].reshape(len(data[b"data"]), 3, 32, 32).transpose(0, 2, 3, 1)
    labels = data[b"labels"]
    return np.array(input_features), np.array(labels)


def load_batch(filename: str) -> Tuple[ndarray, ndarray, ndarray]:
    """Get Training and Test Data from our ciddar data.

    Args:
     filename (str): name of the batch to load

    Returns:
        X (ndarray): images  of d x N
        Y (ndarray): one hot representation of labels of K x n
        y (ndarray): vector containing the label of each image
    """
    dataDict = unpickle(filename)
    print("1", dataDict[b"data"][1, :])
    X = (dataDict[b"data"] / 255).T
    print("2", X[:, 1])
    y = np.array(dataDict[b"labels"])
    Y = np.eye(10)[y].T
    return X, Y, y


def normalize(training_data: ndarray) -> Tuple[ndarray, ndarray, ndarray]:
    """Normalize Training data."""
    mean: ndarray = np.mean(training_data, axis=0)
    std: ndarray = np.std(training_data, axis=0)
    normalized_data: ndarray = (training_data - mean) / std
    return normalized_data, mean, std


def normalize_mean_std(training_data: ndarray, mean, std) -> ndarray:

    return (training_data - mean) / std


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    exps = np.exp(x - np.max(x, axis=0))
    return exps / exps.sum(axis=0)


def evaluate_classifier(X, W, b) -> ndarray:
    return softmax(W @ X + b)


def safe_log(data: ndarray) -> ndarray:
    return np.log(data, where=data != 0)


class Network:
    def __init__(self, data: Dict):
        self.Y = np.copy(data["Y_train"])
        self.y = np.copy(data["y_train"])
        self.X = np.copy(data["X_train"])
        self.X_val = np.copy(data["X_val"])
        self.Y_val = np.copy(data["Y_val"])
        self.y_val = np.copy(data["y_val"])

        self.X_test = np.copy(data["X_test"])
        self.Y_test = np.copy(data["X_test"])


def main():
    """ Load data and run classification experiments """
    X_train, Y_train, y_train = load_batch("data_batch_1")
    X_test, Y_test, y_test = load_batch("test_batch")
    X_val, Y_val, y_val = load_batch(("data_batch_2"))

    X_train, X_train_mean, X_train_std = normalize(X_train)
    X_test = normalize_mean_std(X_test, X_train_mean, X_train_std)
    X_val = normalize_mean_std(X_val, X_train_mean, X_train_std)

    data = {
        "X_train": X_train,
        "Y_train": Y_train,
        "y_train": y_train,
        "X_test": X_test,
        "Y_test": Y_test,
        "y_test": y_test,
        "X_val": X_val,
        "Y_val": Y_val,
        "y_val": y_val,
    }

    network = Network(data)


if __name__ == "__main__":
    main()
