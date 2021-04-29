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


def batch_image_stats(batch_num: int, sample_num: int) -> None:
    """Print basic stats of cidar batch."""
    batch_nums = list(range(1, 6))

    assert batch_num in batch_nums, "Batch Num is out of Range."

    input_features, target_labels = load_batch_cidar(batch_num)

    assert 0 <= sample_num < len(input_features), f"Sample num {sample_num} is not valid"

    print(f"Sample size {sample_num}")
    print(f"Batch num {batch_num}")
    print(f"Shape of input features {input_features.shape} ")
    print(f"Shape of labels {target_labels.shape} ")
    print(f"Number of samples in this batch {len(input_features)}")
    print(
        f"Label Count per class {dict(zip(*np.unique(target_labels, return_counts=True)))}"  # type: ignore
    )

    image = input_features[sample_num]
    label = target_labels[sample_num]

    print(f"Minimum pixel value for sample", image.min())
    print(f"Maximum pixel value for sample", image.max())
    print(f" Shape of sample image {image.shape}")
    print(f"Label of sample {CIDAR_LABELS[label]}")

    # plt.axis("off")
    # plt.imshow(image)
    # plt.show()
    plot_10_images(input_features[:10], target_labels[:10])


def plot_image(image: ndarray) -> None:
    """Plot a single image."""
    plt.imshow(image)
    plt.show()


def plot_10_images(x: ndarray) -> None:
    """Plot 10 images from cidar."""
    assert len(x) == 10, "Images can not be more than 10"

    _, ax = plt.subplots(2, 5)

    for r in range(2):
        for c in range(5):
            im = x[r * c]
            ax[r][c].imshow(im)
            ax[r][c].axis("off")
    plt.show()


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
    X = (dataDict[b"data"]).T
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

        self.W = self.init_weight((self.Y.shape[0], self.X.shape[0]))
        self.b = self.init_bias((self.Y.shape[0], 1))

    def init_weight(self, size: Tuple):
        """Initialize Weight."""
        mu = 0
        sigma = 0.01
        return np.random.normal(mu, sigma, size=size).astype("float64")

    def init_bias(self, size=Tuple):
        """Initialize Bias."""
        mu = 0
        sigma = 0.01

        return np.random.normal(mu, sigma, size=size).astype("float64")

    def evaluate_classifier(self, X) -> ndarray:
        """Evaluate Classifier."""
        return softmax(self.W @ X + self.b)

    def compute_accuracy(self, X, y):
        """Compute Accuracy."""

        P = self.evaluate_classifier(X)

        arg_max_p = np.argmax(P, axis=0)
        count = arg_max_p.T[arg_max_p == self.y].shape[0]
        return (count / arg_max_p.shape[0]) * 100

    def compute_cost(self, X, Y, lamda):
        """Compute Cost."""
        N = X.shape[1]

        P = self.evaluate_classifier(X)
        cost = (1 / N) * -np.sum(Y * np.log(P)) + lamda * np.sum(self.W ** 2)

        return cost

    def compute_gradients_ana(self, X_batch, Y_batch, lamda):
        """Compute gradient analytically."""
        N = X_batch.shape[1]
        P = self.evaluate_classifier(X_batch)
        G = -(Y_batch - P)

        gradient_W = 1 / N * G @ X_batch.T + 2 * lamda * self.W
        gradient_b = np.reshape(1 / N * G @ np.ones(N), (Y_batch.shape[0], 1))

        return gradient_W, gradient_b

    def compute_gradients_num(self, X_batch, Y_batch, lamda=0, h=1e-6):
        grad_W = np.zeros(self.W.shape)
        grad_b = np.zeros(self.b.shape)

        b_try = np.copy(self.b)
        for i in range(len(self.b)):
            self.b = b_try
            self.b[i] = self.b[i] + h
            c1 = self.compute_cost(X_batch, Y_batch, lamda)

            self.b = b_try
            self.b[i] = self.b[i] - 2 * h
            c2 = self.compute_cost(X_batch, Y_batch, lamda)
            grad_b[i] = (c1 - c2) / (2 * h)

        W_try = np.copy(self.W)
        for i in np.ndindex(self.W.shape):
            self.W = W_try
            self.W[i] = self.W[i] + h
            c1 = self.compute_cost(X_batch, Y_batch, lamda)

            self.W = W_try
            self.W[i] = self.W[i] - 2 * h
            c2 = self.compute_cost(X_batch, Y_batch, lamda)
            grad_W[i] = (c1 - c2) / (2 * h)

        return grad_W, grad_b

    def plot_loss(self, loss, loss_val, n_epochs):
        epochs = np.arange(n_epochs)
        plt.plot(epochs, loss, label="Training data")
        plt.plot(epochs, loss_val, label="Validation data")
        plt.ylabel("Costs")
        plt.xlabel("Epochs")
        plt.legend()
        plt.savefig(f"training_validation_error")
        plt.show()

    def visualize_weights(self):
        """Visualize the weight of the matrix """

        for i, row in enumerate(self.W):
            img = (row - row.min()) / (row.max() - row.min())

            plt.subplot(2, 5, i + 1)
            img = np.rot90(np.reshape(img, (32, 32, 3), order="F"), k=3)
            plt.imshow(img)
            plt.axis("off")
            plt.title(CIDAR_LABELS[i])

        plt.savefig("weights.png")
        plt.show()

    def shuffle(self, X, Y):
        pem = np.random.permutation(Y.shape[1])
        return X[:, pem], Y[:, pem]

    def fit(self, X, Y, n_epochs, n_batch=32, eta=0.01, lamda=0):

        loss = np.zeros(n_epochs)
        loss_val = np.zeros(n_epochs)

        for epoch in range(n_epochs):
            print(f"\n Epoch {epoch + 1} n_batch {n_batch}, eta {eta} lamda {lamda} ")

            Xs, Ys = (X, Y)

            for batch in range(n_batch):
                N = int(self.X.shape[1] / n_batch)
                start = (batch) * N
                end = (batch + 1) * N

                X_batch = Xs[:, start:end]
                Y_batch = Ys[:, start:end]

                gradient_W, gradient_b = self.compute_gradients_ana(X_batch, Y_batch, lamda)

                self.W = self.W - eta * gradient_W
                self.b = self.b - eta * gradient_b

            loss[epoch] = self.compute_cost(X, Y, lamda)
            loss_val[epoch] = self.compute_cost(self.X_val, self.Y_val, lamda)

        print(f"Accuracy for training data {self.compute_accuracy(X, self.y)}")
        print(f"Accuracy for validation data {self.compute_accuracy(self.X_val, self.y_val)}")
        print(f"Accuracy for test data {self.compute_accuracy(self.X_test, self.Y_test)}")
        print(self.visualize_weights())
        self.plot_loss(loss, loss_val, n_epochs)


def main():
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
    network.fit(network.X, network.Y, n_batch=100, eta=0.1, n_epochs=40, lamda=0)


if __name__ == "__main__":
    main()
