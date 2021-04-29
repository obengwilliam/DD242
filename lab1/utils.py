"""Utils for building network."""
import numpy as np
import pickle
import matplotlib.pyplot as plt
import scipy.io as sio


def softmax(x):
    """Apply Softmax to x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def LoadBatch(filename: str) -> dict:
    """Copy from the dataset website."""
    with open("dataset/" + filename, "rb") as file:
        dict = pickle.load(file, encoding="bytes")
    return dict


def montage(W):
    """Display the image for each label in W."""
    _, ax = plt.subplots(2, 5)
    for i in range(2):
        for j in range(5):
            im = W[i * 5 + j, :].reshape(32, 32, 3, order="F")
            sim = (im - np.min(im[:])) / (np.max(im[:]) - np.min(im[:]))
            sim = sim.transpose(1, 0, 2)
            ax[i][j].imshow(sim, interpolation="nearest")
            ax[i][j].set_title("y=" + str(5 * i + j))
            ax[i][j].axis("off")
    plt.show()
