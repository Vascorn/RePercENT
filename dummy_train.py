import numpy as np
import torch
import random

def train(epochs= 5):
    for epoch in range(epochs):
        # renerate radom loss values
        loss = np.random.rand()
        print(f"Epoch {epoch + 1} / {epochs}, Loss: {loss:.4f}")


if __name__ == "__main__":
    train()