import numpy as np
import wandb
import time

def train(epochs= 5):
    for epoch in range(epochs):
        # renerate radom loss values
        loss = np.random.rand()
        # wandb.log({"loss": loss})
        # pause for 30 seconds
        time.sleep(30)
        print(f"Epoch {epoch + 1} / {epochs}, Loss: {loss:.4f}")


if __name__ == "__main__":
    train()