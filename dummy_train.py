import numpy as np
import wandb
import time


run = wandb.init(project= "dummy-training", 
                 comment= "testing wandb logging",
                 mode= "online",
                 config= {
                     "epochs": 10
                 })

def train(epochs= 5):
    for epoch in range(epochs):
        # renerate radom loss values
        loss = np.random.rand()
        
        wandb.log({"epoch": epoch, "loss": loss})


if __name__ == "__main__":
    epochs = run.config.epochs
    train(epochs)
    wandb.finish()