import torch

class Trainer:
    def __init__(self, model, optimizer, loss_fn, config):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.epochs = config.get("epochs", 10)

    def train(self, dataloader):
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0

            for batch in dataloader:
                inputs, targets = batch

                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")