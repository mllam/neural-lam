import json

class Logger:
    def __init__(self):
        self.logs = []

    def log(self, epoch, loss):
        self.logs.append({
            "epoch": epoch,
            "loss": loss
        })

    def save(self, path="logs.json"):
        with open(path, "w") as f:
            json.dump(self.logs, f, indent=4)