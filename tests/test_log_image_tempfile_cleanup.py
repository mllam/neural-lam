import os
import glob
import matplotlib.pyplot as plt
import pytest
from neural_lam.custom_loggers import CustomMLFlowLogger

class DummyMLflow:
    @staticmethod
    def log_image(img, name):
        # Simulate logging, do nothing
        pass

def test_log_image_tempfile_cleanup(monkeypatch):
    # Patch mlflow in the logger to use dummy
    monkeypatch.setattr("neural_lam.custom_loggers.mlflow", DummyMLflow)

    logger = CustomMLFlowLogger("exp", "uri", "run")
    key = "test_image"
    # Create a dummy matplotlib figure
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    images = [fig]

    # List all .png files before
    before_pngs = set(glob.glob("*.png"))

    logger.log_image(key, images)

    # List all .png files after
    after_pngs = set(glob.glob("*.png"))

    # The set of new files should be empty (all temp files cleaned up)
    assert before_pngs == after_pngs, "Temporary .png files were not cleaned up!"

    plt.close(fig)
