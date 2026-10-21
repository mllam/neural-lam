---
name: smoke-train
description: Run an end-to-end offline training and evaluation roundtrip on DummyDatastore using the programmatic API. Use to verify pipeline integrity after changes.
---

# Smoke Train and Evaluate

Follow these instructions to run a hermetic, fully offline smoke test verifying graph creation, training, checkpoint generation, and evaluation without network access or GPU requirements.

## 1. Principles of Offline Verification
The smoke-train workflow validates the entire Neural-LAM pipeline under strict constraints:
- **Zero network / S3 access**: Uses in-memory synthetic data from `DummyDatastore` instead of production datasets that download from remote storage.
- **Offline logger configuration**: Sets `WANDB_MODE=offline` to prevent external authentication requirements or remote sync attempts during test runs.
- **In-process datastore registration**: Explicitly maps `DATASTORES[DummyDatastore.SHORT_NAME] = DummyDatastore` so execution succeeds outside pytest harness fixtures.
- **Hermetic directories**: Uses a temporary directory (`tempfile.TemporaryDirectory()`) for both graph generation and training runs, preventing filesystem pollution.
- **Actual checkpoint reloading**: Verifies that the training run writes a checkpoint file to disk, and the evaluation step loads *that specific newly produced checkpoint* (rather than relying on pre-existing artifacts).
- **Minimal evaluation side effects**: Disables example prediction saving and GIF rendering (`n_example_pred=0`) so evaluation executes fast without extraneous file generation.
- **CPU compatibility**: Fully runnable on CPU via `CUDA_VISIBLE_DEVICES=""`.

## 2. In-Process Python Workflow
Run this hermetic test script:

```python
import os
from pathlib import Path
import tempfile
from neural_lam import api
from neural_lam.config import DatastoreSelection, NeuralLAMConfig
from neural_lam.datastore import DATASTORES
from tests.dummy_datastore import DummyDatastore

# Configure offline execution and register in-memory test datastore
os.environ["WANDB_MODE"] = "offline"
DATASTORES[DummyDatastore.SHORT_NAME] = DummyDatastore

with tempfile.TemporaryDirectory() as tmp_dir:
    tmp_path = Path(tmp_dir)

    # 1. Initialize DummyDatastore (in-memory, no downloads)
    datastore = DummyDatastore(n_grid_points=100, n_timesteps=15)
    config = NeuralLAMConfig(
        datastores={
            "main": DatastoreSelection(
                kind=datastore.SHORT_NAME,
                config_path=str(datastore.root_path),
            )
        }
    )

    # 2. Create Graph Mesh in temporary directory
    api.create_graph(
        datastore=datastore,
        name="1level",
        archetype="keisler",
    )

    # 3. Train 1 Epoch
    runs_root = tmp_path / "runs"
    train_run = api.train(
        datastore=datastore,
        config=config,
        model="graph_lam",
        graph="1level",
        epochs=1,
        ar_steps_train=1,
        val_interval=1,
        num_sanity_val_steps=0,
        batch_size=1,
        num_workers=0,
        hidden_dim=4,
        processor_layers=1,
        hidden_layers=1,
        runs_root=str(runs_root),
        logger_run_name="smoke_train",
        val_steps_to_log=[1],
    )

    # Ensure training produced a real checkpoint
    assert train_run.checkpoint_path is not None
    assert train_run.checkpoint_path.exists()

    # 4. Evaluate From the Newly Produced Checkpoint
    eval_run = api.evaluate(
        datastore=datastore,
        config=config,
        load=str(train_run.checkpoint_path),
        model="graph_lam",
        graph="1level",
        hidden_dim=4,
        processor_layers=1,
        hidden_layers=1,
        batch_size=1,
        num_workers=0,
        ar_steps_eval=2,
        val_steps_to_log=[1],
        n_example_pred=0,
        runs_root=str(runs_root),
        logger_run_name="smoke_eval",
    )

    assert eval_run.run_dir.exists()
```

## 3. Verification Check
Execute the smoke-train test in the test suite:

```bash
CUDA_VISIBLE_DEVICES="" uv run pytest tests/test_skills.py -k test_smoke_train_workflow_execution
```
