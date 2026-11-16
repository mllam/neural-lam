# Transfer Learning Compatibility Metadata Design

This is an internal design note for maintainers reviewing transfer-learning
checkpoint compatibility. It is not part of the user-facing documentation
navigation.

Neural-LAM can initialize selected model components from a trusted Lightning
checkpoint while leaving the rest of a target model newly initialized. This
page describes the loading lifecycle and the compatibility metadata used for
graph-based models.

## Selective Initialization

A component is identified by its nested module path. The path joins attribute
names with periods. For example:

```text
forecaster.predictor.processor
```

identifies the `processor` inside `predictor`, inside `forecaster`.

During selective initialization, Neural-LAM:

1. Constructs the complete target model from the target configuration and
   datastore.
2. Loads the source checkpoint through Lightning's `load_from_checkpoint()`.
3. Applies the existing legacy checkpoint key migrations.
4. Validates compatibility metadata for graph-based models.
5. Validates every selected component's state keys, value types and tensor
   shapes.
6. Replaces the selected entries in the fresh target state with source values.
7. Lets Lightning perform one strict load of the resulting complete state.
8. Starts `Trainer.fit()` without a checkpoint path, producing a fresh
   optimizer, epoch, global step and trainer loop.

This differs from `--load`, which passes a checkpoint to `Trainer.fit()` and
resumes complete training state.

## Why Compatibility Metadata Is Needed

Graph topology and static graph features are registered as non-persistent
buffers. They control graph computation but are intentionally absent from the
ordinary model `state_dict`. Parameter names and shapes can therefore match
even when two models use different graphs.

To avoid silently transferring graph-trained weights onto an unverified graph,
graph-based predictors expose compatibility metadata. The forecasting module
stores that metadata in every new Lightning checkpoint under:

```python
checkpoint["neural_lam_pretrained_compatibility"]
```

For a GraphLAM checkpoint, its structure is conceptually:

```python
{
    "forecaster.predictor": {
        "graph_v1": "<sha256 fingerprint>",
    }
}
```

The outer key is the nested path of the component that provides the metadata.
`graph_v1` identifies the fingerprint schema so it can evolve explicitly.

## Graph Fingerprint

`BaseGraphModel` computes one SHA-256 fingerprint from the grid-node count and
the registered graph tensors. For every tensor, the hash includes its attribute
name, position within a tensor collection, dtype, shape and exact bytes.

The registered tensors cover:

- Grid-to-mesh, mesh-to-grid and mesh-to-mesh edge indices.
- Hierarchical mesh-up and mesh-down edge indices when present.
- Grid-to-mesh, mesh-to-grid and mesh-to-mesh edge features.
- Hierarchical mesh-up and mesh-down edge features when present.
- Static mesh-node features.

The fingerprint is cached after its first computation because graph tensors are
immutable model structure. Saving later checkpoints does not repeatedly copy
and hash the graph.

## What The Check Proves

Matching fingerprints establish exact equality of the registered graph tensors
and grid-node count included in the fingerprint. This checks the graph
structure, node indexing and stored graph-feature values used by Neural-LAM.

The fingerprint does not establish compatibility of information outside those
tensors, including:

- State, forcing or static dataset variable semantics.
- Variable ordering and units.
- Normalization statistics.
- Unregistered external metadata.
- The scientific usefulness of transferred representations.

Selected components are checked separately for exact state keys, state value
types and tensor shapes. Users remain responsible for ensuring that dataset and
feature semantics are appropriate for their experiment.

## Current Compatibility Policy

The current feature scope requires the same complete graph representation,
even if a selected component could theoretically be useful across different
graphs. Component-specific graph compatibility and cross-topology transfer are
out of scope.

The loading policy is:

- Matching fingerprint: selective loading may continue.
- Stored fingerprint mismatch: loading always fails and cannot be overridden.
- Missing metadata: loading fails by default.
- Missing metadata with `--allow_missing_compatibility_metadata`: loading emits
  a warning and continues, making the caller responsible for manually checking
  graph compatibility.

The override exists for trusted checkpoints created before compatibility
metadata was introduced. It never bypasses a known mismatch.

Compatibility validation only runs during selective initialization. Full
checkpoint loading and normal `--load` resume remain available for older
checkpoints because they reconstruct the complete saved model rather than
combining source and target models.

## Usage

Transfer GraphLAM's processor and mesh embedders into a fresh target model:

```bash
python -m neural_lam.train_model \
    --config_path <target-config> \
    --model graph_lam \
    --graph <target-graph> \
    --pretrained_checkpoint <source-checkpoint> \
    --pretrained_components \
        forecaster.predictor.processor \
        forecaster.predictor.mesh_embedder \
        forecaster.predictor.m2m_embedder
```

Freeze the transferred processor parameters:

```bash
--freeze_components forecaster.predictor.processor
```

Use a trusted legacy checkpoint only after manually confirming its graph:

```bash
--allow_missing_compatibility_metadata
```

Freezing controls `requires_grad` for the current run and is not stored in a
checkpoint state dictionary. Repeat `--freeze_components` when resuming a run
whose parameters should remain frozen.

## Extension Points

Selective state loading is model-independent and uses nested module paths plus
`torch.nn.Module.get_submodule()`. A model component can provide additional
compatibility metadata by implementing
`get_pretrained_compatibility_metadata()`. The forecasting module collects such
metadata by module path and compares it during selective initialization.

GraphLAM is the architecture covered by the current transfer tests. Other
architectures require architecture-specific tests before being documented as
supported transfer scenarios.
