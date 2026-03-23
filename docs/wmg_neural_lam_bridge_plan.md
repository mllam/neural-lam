# weather-model-graphs <-> neural-lam bridge plan (issue #384)

This document captures the agreed implementation direction from issue #384 and
its discussion thread.

## Goal

Bridge graph creation between `weather-model-graphs` (WMG) and `neural-lam`
without changing the current tensor-on-disk contract expected by
`neural_lam.utils.load_graph`.

## Agreed architecture

1. Graph construction: handled by WMG archetype constructors.
2. Serialization: handled by `wmg.save.to_neural_lam(...)`.
3. Execution interface: `python -m neural_lam.create_graph` orchestrates
   datastore loading, WMG graph creation call, and save location selection.

## Scope implemented in neural-lam

1. `neural_lam.create_graph` supports archetype-oriented CLI arguments:
   - `--archetype keisler|graphcast|hierarchical`
   - `--graph_name`
   - `--output_dir`
   - `--mesh_node_distance`
   - `--max_num_levels`
   - `--level_refinement_factor`
2. Backend selection is explicit with `--backend auto|wmg|legacy`:
   - `auto`: use WMG if available, otherwise fallback to legacy generator.
   - `wmg`: require WMG and fail if unavailable.
   - `legacy`: always use existing in-repo graph generation.
3. Existing flags remain supported for compatibility:
   - `--name` (alias of `--graph_name`)
   - `--levels` (alias of `--max_num_levels`)
   - `--hierarchical` (alias mapping to archetype `hierarchical`)
4. Default output remains under `graph/<graph_name>` in datastore root to stay
   compatible with training/evaluation code paths.

## Upstream WMG dependency work

The WMG side still needs a tracked issue/PR that guarantees
`wmg.save.to_neural_lam(...)` emits the exact tensor-on-disk file contract used
by `neural-lam`.

Planned upstream deliverables:

1. Tensor-on-disk writer in WMG with parity tests.
2. Validation hooks so both repos can run the same format checks.
3. Explicit behavior for flat (`keisler` / `graphcast`) and hierarchical
   outputs.

## Future work (deferred)

1. Finalize tensor-on-disk spec details and validator ownership across repos.
2. Decide normalization contract fully:
   - Current neural-lam behavior assumes raw edge features written to disk and
     normalization in `load_graph`.
3. Add optional format metadata/version tagging to support non-breaking format
   evolution.
4. Add optional node-id map outputs to support latent-space visualization and
   round-trip debugging.
5. Extend CLI + serializer path to global/spherical topologies (tracked
   separately, e.g. icosahedral support).
6. Add end-to-end integration tests across repos:
   - WMG archetype -> to_neural_lam -> neural-lam loader -> model forward.
