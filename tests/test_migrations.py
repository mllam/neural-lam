# Third-party
import pytest

# First-party
from neural_lam import migrations


def test_rename_keys_renames_matching_prefix_only():
    rename = migrations.rename_keys("old.prefix", "new.prefix")
    state_dict = {
        "old.prefix.weight": 1,
        "old.prefix.nested.bias": 2,
        "unrelated.old.prefix.weight": 3,
        "other.weight": 4,
    }

    result = rename(state_dict)

    assert result == {
        "new.prefix.weight": 1,
        "new.prefix.nested.bias": 2,
        "unrelated.old.prefix.weight": 3,
        "other.weight": 4,
    }


def test_remove_keys_drops_present_keys_and_ignores_missing():
    remove = migrations.remove_keys(["a", "missing"])
    state_dict = {"a": 1, "b": 2}

    result = remove(state_dict)

    assert result == {"b": 2}


def test_migration_requires_exactly_one_of_apply_or_hard_break():
    with pytest.raises(ValueError):
        migrations.Migration(target_version=1, description="neither")

    with pytest.raises(ValueError):
        migrations.Migration(
            target_version=1,
            description="both",
            apply=lambda sd: sd,
            hard_break=True,
        )


def test_apply_migrations_runs_only_migrations_above_from_version():
    fake_registry = [
        migrations.Migration(
            target_version=1,
            description="add a",
            apply=lambda sd: {**sd, "a": 1},
        ),
        migrations.Migration(
            target_version=2,
            description="add b",
            apply=lambda sd: {**sd, "b": 2},
        ),
    ]
    original = migrations.MIGRATIONS
    migrations.MIGRATIONS = fake_registry
    try:
        state_dict, reached_version = migrations.apply_migrations(
            {}, from_version=1
        )
        assert state_dict == {"b": 2}
        assert reached_version == 2
    finally:
        migrations.MIGRATIONS = original


def test_apply_migrations_is_a_noop_when_already_current():
    state_dict = {"x": 1}
    result, reached_version = migrations.apply_migrations(
        state_dict, from_version=migrations.CURRENT_CHECKPOINT_VERSION
    )
    assert result == {"x": 1}
    assert reached_version == migrations.CURRENT_CHECKPOINT_VERSION


def test_apply_migrations_raises_clear_error_on_hard_break():
    fake_registry = [
        migrations.Migration(
            target_version=1, description="cannot be bridged", hard_break=True
        ),
    ]
    original = migrations.MIGRATIONS
    migrations.MIGRATIONS = fake_registry
    try:
        with pytest.raises(RuntimeError, match="cannot be bridged"):
            migrations.apply_migrations({}, from_version=0)
    finally:
        migrations.MIGRATIONS = original


def test_pre_refactor_flat_checkpoint_migrates_to_current_layout():
    """Regression test for the two real migrations this registry replaced
    in `ForecasterModule.on_load_checkpoint` (see #48)."""
    legacy_state_dict = {
        "g2m_gnn.grid_mlp.0.weight": "w0",
        "g2m_gnn.grid_mlp.0.bias": "b0",
        "some_other_layer.weight": "w1",
        "interior_mask_bool": "mask",
        "per_var_std": "std",
    }

    state_dict, reached_version = migrations.apply_migrations(
        legacy_state_dict, from_version=0
    )

    assert reached_version == migrations.CURRENT_CHECKPOINT_VERSION
    assert state_dict == {
        "forecaster.predictor.encoding_grid_mlp.0.weight": "w0",
        "forecaster.predictor.encoding_grid_mlp.0.bias": "b0",
        "forecaster.predictor.some_other_layer.weight": "w1",
        "interior_mask_bool": "mask",
        "per_var_std": "std",
    }
