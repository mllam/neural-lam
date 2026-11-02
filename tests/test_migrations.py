# Third-party
import pytest
import torch

# First-party
from neural_lam import migrations
from neural_lam.migrate_checkpoint import migrate_checkpoint_file


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
    with pytest.raises(ValueError, match="Migration must set exactly one of"):
        migrations.Migration(target_version=1, description="neither")

    with pytest.raises(ValueError, match="Migration must set exactly one of"):
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
    }


def test_detect_checkpoint_version_uses_explicit_tag():
    assert (
        migrations.detect_checkpoint_version(
            {"neural_lam_checkpoint_version": 2}
        )
        == 2
    )
    assert (
        migrations.detect_checkpoint_version(
            {"neural_lam_checkpoint_version": "3"}
        )
        == 3
    )


def test_detect_checkpoint_version_infers_v0_for_untagged_flat_layout():
    checkpoint = {
        "state_dict": {
            "g2m_gnn.grid_mlp.0.weight": 1,
            "interior_mask_bool": 2,
        }
    }
    assert migrations.detect_checkpoint_version(checkpoint) == 0


def test_detect_checkpoint_version_infers_v2_for_untagged_modular_layout():
    checkpoint = {
        "state_dict": {
            "forecaster.predictor.encoding_grid_mlp.0.weight": 1,
            "interior_mask_bool": 2,
        }
    }
    assert migrations.detect_checkpoint_version(checkpoint) == 2


def test_grid_mlp_migration_handles_prefixed_keys():
    """Ensure that checkpoints saved with forecaster.predictor prefix
    still have g2m_gnn.grid_mlp renamed to encoding_grid_mlp."""
    state_dict = {
        "forecaster.predictor.g2m_gnn.grid_mlp.0.weight": "w0",
        "forecaster.predictor.other.weight": "w1",
    }
    migrated, _ = migrations.apply_migrations(state_dict, from_version=0)
    assert "forecaster.predictor.encoding_grid_mlp.0.weight" in migrated
    assert "forecaster.predictor.g2m_gnn.grid_mlp.0.weight" not in migrated


def test_migrate_checkpoint_file_upgrades_file_on_disk(tmp_path):
    """Test standalone migrate_checkpoint_file upgrades a file on disk."""
    load_path = tmp_path / "legacy.ckpt"
    save_path = tmp_path / "migrated.ckpt"

    legacy_ckpt = {
        "state_dict": {
            "g2m_gnn.grid_mlp.0.weight": torch.tensor([1.0]),
            "interior_mask_bool": torch.tensor([True]),
        }
    }
    torch.save(legacy_ckpt, load_path)

    migrate_checkpoint_file(str(load_path), str(save_path))

    migrated_ckpt = torch.load(
        save_path, map_location="cpu", weights_only=False
    )
    assert (
        migrated_ckpt["neural_lam_checkpoint_version"]
        == migrations.CURRENT_CHECKPOINT_VERSION
    )
    assert (
        "forecaster.predictor.encoding_grid_mlp.0.weight"
        in migrated_ckpt["state_dict"]
    )
    assert "interior_mask_bool" not in migrated_ckpt["state_dict"]


def test_migrate_checkpoint_file_noop_when_already_current(tmp_path, capsys):
    """Test standalone migrate_checkpoint_file is a no-op if already current."""
    load_path = tmp_path / "current.ckpt"
    save_path = tmp_path / "output.ckpt"

    current_ckpt = {
        "neural_lam_checkpoint_version": migrations.CURRENT_CHECKPOINT_VERSION,
        "state_dict": {},
    }
    torch.save(current_ckpt, load_path)

    migrate_checkpoint_file(str(load_path), str(save_path))

    captured = capsys.readouterr()
    assert "nothing to migrate" in captured.out
    assert not save_path.exists()
