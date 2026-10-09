"""Standalone script to upgrade a checkpoint's state dict in place.

Wraps the same `neural_lam.migrations.apply_migrations` used by
`ForecasterModule.on_load_checkpoint`, so a checkpoint can be migrated
once and re-saved instead of being migrated in memory on every load. See
issue #48 for the motivation.
"""

# Standard library
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from typing import Optional

# Third-party
import torch

# Local
from . import migrations


def migrate_checkpoint_file(load_path: str, save_path: str) -> None:
    """
    Migrate the checkpoint at `load_path` to the current state dict schema
    version and save the result to `save_path`.

    Parameters
    ----------
    load_path : str
        Path to the checkpoint file to migrate.
    save_path : str
        Path to save the migrated checkpoint to.
    """
    checkpoint = torch.load(load_path, map_location="cpu", weights_only=False)
    checkpoint_version = checkpoint.get("neural_lam_checkpoint_version", 0)

    if checkpoint_version >= migrations.CURRENT_CHECKPOINT_VERSION:
        print(
            f"Checkpoint is already at version {checkpoint_version}, "
            "nothing to migrate."
        )
        return

    checkpoint["state_dict"], checkpoint_version = migrations.apply_migrations(
        checkpoint["state_dict"], checkpoint_version
    )
    checkpoint["neural_lam_checkpoint_version"] = checkpoint_version

    torch.save(checkpoint, save_path)
    print(
        f"Migrated checkpoint to version {checkpoint_version}, "
        f"saved to {save_path}"
    )


def cli(input_args: Optional[list[str]] = None) -> None:
    """
    Parse CLI arguments and call `migrate_checkpoint_file`.

    Parameters
    ----------
    input_args : list[str] or None, optional
        Argument list forwarded to :class:`argparse.ArgumentParser`. When
        ``None``, ``sys.argv`` is used.
    """
    parser = ArgumentParser(
        description="Upgrade a neural-lam checkpoint's state dict",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--load",
        type=str,
        required=True,
        help="Path to checkpoint file to migrate",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Path to save migrated checkpoint to. Defaults to "
        "'upgraded_<load file name>' next to the input file.",
    )
    args = parser.parse_args(input_args)

    save_path = args.save
    if save_path is None:
        load_dirname, load_basename = os.path.split(args.load)
        save_path = os.path.join(load_dirname, f"upgraded_{load_basename}")

    migrate_checkpoint_file(load_path=args.load, save_path=save_path)


if __name__ == "__main__":
    cli()
