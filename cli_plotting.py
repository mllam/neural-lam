# Third-party
import click

# First-party
from neural_lam.vis import verify_inference


@click.command()
@click.argument("file_path", type=click.Path(exists=True), required=True)
@click.argument("save_path", type=click.Path(), required=True)
@click.option(
    "--feature_channel",
    "-f",
    default=0,
    help="Feature channel to use. Default is 0.",
    type=int,
    show_default=True,
)
def main(file_path: str, save_path: str, feature_channel: int) -> None:
    """
    Command line tool for verifying neural_lam inference results.
    """
    verify_inference(file_path, save_path, feature_channel)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
