from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import click

from pathseg_fewshot.datasets.episode_sampler import (
    EpisodeSpec,
    MinImagesConsumingEpisodeSampler,
)


@click.command()
@click.option(
    "--tile-index-parquet",
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
@click.option(
    "--output-json",
    type=click.Path(path_type=Path),
    required=True,
)
@click.option("--ways", type=int, required=True, help="Number of ways for the episode.")
@click.option("--queries", type=int, required=True, help="Number of queries for the episode.")
@click.option(
    "--shots", type=int, required=True, help="Number of shots for the episode."
)
@click.option(
    "--min-class-pixels",
    type=int,
    default=500,
    show_default=True,
    help="Keep (tile,class) rows only if class_pixels >= this.",
)
@click.option(
    "--datasets",
    type=str,
    default="bcss,ignite",
    show_default=True,
    help="Comma-separated list of dataset IDs to include.",
)
def main(
    tile_index_parquet: Path,
    output_json: Path,
    ways: int,
    shots: int,
    queries: int,
    min_class_pixels: int,
    datasets: str,
) -> None:
    """
    Build a few-shot episode list for validation.
    """
    click.echo("Done.")


if __name__ == "__main__":
    main()
