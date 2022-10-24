from pathlib import Path
import sys

import nbformat
from nbformat import v4 as nbf
from nbformat.v4 import convert as nbf_converter

import typer


def change_renderer(input_file: Path, renderer: str = typer.Option("notebook_connected", "--renderer")):
    nb = nbformat.read(input_file, as_version=nbformat.NO_CONVERT)

    nb = nbf_converter.upgrade(nb)
    cell = nbf.new_code_cell(f'import plotly.io as pio; pio.renderers.default = "{renderer}"')

    nb.cells.insert(0, cell)

    nbformat.write(nb, sys.stdout, version=nbformat.NO_CONVERT)


if __name__ == "__main__":
    typer.run(change_renderer)