from .HAWC2IO import inspect as _inspect
import click


@click.command()
@click.argument("filename", type=click.Path(exists=True))
def inspect(filename):
    _inspect(filename)
