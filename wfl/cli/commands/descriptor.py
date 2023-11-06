import click
from wfl.cli import cli_options as opt
import wfl.descriptors.quippy

@click.command("quippy")
@click.pass_context
@click.option("--local", is_flag=True, help="calculate a local (per-atom) descriptor")
@click.option("--force", is_flag=True, help="overwrite existing info or arrays item if present")
@click.option("--descriptor", type=click.STRING, required=True, help="quippy.descriptors.Descriptor arg string")
@click.option("--key", required=True, type=click.STRING, help="Atoms.info (global) or Atoms.arrays (local) for descriptor vector")
@opt.inputs
@opt.outputs
def quippy(ctx, inputs, outputs, descriptor, key, local, force):
    calculate_descriptor(inputs, outputs, descriptor, key, local, force)

def calculate_descriptor(inputs, outputs, descriptor, key, local, force):
    wfl.descriptors.quippy.calculate(
        inputs=inputs,
        outputs=outputs,
        descs=descriptor,
        key=key,
        per_atom=local,
        force=force
    )
