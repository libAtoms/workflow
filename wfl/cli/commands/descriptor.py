import click
from wfl.cli import cli_options as opt

@click.command("quippy")
@click.pass_context
@click.option("--local", is_flag=True, help="calculate a local (per-atom) descriptor")
@click.option("--force", is_flag=True, help="overwrite existing info or arrays item if present")
@opt.inputs
@opt.outputs
@opt.descriptor
@opt.key
def quippy(ctx, inputs, outputs, descriptor, key, local, force):
    calculate_descriptor(inputs, outputs, descriptor, key, local, force)

def calculate_descriptor(inputs, outputs, descriptor, key, local, force):
    wfl.descriptors.quippy.calc(
        inputs=inputs,
        outputs=outputs,
        descs=descriptor,
        key=key,
        local=local,
        force=force
    )
