from pathlib import Path
import click
from wfl.cli import cli_options as opt
from wfl.configset import ConfigSet, OutputSpec
import wfl.descriptors.quippy
import wfl.select.by_descriptor
from .descriptor import calculate_descriptor

@click.command("cur")
@click.option("--n-configs", "-N", type=click.INT, required=True,
              help="number of configs to select")
@click.option("--keep_descriptor", is_flag=True, help="keep the descriptor value in the final config file")
@click.option("--kernel_exponent", type=click.FLOAT, help="exponent of dot-product for kernel")
@click.option("--deterministic", is_flag=True, help="use deterministic (not stochastic) CUR selection")
@click.pass_context
@opt.inputs
@opt.outputs
@opt.descriptor
@opt.key
def cur(ctx, inputs, outputs, n_configs, key, descriptor, keep_descriptor,
                kernel_exponent, deterministic):
    """Select structures by CUR"""    

    if descriptor is None:
        if descriptor_key is None:
            raise RuntimeError('CUR-global needs --descriptor or --descriptor-key')
    clean_tmp_files = False
    if descriptor is not None:
        # calculate descriptor
        if key is None:
            key = '_CUR_desc'
        calculate_descriptor(inputs, OutputSpec('_tmp_desc.xyz'), descriptor, key, local=False, force=True)
        inputs = ConfigSet(['_tmp_desc.xyz'])
        clean_tmp_files = True

    wfl.select.by_descriptor.CUR_conf_global(
        inputs=inputs,
        outputs=outputs,
        num=n_configs,
        at_descs_info_key=key, kernel_exp=kernel_exponent, stochastic=not deterministic,
        keep_descriptor_info=keep_descriptor)

    if clean_tmp_files:
        for input_file in inputs:
            Path(input_file).unlink()



