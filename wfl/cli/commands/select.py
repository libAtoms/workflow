from pathlib import Path
import click
from wfl.cli import cli_options as opt
from wfl.configset import ConfigSet, OutputSpec
import wfl.descriptors.quippy
import wfl.select.by_descriptor
from .descriptor import calculate_descriptor
from wfl.select.simple import by_bool_func

@click.command("cur")
@click.option("--n-configs", "-N", type=click.INT, required=True,
              help="number of configs to select")
@click.option("--keep_descriptor", is_flag=True, help="keep the descriptor value in the final config file")
@click.option("--kernel_exponent", type=click.FLOAT, help="exponent of dot-product for kernel")
@click.option("--deterministic", is_flag=True, help="use deterministic (not stochastic) CUR selection")
@click.option("--key", required=True, type=click.STRING, help="Atoms.info (global) or Atoms.arrays (local) for descriptor vector")
@click.pass_context
@opt.inputs
@opt.outputs
def cur(ctx, inputs, outputs, n_configs, key, keep_descriptor,
                kernel_exponent, deterministic):
    """Select structures by CUR"""    

    wfl.select.by_descriptor.CUR_conf_global(
        inputs=inputs,
        outputs=outputs,
        num=n_configs,
        at_descs_info_key=key, kernel_exp=kernel_exponent, stochastic=not deterministic,
        keep_descriptor_info=keep_descriptor)


@click.command("lambda")
@click.option("--exec-code", "-e", required=True, 
help='String to be evaluated by the lambda function. Will be substituted into `eval(\"lambda atoms: \" + exec_code)`.')
@click.pass_context
@opt.inputs
@opt.outputs
def by_lambda(ctx, inputs, outputs, exec_code):
    """selects atoms based on a lambda function"""

    if outputs.done():
        print(f'Not filtering with a lambda function, because {outputs} are done.')
        return
    
    for at in inputs: 
        fun = eval("lambda atoms: " + exec_code) 
        if fun(at):
            outputs.store(at)
    outputs.close()
    