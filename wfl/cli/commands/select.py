import click
import numpy as np
from wfl.cli import cli_options as opt
import wfl.descriptors.quippy
import wfl.select.by_descriptor
from wfl.select.simple import by_bool_func


@click.command("cur")
@click.option("--n-configs", "-N", type=click.INT, required=True,
              help="number of configs to select")
@click.option("--keep_descriptor", is_flag=True, help="keep the descriptor value in the final config file")
@click.option("--kernel_exponent", type=click.FLOAT, help="exponent of dot-product for kernel")
@click.option("--deterministic", is_flag=True, help="use deterministic (not stochastic) CUR selection")
@click.option("--key", required=True, type=click.STRING, help="Atoms.info (global) or Atoms.arrays (local) for descriptor vector")
@click.option("--stochastic-seed", type=click.INT, help="seed for `np.random.default_rng()` in stochastic CUR.")
@click.pass_context
@opt.inputs
@opt.outputs
def cur(ctx, inputs, outputs, n_configs, key, keep_descriptor,
                kernel_exponent, deterministic, stochastic_seed):
    """Select structures by CUR"""

    wfl.select.by_descriptor.CUR_conf_global(
        inputs=inputs,
        outputs=outputs,
        num=n_configs,
        at_descs_info_key=key, kernel_exp=kernel_exponent, stochastic=not deterministic,
        keep_descriptor_info=keep_descriptor,
        rng=np.random.default_rng(stochastic_seed))


@click.command("lambda")
@click.option("--exec-code", "-e", required=True,
help='String to be evaluated by the lambda function. Will be substituted into `eval(\"lambda atoms: \" + exec_code)`.')
@click.pass_context
@opt.inputs
@opt.outputs
def by_lambda(ctx, inputs, outputs, exec_code):
    """selects atoms based on a lambda function"""

    at_filter_fun = eval("lambda atoms: " + exec_code)

    by_bool_func(
        inputs=inputs,
        outputs=outputs,
        at_filter=at_filter_fun)
