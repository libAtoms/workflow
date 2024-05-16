import setuptools

setuptools.setup(
    name="wfl",
    version="0.2.5",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(exclude=["tests"]),
    install_requires=["click>=7.0", "numpy", "ase>=3.22.1", "pyyaml", "spglib", "docstring_parser",
                      "expyre-wfl", "universalSOAP"],
    entry_points="""
    [console_scripts]
    wfl=wfl.cli.cli:cli
    gap_rss_iter_fit=wfl.cli.gap_rss_iter_fit:cli
    dft_convergence_test=wfl.cli.dft_convergence_test:cli
    reactions_iter_fit=wfl.cli.reactions_iter_fit:cli
    """
)
