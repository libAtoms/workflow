import setuptools

setuptools.setup(
    name="wfl",
    version="0.1.0b",
    packages=setuptools.find_packages(exclude=["test"]),
    install_requires=["click>=7.0", "numpy", "ase", "pyyaml", "spglib",
                      "sphinx-book-theme", "docutils==0.16",
                      "expyre-wfl @ https://github.com/libAtoms/ExPyRe/tarball/main",
                      "universalSOAP @ https://github.com/libAtoms/universalSOAP/tarball/main"],
    entry_points="""
    [console_scripts]
    wfl=wfl.cli.cli:cli
    gap_rss_iter_fit=wfl.cli.gap_rss_iter_fit:cli
    dft_convergence_test=wfl.cli.dft_convergence_test:cli
    reactions_iter_fit=wfl.cli.reactions_iter_fit:cli
    xpr=wfl.expyre.cli.cli:cli
    """
)
