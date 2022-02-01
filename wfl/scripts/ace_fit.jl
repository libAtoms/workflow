#!/usr/bin/env julia

using ACE1pack, JSON, ArgParse
parser = ArgParseSettings(descirption="Fit an ACE potential from parameters file")
@add_arg_table parser begin
    "--fit-params", "-p"
        help = "A JSON filename with parameters for the fit"
    "--dry-run"
        help = "Only quickly compute various sizes, etc"
        action = :store_true
end

args = parse_args(parser)
fit_params = ACE1pack.json_to_params(args["fit-params"])

if args["dry-run"]
    # dataset size
    data = ACE1pack.read_data(fit_params["data"])    
    # basis size
    ACE_basis = ACE1pack.generate_rpi_basis(fit_params["rpi_basis"])
    pair_basis = ACE1pack.generate_pair_basis(fit_params["pair_basis"])
    dry_fit_filename = fit_params["ACE_fname_stem"] * ".size"
    open(dry_fit_filename, "w") do fit_info
        # TODO check that $(length(data)) is what we need
        write(fit_info, "LSQ matrix rows $(length(data)) basis $(length(ACE_basis) + length(pair_basis))\n") 
    end
    exit(0)
end

# TODO: option to read from a saved database
# TODO: ACE_FIT_BLAS_THREADS??
ACE1pack.fit_ace(fit_params)





