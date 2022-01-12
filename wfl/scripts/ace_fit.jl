#!/usr/bin/env julia

pot_suffixes = [".json", ".yace"]

using ArgParse
parser = ArgParseSettings(description="Fit an ACE potential from data in a file")
@add_arg_table parser begin
    "--atoms_filename", "-a"
    help = "one or more xyz files with fitting configurations (required)"
        arg_type = String
        nargs = '+'
        required = true
    "--outfile_base", "-O"
        help = "output file for potential without suffix, also used as base for other output like lsq database (required)" 
        arg_type = String
        required = true
    "--outfile_format", "-o"
        help = "format(s) to write output, allowed values: " * join(pot_suffixes, " ")
        nargs = '+'
        default = [".yace"]
        arg_type = String
    "--dbfile"
        help = "file to save or read database, defaults to outfile_base * '.db' "
        arg_type = String
    "--load_dbfile"
        help = "load LSQ dbfile from previous run if available"
        action = :store_true
    "--save_dbfile"
        help = "save LSQ dbfile to use in subsequent runs"
        action = :store_true
    "--r0", "-r"
        help = "typical range, default to mean of rnn() of species.  Required if any rnn() values are not available."
        arg_type = Float64
    "--cutoff_mb_inner", "-I"
        help = "manybody inner cutoff, default to 0.8 * r0. Mutually exclusive with \"cutoffs_mb\""
        arg_type = Float64
    "--cutoff_mb_outer", "-C"
        help = "manybody cutoff, default to 2.0 * r0. Mutually exclusive with \"cutoffs_mb\""
        arg_type = Float64
    "--cutoffs_mb"
        help = "JSON sting specifying per-element-pair cutoff parsed into " *
               "Dict((AtomicNumber, AtomicNumber) => (Float, Float)). " *
               "AtomicNumber should be stored as str(element) (e.g. \"H\") " *
               "Mutually exclusive with (r_inner_mb and cutoff_mb)"
    "--cutoff_pair_outer", "-c"
        help = "pair potential cutoff, default to 3.0 * r0"
        arg_type = Float64
    "--body_order"
        help = "body order (correlation_order + 1)"
        arg_type = Int
    "--correlation_order", "-N"
        help = "correlation order"
        arg_type = Int
        default = 2 
    "--degree", "-D"
        help = "manybody potential polynomial degree." *
            "either (string parsed to) Int or a JSON string specifying polynomial degree " *
            "for each correlation order. Dictionary keys can also be a tuple of " *
            "(correlation_order, AtomicNumber)." 
        arg_type = String 
        default = "12"
    "--degree_pair", "-d"
        help = "pair potential polynomial degree"        
        arg_type = Int
        default = 3
    "--solver", "-s"
        help = "solver type and arguments.  First argument will be converted to solver symbol, " *
               "and second used as JSON string for its arguments.  An encoded scalar or array " *
               "will result in one or more tuple elements passed to lsqfit in the 'solver' argument. "*
               "To pass an argument which itself is an array, encode it as an array of arrays, e.g. "*
               "'-s lap \"[[1.0, 2.0]]\"' will pass `(:lap, [1.0, 2.0])`. (repeated to loop over different sets)"
        arg_type = String
        nargs = 2
        metavar = [ "solver_type", "solver_args_json_str" ]
        default = [[ "rrqr", "1.0e-7" ]]
        action = :append_arg
    "--key", "-k"
        help = "info/arrays keys for energy (if first arg is E), forces (first arg F), " *
               "virial property (first arg V).  If none are specified will default to "*
               "E -> energy, F -> forces, V -> virial, S -> stress (internally converted to virial, currently unsupported), "*
               "but if some are specified the remainder will not be used for fitting (repeated)."
        arg_type = String
        nargs = 2
        metavar = ["EFVS", "key_string"]
        action = :append_arg
    "--weights", "-w"
        help = "JSON encoded dict with dict of weights for each config type, with keys indicating " *
               "config type (including special value 'default'), and values of dicts with E/F/V as keys " *
               "for energy, force, and virial weights, respectively. Example {\"default\" : {\"E\" : 1.0, \"F\": 5.0, \"V\": 1.0}} " *
               "(repeated to loop over different sets)." 
        arg_type = String
        default = ["{}"]
        action = :append_arg
    "--E0"
        help = "WARNING: do not use unless you know what you are doing - will result in potentials that "*
               "do not necessarily predict isolated atom energies correctly. E0 value for each element symbol, "*
               "instead of using config_type=='isolated_atom'. Cannot be specified for elements that have "*
               "config_type=='isolated_atom'"
        arg_type = String
        metavar = ["chem_symbol", "E0"]
        nargs = 2
        action = :append_arg
    "--dry_run", "-n"
        help = "dry run - only compute various sizes, but no expensive things like assembling Lsq matrix or fitting"
        action = :store_true
end

args = parse_args(parser)
if length(args["solver"]) > 1
    # :append_arg adds to default, so remove it if --solver was specified
    popfirst!(args["solver"])
end
if length(args["weights"]) > 1
    # :append_arg adds to default, so remove it if --weights was specified
    popfirst!(args["weights"])
end

@show args


function parse_degree_site(input::AbstractString)

    try
        # parse single integer
        degree = parse(Int, input)
        return degree
    catch err
        # throw anything unexpected
        if ~isa(err, ArgumentError)
            throw(err)
        end
    end

    # parse JSON
    raw_dict = JSON.parse(input)
    out_dict = Dict() 
    for (key, val) in raw_dict
        if key == "default"
            out_dict[key] = val
        elseif occursin(",", key) 
            # parse key as a tuple
            new_key = parse_degree_string_into_tuple(key)
            out_dict[new_key] = val
        else
            # key is a correlation order (Int)
            new_key = parse(Int, key)
            out_dict[new_key] = val
        end
    end

    return out_dict
end

function parse_degree_string_into_tuple(input::AbstractString)
    input = strip(input, ['(', ')'])
    entries = [strip(s) for s in split(input, ",")]
    correlation_order = parse(Int, entries[1])
    atomic_number = AtomicNumber(Meta.parse(entries[2]))
    return (correlation_order, atomic_number)
end


using JSON
using IPFitting
using ACE
using JuLIP
using Statistics
using LinearAlgebra

if haskey(ENV, "ACE_FIT_BLAS_THREADS")
    nprocs = parse(Int, ENV["ACE_FIT_BLAS_THREADS"])
    @warn "Using $nprocs threads for BLAS"
    BLAS.set_num_threads(nprocs)
end

keys = Dict("E" => "_NONE_", "F" => "_NONE_", "V" => "_NONE_")
if length(args["key"]) == 0
    keys["E"] = "energy"
    keys["F"] = "forces"
    keys["V"] = "virial"
else
    for EFV_key in args["key"]
        if EFV_key[1] == "S"
            error("fitting key stress S not supported")
        end
        keys[EFV_key[1]] = EFV_key[2]
    end
end
@show keys

E0 = Dict{Symbol, Float64}()
if length(E0) > 0
    @warn "Using E0 values from command line: will not produce potential with correct single " *
          "atom energy unless corresponding configurations are in fitting database"
end
for elem_e0 in args["E0"]
    E0[Symbol(elem_e0[1])] = parse(Float64, elem_e0[2])
end

suffixes = [s[1] == '.' ? s : "." * s for s in args["outfile_format"]]
for suffix in suffixes
    if ! (suffix in pot_suffixes)
        error("Unknown file type suffix '" * suffix * "'")
    end
end
@show suffixes

if length(args["body_order"]) != 0
    if length(args["correlation_order"]) != 0
        Throw(ArgumentError("Both \"body_order\" and \"correlation_order\" are given."))
    end
    N = args["body_order"] - 1
else
    N = args["correlation_order"]
end

# Acutlaly, I and Noam use deg_site differently. 
# I: to construct ACE.RPI.SparsePSHDegreeM, 
# Noam: as maxdeg. 
deg_site = parse_degree_site(args["degree"])
deg_pair = args["degree_pair"]

####################################################################################################

cfgs = Vector{Dat}()
for atfile in args["atoms_filename"]
    append!(cfgs, IPFitting.Data.read_xyz(atfile, energy_key=keys["E"], force_key=keys["F"], virial_key=keys["V"]))
end
@show unique(configtype.(cfgs));

# calculate E0 for each chemical species
species = Set()
for cfg in cfgs
    global species = union!(species, JuLIP.chemical_symbol.(cfg.at.Z))
    if configtype(cfg) == "isolated_atom"
        symb = JuLIP.chemical_symbol(cfg.at.Z[1])
        if haskey(E0, symb)
            error("Got isolated_atom config and command line E0 value for species " * String(symb))
        else
            E0[symb] = cfg.D["E"][1]
        end
    end
end
@show species
@show E0

for sp in species
    if ! haskey(E0, sp)
        error("No E0 for species $sp")
    end
end

r0 = args["r0"]
if isnothing(args["r0"])
    @warn "No --r0, using mean of rnn of species present " * join([(sp, rnn(sp)) for sp in species]," ")
    if any(rnn.(species) .<= 0.0)
        error("Some species has rnn <= 0")
    end
    r0 = Statistics.mean(rnn.(species))
end


something = nothing
# inner and outer cutoffs
if ~isnothing(args["cutoffs_mb"])
    if ~isnothing(args["r_inner_mb"]) or ~isnothing(args["cutoff_mb"])
        throw(Argumenterror("\"cutoffs_mb\" and (\"r_inner_mb\" & \"cutoff_mb\") are mutually exclusive"))
    end
    # deal with multi-body cutoffs
    cuttoffs = something 
    rcut = nothing
    rin = nothing
elseif true
    # check for inner cutoff
    # check for outer cutoff
    # set defaults or given
    cutoffs = nothing
    rcut = something
    rin = something 
end
# rin_mb = isnothing(args["r_inner_mb"]) ? 0.8*r0 : args["r_inner_mb"]
# rcut_mb = isnothing(args["cutoff_mb"]) ? 2.0*r0 : args["cutoff_mb"]
# rcut_pair = isnothing(args["cutoff_pair"]) ? 3.0*r0 : args["cutoff_pair"]

# construction of a (basic) basis for site energies 
@show "rpi_basis", N, deg_site, r0, rin_mb, rcut_mb


# write out explicitly what before was in function rpi_basis()
pcut=2
pin=2
constants = false
# species=[:C, :H] # defined above, don't need anymore

# separate definitions depending on how cutoffs and transform are defined
# might work if combined together, but I'm not confident enough that'd 
# work as it's supposed to.
if ~isnothing(cutoffs)

    # eg - The first example I got had the following
    maxdeg = 1.0
    #n weights
    Dn = Dict( "default" => 1.0 )
    #l weights
    Dl = Dict( "default" => 1.5 ) 

    Deg = ACE.RPI.SparsePSHDegreeM(Dn, Dl, deg_site)


    # keep the transforms as, but need to specify it for each element pair
    trans_general = PolyTransform(2, r0)
    # Could put `(:X, :X)` => PolyTransform(2, r0) to cover all species?
    transforms = Dict(
            (:C, :C) => trans_general, 
            (:C, :H) => trans_general, 
            (:H, :H) => trans_general)

    trans = multitransform(transforms, cutoffs=cutoffs)
    rbasis = transformed_jacobi(get_maxn(Deg, maxdeg, species), trans; pcut=pcut, pin=pin)

else
    maxdeg = deg_site
    Deg = SparsePSHDegree()
    trans = PolyTransform(2, r0)
    rbasis = transformed_jacobi(get_maxn(Deg, maxdeg, species), trans, rcut_mb, rin_mb; pcut=pcut, pin=pin)
end

basis1p = BasicPSH1pBasis(rbasis; species=species, D=Deg)
Bsite=RPIBasis(basis1p, N, Deg, maxdeg, constants)


# pair potential basis 
@show "pair_basis", r0, deg_pair, rcut_pair

pair_rin = 0.0
pair_trans = PolyTransform(2, r0)
rbasis = transformed_jacobi(deg_pair, pair_trans, rcut_pair, pair_rin; pcut=2, pin=0)
Bpair = PolyPairBasis(rbasis, species)

B = JuLIP.MLIPs.IPSuperBasis([Bpair, Bsite]);
@show length(B)

if args["dry_run"]
   nrows = 0
   for (okey, d, _) in IPFitting.observations(cfgs)
      len = length(IPFitting.observation(d, okey))
      global nrows += len
   end
   open(args["outfile_base"] * ".size", "w") do finfo
       write(finfo, "LSQ matrix rows $(nrows) basis $(length(B))\n")
   end
   exit(0)
end

# reload or calculate db
if isnothing(args["dbfile"])
    args["dbfile"] = args["outfile_base"] * ".db"
end
dB = nothing
if args["load_dbfile"]
    if isfile(args["dbfile"] * "_info.json")
        dB = LsqDB(args["dbfile"])
        @warn "Reloaded database"
    else
        @warn "Asked to reload database but info.json file not found"
    end
end
if isnothing(dB)
    dB = LsqDB(args["save_dbfile"] ? args["dbfile"] : "", B, cfgs)
end

Vref = OneBody(E0)

# 1.05 - sacrifice 5% of min error for regularization
# IP, lsqinfo = lsqfit(dB, solver=(:rid, 1.05), weights = weights, Vref = Vref);

@warn "finally starting lsqfit"
@show args["weights"]

for (weights_i, weights_json) in enumerate(args["weights"])
    weights = Dict()
    for (config_type, weight_dict) in JSON.parse(weights_json)
        weights[config_type] = weight_dict
    end
    @show weights

    for (solver_i, solver) in enumerate(args["solver"])
        solver = Tuple(cat(Symbol(solver[1]), JSON.parse(solver[2]), dims=(1,1)))
        @show solver

        IP, lsqinfo = lsqfit(dB, solver=solver, weights = weights, Vref = Vref);

        add_fits_serial!(IP, cfgs)
        rmse_, rmserel_ = rmse(cfgs);
        rmse_table(rmse_, rmserel_)

        name_suffix = ""
        if length(args["weights"]) > 1
            name_suffix *= "_weights_i_$(weights_i)"
        end
        if length(args["solver"]) > 1
            name_suffix *= "_solver_i_$(solver_i)"
        end
        for suffix in suffixes
            outfile = args["outfile_base"] * name_suffix * suffix
            if suffix == ".json"
                save_dict(outfile, Dict("IP" => write_dict(IP), "info" => lsqinfo))
            elseif suffix == ".yace"
                ACE.ExportMulti.export_ACE(outfile, IP)
            else
                error("IMPOSSIBLE! Trying to write to unknown file type suffix '" * suffix * "'")
            end
        end
    end
end

## DISTRIBUTION A: Approved for public release, distribution is unlimited
