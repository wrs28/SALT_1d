module SALT_2d

( export InputStruct, ChannelStruct, processInputs, updateInputs!, computeCF,
computePole_L, computeZero_L, computeUZR_L, computeK_NL1, computePole_NL1,
computeZero_NL1, computeUZR_NL1, computeK_NL2, computePole_NL2, computeZero_NL2,
computeUZR_NL2, computeK_NL2_parallel, computePole_NL2_parallel, computeZero_NL2_parallel,
computeUZR_NL2_parallel, compute_scatter, computeS, analyze_output, analyze_input,
P_wait)
# solve_SPA, solve_scattered, solve_single_mode_lasing, solve_CPA, bootstrap

using NLsolve
using Formatting
using Interpolations

################################################################################
include("SALT_2d_core.jl")
include("SALT_2d_eigensolvers.jl")
# include("SALT_2d_analysis.jl")
include("SALT_2d_scattering.jl")
include("SALT_2d_scattering_analysis.jl")
include("SALT_2d_parallel.jl")
# include("SALT_2d_lasing.jl")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~
end #end of module SALT_2d
