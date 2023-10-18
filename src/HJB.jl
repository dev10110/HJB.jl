module HJB

import DifferentialEquations as DiffEq
using StaticArrays
import Base.Threads.@threads
import Base.step
import ForwardDiff


include("types.jl")
include("grids.jl")
include("spatial_derivatives.jl")
include("hjb_problem.jl")
include("numerical_hamiltonian.jl")

end
