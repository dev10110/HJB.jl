module HJB

# import DifferentialEquations as DiffEq
import SciMLBase
using StaticArrays # Check if I really need this
using OffsetArrays
import Base.Threads.@threads
import Base.step
using LinearAlgebra
using RecipesBase
using LazySets


include("grids.jl")
include("spatial_derivatives.jl")
include("hjb_problem.jl")
include("hamiltonians.jl")
include("plotting.jl")

end
