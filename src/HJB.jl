module HJB

import DifferentialEquations as DiffEq
using StaticArrays
import Base.Threads.@threads
import Base.step


include("grids.jl")
include("spatial_derivatives.jl")
include("hjb_problem.jl")

end
