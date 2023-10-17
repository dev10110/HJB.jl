
"""
encodes the problem

    dV/dt + H(t, x, V, dV/dx, p) = 0
    V(tf, x) = l(x)

over the interval tspan=(t0, tf)
"""
struct HJBProblem{F,TH,TL,TP}
    H::TH
    l::TL
    tspan::Tuple{F,F}
    params::TP
end

struct HJBSolution_MOL{P,G,S}
    prob::P
    grid::G
    sol::S
end

# allow the problem to be constructed without params
HJBProblem(H, l, tspan) = HJBProblem(H, l, tspan, nothing)

# allow the HJBSolution to be accessed like a function
# TODO: make this much more efficient
function (sol::HJBSolution_MOL)(t, x)
    return sol.grid(sol.sol(t), x)
end

function (sol::HJBSolution_MOL)(t)
    return sol.sol(t)
end

# # not sure if this works
# function (sol::HJBSolution_MOL)(t, deriv::Val{M}) where M
#   return sol.sol(t, deriv)
# end

# convert the HJBProblem into an ODEProblem, given a grid
function get_ODEProblem(prob::HJBProblem, grid::Grid{D,F}) where {D,F}

    # define the RHS of the ODE Problem
    function RHS!(dV, V, p, t)

        @inbounds @threads for ind in CartesianIndices(V)
            x = ind2state(grid, ind)
            DxV = SVector{D,F}(gradient(V, grid, ind, dim) for dim = 1:D)
            dV[ind] = -prob.H(t, x, V[ind], DxV, p)
        end
        return
    end

    # construct the initial condition
    V0 = allocate_grid(grid)
    fill_grid!(V0, grid, prob.l)

    # construct the ODE Problem
    ode_func = DiffEq.ODEFunction(RHS!)
    ode_prob = DiffEq.ODEProblem(ode_func, V0, prob.tspan, prob.params)

    return ode_prob
end


function solve_MOL(prob::HJBProblem, grid::Grid{D,F}; kwargs...) where {D,F}

    ode_prob = get_ODEProblem(prob, grid)

    sol = DiffEq.solve(ode_prob; kwargs...)

    return HJBSolution_MOL(prob, grid, sol)

end
