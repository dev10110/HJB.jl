
abstract type TimeMethod end

struct Direct <: TimeMethod end

struct LaxFriedrichs <: TimeMethod end



## allow the HJBSolution to be accessed like a function
## TODO: make this much more efficient
#function (sol::HJBSolution_MOL)(t, x)
#    return sol.grid(sol.sol(t), x)
#end
#
#function (sol::HJBSolution_MOL)(t)
#    return sol.sol(t)
#end

# # not sure if this works
# function (sol::HJBSolution_MOL)(t, deriv::Val{M}) where M
#   return sol.sol(t, deriv)
# end

# use forward diff to get the gradient of the Hamiltonian wrt to the co-states
function get_DpH(prob)

    function DpH(t, x, V, DxV)
        return ForwardDiff.gradient(p -> prob.H(t, x, V, p, prob.params), DxV)
    end

    return DpH
end

# provide the default
function get_ODE_RHS(prob::HJBProblem, grid::Grid{D,F}) where {D,F}
    return get_ODE_RHS(prob, grid, Direct(), Simple())
end

function get_ODE_RHS(
    prob::HJBProblem,
    grid::Grid{D,F},
    timeMethod::Direct,
    gradMethod::G,
) where {D,F,G<:GradientMethod}

    # construct the RHS function
    function RHS!(dV, V, p, t)

        @inbounds @threads for ind in CartesianIndices(V)

            # get the state
            x = ind2state(grid, ind)

            # get the left and right gradients
            DxVm, DxVp = gradient(gradMethod, V, grid, ind)

            # evaluate the numerical hamiltonian
            DxV = (DxVm + DxVp) / 2
            dV[ind] = -prob.H(t, x, V[ind], DxV, p)

        end
        return
    end

    return RHS!

end


function get_ODE_RHS(
    prob::HJBProblem,
    grid::Grid{D,F},
    timeMethod::LaxFriedrichs,
    gradMethod::G,
) where {D,F,G<:GradientMethod}
    # get the numerical hamiltonian
    numH = get_numH_lax_friedrichs(prob, grid)

    # construct the RHS function
    function RHS!(dV, V::AF, p, t) where {F,AF<:AbstractArray{F}}

        @inbounds @threads for ind in CartesianIndices(V)

            # get the state
            x = ind2state(grid, ind)

            # get the left and right gradients
            DxVm, DxVp = gradient(gradMethod, V, grid, ind)

            # evaluate the numerical hamiltonian
            nH::F = numH(t, x, V[ind], DxVm, DxVp, p)
            dV[ind] = -nH

        end
        return
    end

    return RHS!

end

# convert the HJBProblem into an ODEProblem, given a grid
function get_ODEProblem(
    prob::HJBProblem,
    grid::Grid{D,F};
    timeMethod::TM = Direct(),
    gradMethod::GM = Simple(),
) where {D,F,TM<:TimeMethod,GM<:GradientMethod}

    RHS! = get_ODE_RHS(prob, grid, timeMethod, gradMethod)

    # construct the initial condition
    V0 = allocate_grid(grid)
    fill_grid!(V0, grid, prob.l)

    # construct the ODE Problem
    ode_func = DiffEq.ODEFunction(RHS!)
    ode_prob = DiffEq.ODEProblem(ode_func, V0, reverse(prob.tspan), prob.params)

    return ode_prob
end


function solve_MOL(
    prob::HJBProblem,
    grid::Grid{D,F};
    timeMethod::TM = Direct(),
    gradMethod::GM = Simple(),
    kwargs...,
) where {D,F,TM<:TimeMethod,GM<:GradientMethod}

    ode_prob = get_ODEProblem(prob, grid; timeMethod = timeMethod, gradMethod = gradMethod)

    sol = DiffEq.solve(ode_prob; kwargs...)

    return HJBSolution_MOL(prob, grid, sol)

end
