#########################
# DIRECT METHOD
#########################

function get_numH_direct(prob::HJBProblem, grid::Grid)

    function numH(t, x, V, DxVm, DxVp, params)

        DxV = (DxVm + DxVp) / 2
        Δ = (DxVp - DxVm) / 2

        return prob.H(t, x, V, (DxVm + DxVp) / 2, params) - α  * Δ

        # return H_avg
    end

    return numH
end

#########################
# LAX_FRIEDRICHS SCHEME
#########################


# # use forward diff to get the gradient of the Hamiltonian wrt to the co-states
# function get_DpH(prob)
# 
#     function DpH(t, x, V, DxV)
#         return ForwardDiff.gradient(p -> prob.H(t, x, V, p, prob.params), DxV)
#     end
# 
#     return DpH
# end
# 


# function dissipation_coefficients(t, x, V, DxVm, DxVp, DpH)
# 
#     # should implement
#     #   α = max_{p ∈ [DxVm, DxVp]} ( abs(∂H(t, x, V, p) / ∂p ) ) where the abs and max are element-wise
#     # here we provide a hacky way to implement this
# 
#     # the dot brodcasts over state-space dimension
#     αm = abs.(DpH(t, x, V, DxVm))
#     α0 = abs.(DpH(t, x, V, (DxVm + DxVp) / 2))
#     αp = abs.(DpH(t, x, V, DxVp))
# 
#     # find the max
#     α = max.(αm, α0, αp)
# 
#     return α
# 
# end
# 


function get_numH_lax_friedrichs(prob::HJBProblem, grid::Grid, dissipation_coefficients)
    # loosely based on https://www.cs.ubc.ca/~mitchell/ToolboxLS/toolboxLS-1.1.pdf Sec 3.6.2

    # DpH = get_DpH(prob)

    function numH(t, x, V, DxVm, DxVp, params)

        H_avg = prob.H(t, x, V, (DxVm + DxVp) / 2, params)

        # TODO: allow the dissipation coefficients evaluation method to be specified
        α = dissipation_coefficients(t, x)
        # α = dissipation_coefficients(t, x, V, DxVm, DxVp, DpH)

        H_dis = α' * ((DxVp - DxVm) / 2 )

        return H_avg - H_dis

    end


end

# function get_ODE_RHS_lax_friedrichs(prob::HJBProblem, grid::Grid, dissipation_coefficients)
# 
#     numH = get_numH_lax_friedrichs(prob, grid, dissipation_coefficients)
# 
#     # define the RHS of the ODE Problem
#     function RHS!(dV, V, p, t)
# 
#         @inbounds @threads for ind in CartesianIndices(V)
#             x = ind2state(grid, ind)
# 
#             DxVm, DxVp = gradient(V, grid, ind) # get the left and right gradients
# 
#             # evaluate the lax-friedrichs numerical hamiltonian
#             dV[ind] = -numH(t, x, V[ind], DxVm, DxVp)
# 
#         end
#         return
#     end
# 
#     return RHS!
# end
# 
# 
