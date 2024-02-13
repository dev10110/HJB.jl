

"""
    update_ghost_nodes!(V, grid)

`V` is the array that contains the value function, specified using the `grid`. 

updates (in-place) the values of the ghost nodes, using the boundary gradients and extrapolating. 
TODO(dev): support periodic boundaries
"""
function update_ghost_nodes!(data, grid::Grid{D,F}) where {D,F}

    for dim = 1:D
        update_ghost_nodes_left!(data, grid, dim)
        update_ghost_nodes_right!(data, grid, dim)
    end

end

function update_ghost_nodes_left!(data, grid::Grid{D,F}, dim) where {D,F}

    @inbounds @threads for ind in LeftBoundaryIndices(grid, dim)

        # estimate the gradient
        grad = gradient(RightGradient(), data, grid, ind, dim)

        # fill values
        for i = 1:grid.padding
            ghost_ind = step(ind, dim, -i)
            data[ghost_ind] = data[ind] + grad * (-i * grid.dx[dim])
        end

    end
end

function update_ghost_nodes_right!(data, grid::Grid{D,F}, dim) where {D,F}

    @inbounds @threads for ind in RightBoundaryIndices(grid, dim)

        # estimate the gradient
        grad = gradient(LeftGradient(), data, grid, ind, dim)

        # fill values in the ghost cells
        for i = 1:grid.padding
            ghost_ind = step(ind, dim, i)
            data[ghost_ind] = data[ind] + grad * (i * grid.dx[dim])
        end
    end
end



abstract type NumericalHamiltonianMethod end
const NHM = NumericalHamiltonianMethod # type alias to save typing

"""
    SimpleNHM(left_grad, right_grad)
is a numerical hamiltonian method that implements
```
hamil(t, x, V, ∇V) = hamil(t, x, V, (gp + gm) / 2)
```
where `gm, gp` are the left and right gradients `V`.
"""
struct SimpleNHM{LG,RG} <: NHM
    left_grad::LG
    right_grad::RG
end

"""
    LocalLaxFriedrichsNHM(left_grad, right_grad, dissipation_func)
is a numerical hamiltonian method that implements

```
hamil(t, x, V, ∇V) = hamil(t, x, V, (gp + gm) / 2) - α' * (gp - gm) / 2
```
where `gm, gp` are the left and right gradients of `V`, and `α = dissipation_func(t, x)`
"""
struct LocalLaxFriedrichsNHM{LG,RG,F} <: NHM
    left_grad::LG
    right_grad::RG
    dissipation_func::F
end


"""
    propagate_hamiltonian!(hamil, time, grid, V, dVdt, method::NumericalHamiltonianMethod)

`V` is the array that contains the value function, specified using the `grid`. 
`dvdt` is updated (in-place) to contain the `dV/dt` term. 

Ideally, we want 
```
dV/dt = -hamil(t, x, V, ∇V)
```
but since `V` may not be continuously differentiable, we must use the left and right gradients carefully. These are specified in the `method.left_grad` and `method.right_grad` functions. 

The specific mathematical form of the numerical hamiltonian is described in the docs for each method. 
"""

function propagate_hamiltonian!(
    hamil,
    time,
    grid::Grid{D,F},
    data,
    ddata_dt,
    method::M,
) where {D,F,M<:SimpleNHM}

    @inbounds @threads for ind in DomainIndices(grid)

        # get the x value
        x = ind2state(grid, ind)

        # get the V value
        V = data[ind]

        # get the left and right gradients
        gm = SVector{D}(ntuple(dim -> gradient(method.left_grad, data, grid, ind, dim), D))
        gp = SVector{D}(ntuple(dim -> gradient(method.right_grad, data, grid, ind, dim), D))

        # store the dV/dt
        ddata_dt[ind] = -hamil(time, x, V, (gm + gp) / 2)
    end

end

function propagate_hamiltonian!(
    hamil,
    time,
    grid::Grid{D,F},
    data,
    ddata_dt,
    method::M,
) where {D,F,M<:LocalLaxFriedrichsNHM}

    @inbounds @threads for ind in DomainIndices(grid)

        # get the x value
        x = ind2state(grid, ind)

        # get the V value
        V = data[ind]

        # get the left and right DxV
        gm = SVector{D}(ntuple(dim -> gradient(method.left_grad, data, grid, ind, dim), D))
        gp = SVector{D}(ntuple(dim -> gradient(method.right_grad, data, grid, ind, dim), D))

        # determine the numerical dissipation
        α = method.dissipation_func(time, x)

        # store the dV/dt for this index
        ddata_dt[ind] = -hamil(time, x, V, (gm + gp) / 2) + (1 / 2) * α' * (gm - gp)
    end

end

function get_ODE_RHS!(hamil, grid, method::M) where {M<:NHM}

    function ODE_RHS!(DV, V, params, time)

        # set all the DV to 0 first
        # DV .= 0 # not sure if i need to do this

        # first update the ghost nodes
        update_ghost_nodes!(V, grid)

        # next update all the interior cells
        propagate_hamiltonian!(hamil, time, grid, V, DV, method)

        return

    end

    return ODE_RHS!
end

default_nhm_method() = SimpleNHM(LeftGradient(), RightGradient())

function get_ODEProblem(hamiltonian, l, grid, tspan, method::NHM = default_nhm_method())

    # create the initial grid
    V0 = allocate_grid(grid, l)

    # get the RHS
    RHS! = get_ODE_RHS!(hamiltonian, grid, method)

    # construct the ODE Problem
    prob = SciMLBase.ODEProblem(RHS!, V0, tspan)

    return prob

end
