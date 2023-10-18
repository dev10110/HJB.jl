

#TODO: provide interpolation methods

# # method to get the value at x from data, using the grid
# function (grid::Grid)(data, state)
#   ind = state2ind(grid, state)
#   return data[ind]
# end

"""
    ind2state(grid, ind)

returns the state (as a tuple) corresponding to the index  `ind` in the `grid`
"""
@inline function ind2state(grid::Grid{D,F}, ind::CartesianIndex) where {D,F}
    return ntuple(i -> grid.x0[i] + (ind[i] - 1) * grid.dx[i], D)
end

"""
    state2ind(grid, state)

returns the index in the `grid` (as a `CartesianIndex`) corresponding to the `state`.
"""
@inline function state2ind(grid::Grid{D,F}, state) where {D,F}
    ind = ntuple(i -> round(Int, (state[i] - grid.x0[i]) ÷ grid.dx[i]) + 1, D)
    return CartesianIndex(ind...)
end

"""
    get_axes(grid, d)
returns a `StepRangeLen` describing the `d` dimension of the grid
"""
function get_axes(grid::Grid{D,F}, d) where {D,F}
    @assert 1 <= d <= D
    return grid.x0[d] .+ grid.dx[d] * (0:(grid.Ns[d] - 1))
end

"""
    get_axes(grid)
returns a tupe of `get_axes(grid, d)` for each dimension `d`
"""
function get_axes(grid::Grid{D,F}) where {D,F}
    return ntuple(i -> get_axes(grid, i), D)
end

function allocate_grid(grid::Grid)
    data = zeros(grid.Ns)
    return data
end


# import Base.CartesianIndicies
# """
#     CartesianIndicies(grid)
# small helper function to iterate over a grid
# """
# function Base.CartesianIndicies(grid::Grid{D, F}) where {D, F}
#     return CartesianIndicies(grid.Ns)
# end



"""
    allocate_grid(T=Float64, grid)
returns a `zero` array of type T corresponding to the grid
[WARNING] allocates an array of size `grid.Ns`
"""
function allocate_grid(T, grid::Grid)
    data = zeros(T, grid.Ns)
    return data
end

"""
    allocate_grid(T=Float64, grid, f)
returns an array of type T corresponding to the grid, where each cell is filled with `x->f(x)`.
[WARNING] allocates an array of size `grid.Ns`
"""
function allocate_grid(T, grid::Grid, f)
    data = Array{T}(undef, grid.Ns)
    fill_grid!(data, grid, f)
    return data
end

"""
    fill_grid!(data, grid, f)
applies the function `x->f(x)` to each point in the grid, and stores the result into `data`. Does not allocate new memory.
"""
function fill_grid!(data, grid::Grid, f)
    @assert size(data) == grid.Ns
    @inbounds @threads for ind in CartesianIndices(data)
        data[ind] = f(ind2state(grid, ind))
    end
end
