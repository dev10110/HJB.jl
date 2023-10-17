
struct Grid{D,F}
    x0::NTuple{D,F} # origin of the grid
    dx::NTuple{D,F} # spacing along each dimension
    Ns::NTuple{D,Integer} # number of cells in each dimension
end

function Grid(xs::T) where {D,R<:AbstractRange,T<:NTuple{D,R}}

    x0 = Tuple(ntuple(i -> xs[i][1], D) |> collect)
    dx = Tuple(ntuple(i -> step(xs[i]), D) |> collect)
    Ns = Tuple(ntuple(i -> length(xs[i]), D) |> collect)

    return Grid(x0, dx, Ns)

end

#TODO: provide interpolation methods

# # method to get the value at x from data, using the grid
# function (grid::Grid)(data, state)
#   ind = state2ind(grid, state)
#   return data[ind]
# end


@inline function ind2state(grid::Grid{D,F}, ind::CartesianIndex) where {D,F}
    return ntuple(i -> grid.x0[i] * ind[i] * grid.dx[i], D)
end

@inline function state2ind(grid::Grid{D,F}, state) where {D,F}
    ind = ntuple(i -> state[i] ÷ grid.dx[i], D)
    return CartesianIndex(ind)
end

function allocate_grid(grid::Grid)
    data = zeros(grid.Ns)
    return data
end

# optional type parameter T
function allocate_grid(T, grid::Grid)
    data = zeros(T, grid.Ns)
    return data
end

function fill_grid!(data, grid::Grid, f)
    @assert size(data) == grid.Ns
    @inbounds @threads for ind in CartesianIndices(data)
        data[ind] = f(ind2state(grid, ind))
    end
end
