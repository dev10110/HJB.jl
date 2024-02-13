
"""
    Grid(x0, dx, Ns)

parameterize a grid, such that `Array{T, D}` can be mapped to a physical domain of `D` dimensions. 

index `(0, 0, ..., 0)` will correspond to the physical point `grid.x0`

Note, this package makes heavy use of `OffsetArrays` to do its array indexing. Use `ind2state` and `state2ind` to correctly map between indices and physical points. 

"""
struct Grid{D,F}
    x0::NTuple{D,F} # origin of the grid
    dx::NTuple{D,F} # spacing along each dimension
    Ns::NTuple{D,Int} # number of cells in each dimension
    padding::Int
end


"""
    Grid( (x1s, x2s, ...) )
a simpler constructor for a grid, where you pass in a tuple of `xis <: AbstractRange`. 
"""
function Grid(xs::T, padding = 3) where {D,R<:AbstractRange,T<:NTuple{D,R}}

    x0 = Tuple(ntuple(i -> xs[i][1], D) |> collect)
    dx = Tuple(ntuple(i -> step(xs[i]), D) |> collect)
    Ns = Tuple(ntuple(i -> length(xs[i]), D) |> collect)

    return Grid(x0, dx, Ns, padding)

end

#TODO: provide interpolation methods

# # method to get the value at x from data, using the grid
# function (grid::Grid)(data, state)
#   ind = state2ind(grid, state)
#   return data[ind]
# end

"""
    ind2state(grid::Grid, ind::CartesianIndex)
returns a `SVector` with the state at the `ind` in the `grid`
"""
@inline function ind2state(grid::Grid{D,F}, ind::CartesianIndex) where {D,F}
    return SVector{D,F}(ntuple(i -> grid.x0[i] + ind[i] * grid.dx[i], D))
end

"""
    state2ind(grid::Grid, state)
returns a `CartesianIndex` corresponding to the `state` in the `grid`
"""
@inline function state2ind(grid::Grid{D,F}, state) where {D,F}
    # ind = ntuple(i -> round(Int, (state[i] - grid.x0[i]) ÷ grid.dx[i]), D)
    ind = ntuple(i -> Int(cld((state[i] - grid.x0[i]), grid.dx[i])), D)
    return CartesianIndex(ind)
end



"""
    DomainIndices(grid)

returns a `CartesianIndices` of a grid that correspond to the main computation domain, i.e., excluding any padding cells.
"""
function DomainIndices(grid::Grid{D,F}) where {D,F}
    return CartesianIndices(ntuple(i -> (0:(grid.Ns[i]-1)), D))
end

"""
    LeftBoundaryIndices(grid, dim)
returns a `CartesianIndices` of a `grid` that correspond to the left boundary of the computation domain along the `dim` dimension
"""
function LeftBoundaryIndices(grid::Grid{D,F}, dim) where {D,F}
    return CartesianIndices(ntuple(i -> (i == dim) ? (0:0) : (0:(grid.Ns[i]-1)), D))
end

"""
    RightBoundaryIndices(grid, dim)
returns a `CartesianIndices` of a `grid` that correspond to the right boundary of the computation domain along the `dim` dimension
"""
function RightBoundaryIndices(grid::Grid{D,F}, dim) where {D,F}
    Nax = grid.Ns[dim] - 1
    return CartesianIndices(ntuple(i -> (i == dim) ? (Nax:Nax) : (0:(grid.Ns[i]-1)), D))
end

"""
    getDomainAxes(grid::Grid, dim)

returns a range that defines the `axis` of the grid along `dim`. Useful for plotting. 
"""
function getDomainAxes(grid::Grid{D,F}, dim::Integer) where {D,F}
    return grid.x0[dim] .+ (0:(grid.Ns[dim]-1)) * grid.dx[dim]
end

"""
    allocate_grid([T=Float64,] grid::Grid{D, F}) where {D, F}

returns an `OffsetArray` that will be used to store the value function. The OffsetArray will store values of type `T`, which is `Float64` by default. 
The offset array is of `D` dimensions, and has the correct number of entries along each dimension. 

Elements are uninitialized.
"""
function allocate_grid(T, grid::Grid{D,F}) where {D,F}

    # allocate the memory
    new_size = grid.Ns .+ (2 * grid.padding)
    data = Array{T}(undef, new_size)

    # convert to offset arrays
    new_axes = ntuple(i -> ((-grid.padding):(grid.Ns[i]+grid.padding-1)), D)
    odata = OffsetArray(data, new_axes...)

    return odata
end

"""
    allocate_grid([T=Float64,] grid::Grid{D, F}, f, v=0) where {D, F}

returns an `OffsetArray` that will be used to store the value function. The OffsetArray will store values of type `T`, which is `Float64` by default. 
The offset array is of `D` dimensions, and has the correct number of entries along each dimension. 
Each element of the array will be filled with `f(x)` where `x` is the physical location corresponding to the grid cell. All padding cells are filled with `v`. 
"""
function allocate_grid(T, grid::Grid{D,F}, f, v = 0) where {D,F}

    odata = allocate_grid(T, grid)

    # now fill with the function
    @threads for ind in CartesianIndices(odata)
        if ind in DomainIndices(grid)
            x = ind2state(grid, ind)
            odata[ind] = f(x)
        else
            odata[ind] = v
        end
    end

    return odata
end

function allocate_grid(grid::Grid{D,F}, f, v = 0) where {D,F}
    return allocate_grid(Float64, grid, f, v)
end
