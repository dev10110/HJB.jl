
# return the index of the cell n steps away in direction dir
@inline function step(ind::CartesianIndex, dir::Integer, n::Integer)
    return Base.setindex(ind, ind[dir] + n, dir)
end

function gradient(
    data,
    grid::Grid{D,F},
    ind::CartesianIndex{D},
    dim::Integer;
    order = 1,
) where {D,F}

    @assert order == 1 # TODO: allow for higher order derivatives

    if ind[dim] == 1
        return gradient_left(data, grid, ind, dim; order = order)
    elseif ind[dim] == size(data, dim)
        return gradient_right(data, grid, ind, dim; order = order)
    else
        return gradient_interior(data, grid, ind, dim; order = order)
    end

end

function gradient_left(
    data,
    grid::Grid{D,F},
    ind::CartesianIndex{D},
    dim::Integer;
    order = 1,
) where {D,F}

    p = step(ind, dim, 1)
    n = ind
    return (data[p] - data[n]) / grid.dx[dim]

end

function gradient_right(
    data,
    grid::Grid{D,F},
    ind::CartesianIndex{D},
    dim::Integer;
    order = 1,
) where {D,F}

    p = ind
    n = step(ind, dim, -1)
    return (data[p] - data[n]) / grid.dx[dim]

end

function gradient_interior(
    data,
    grid::Grid{D,F},
    ind::CartesianIndex{D},
    dim::Integer;
    order = 1,
) where {D,F}

    p = step(ind, dim, 1)
    n = step(ind, dim, -1)
    return (data[p] - data[n]) / (2 * grid.dx[dim])

end



