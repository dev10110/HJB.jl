
# return the index of the cell n steps away in direction dir
@inline function step(ind::CartesianIndex, dir::Integer, n::Integer)
    return Base.setindex(ind, ind[dir] + n, dir)
end

@inline function step(ind::CartesianIndex, dir::Integer, n::Integer, s::Integer) # where s is the size
    new_i = clamp(ind[dir] + n, 1, s) # will automatically implement the 0-Neumann boundary condition
    return Base.setindex(ind, new_i, dir)
end

abstract type GradientMethod end

# simple first or second order differences
struct Simple <: GradientMethod end

# Weighted Essentially Non-Oscillatory Method
struct WENO <: GradientMethod end

# provide defaults
function gradient(data, grid, ind, dim)
    return gradient(Simple(), data, grid, ind, dim)
end

function gradient(data, grid, ind)
    return gradient(Simple(), data, grid, ind)
end

function gradient(
    method::Simple,
    data::AT,
    grid::Grid{D,F},
    ind::CartesianIndex{D},
    dim::Integer,
) where {D,F,T,AT<:AbstractArray{T}}

    # automatically imposes the Neuman BC
    @inbounds begin
        ND = grid.Ns[dim]
        indH = step(ind, dim, 1, ND)
        indL = step(ind, dim, -1, ND)

        gp = (data[indH] - data[ind]) / (grid.dx[dim])
        gm = (data[ind] - data[indL]) / (grid.dx[dim])
    end
    return (gm, gp)
end

function gradient(
    method::WENO,
    data::AT,
    grid::Grid{D,F},
    ind::CartesianIndex{D},
    dim::Integer,
) where {D,F,T,AT<:AbstractArray{T}}

    @inbounds begin

        ND = grid.Ns[dim]

        sub_inds = @SVector [step(ind, dim, n, ND) for n = -3:3]

        sub_data = ntuple(i -> data[sub_inds[i]], 7)

        res = weno_derivatives(sub_data, grid.dx[dim])

    end
    return res

    # if ind[dim] < 4 || ind[dim] > (ND - 4)
    #     return gradient_boundary(data, grid, ind, dim)
    # else
    #     return gradient_interior(data, grid, ind, dim)
    # end



end



# function gradient_interior(data::AT, grid::Grid{D,F}, ind::CartesianIndex{D}, dim::Integer) where {D,F, T, AT<:AbstractArray{T}}
#
#     # # assume it is an interior point
#     # @inbounds begin
#     #     sub_inds = @SVector [step(ind, dim, n) for n = -3:3]
#
#     #     sub_data = ntuple(i ->  data[sub_inds[i]], 7)
#
#     #     res = weno_derivatives(sub_data, grid.dx[dim])
#     # end
#     # return res
#
# end
#
# function gradient_boundary(data::AT, grid::Grid{D,F}, ind::CartesianIndex{D}, dim::Integer) where {D,F, T, AT<:AbstractArray{T}}
#     # assume it is a boundary point
#
#
#
#     # #### WENO
#     # @inbounds begin
#     #     ND = grid.Ns[dim]
#
#     #     sub_inds = @SVector [step(ind, dim, n, ND) for n = -3:3]
#
#     #     sub_data = ntuple(i ->  data[sub_inds[i]], 7)
#
#     #     res = weno_derivatives(sub_data, grid.dx[dim])
#     # end
#     # return res
#
# end


# now get the gradient vector
function gradient(
    method::M,
    data::AT,
    grid::Grid{D,F},
    ind::CartesianIndex{D},
) where {D,F,T,AT<:AbstractArray{T},M<:GradientMethod}

    gs = SVector{D,Tuple{T,T}}(gradient(method, data, grid, ind, dim) for dim = 1:D)

    return first.(gs), last.(gs)
end

# function gradient(
#     data,
#     grid::Grid{D,F},
#     ind::CartesianIndex{D},
#     dim::Integer;
#     order = 1,
# ) where {D,F}
#
#     @assert order == 1 # TODO: allow for higher order derivatives
#
#     if ind[dim] == 1
#         return gradient_left(data, grid, ind, dim; order = order)
#     elseif ind[dim] == size(data, dim)
#         return gradient_right(data, grid, ind, dim; order = order)
#     else
#         return gradient_interior(data, grid, ind, dim; order = order)
#     end
#
# end

## Stencils
#
# using Memoize
# using LRUCache

# functions to determine the stencils
function stencil(x::VF, x₀::Real, m::Integer) where {F,VF<:AbstractVector{F}}
    # credit: https://discourse.julialang.org/t/generating-finite-difference-stencils/85876/5
    l = 0:(length(x) - 1)
    @assert m in l # throw(ArgumentError("order $m ∉ $l"))
    A = @. (x' - x₀)^l / factorial(l)
    return A \ (l .== m) # vector of weights w
end


# @memoize LRU{Tuple{Integer, Integer}, Vector{Float64}}(maxsize=64)
function uniform_stencil(l::Integer, r::Integer)
    return stencil(l:r, 0, 1)::Vector{Float64}
end

# provide some convenient specializations, so that the cache isnt used in these cases
function derivative(ϕ, dx, i, left::Val{-1}, right::Val{1})
    # central differences, 2nd order
    return (ϕ[i + 1] - ϕ[i - 1]) / (2 * dx)
end


function derivative(ϕ, dx, i, left::Val{-1}, right::Val{0})
    # backward derivative
    return (ϕ[i] - ϕ[i - 1]) / (dx)
end


function derivative(ϕ, dx, i, left::Val{0}, right::Val{1})
    # forward derivative
    return (ϕ[i + 1] - ϕ[i]) / (dx)
end

# define the general case
function derivative(ϕ, dx, i, left::Val{L}, right::Val{R}) where {L,R} # the general form

    # using a stencil (stencil_l:stencil_r) .+ ind
    # to compute the derivative at index ind

    # get the stencil (potentially using the cache)
    s = uniform_stencil(L, R)

    # get the actual gradient
    return @views @inbounds (s' * ϕ[(ind + L):(ind + R)]) / dx
end

function left_weno(ϕ, dx)
    vm = @SVector [(ϕ[i + 1] - ϕ[i]) / dx for i = 1:5]
    return weno(vm)
end

function right_weno(ϕ, dx)
    vp = @SVector [(ϕ[8 - i] - ϕ[7 - i]) / dx for i = 1:5]
    return weno(vp)
end



function weno_derivatives(ϕ, dx)
    # given a list of ϕs = [ ϕ_{i-3}:ϕ_{i+3} ] # i.e, 7 numbers
    # returns the weno left and right derivatives at the middle node

    # see chapter 3.5 of Osher, Fedkiw Level Set Methods and Dynamic Implicit Surfaces

    # this is the logic, but its been simplified below for performance
    # ind = 4
    # ind_m = (ind-2):+1:(ind+2)
    # ind_p = (ind+2):-1:(ind-2)
    # vm = Tuple( (ϕ[i] - ϕ[i-1]) / dx for i=ind_m )
    # vp = Tuple( (ϕ[i+1] - ϕ[i]) / dx for i=ind_p )

    # vm = ntuple(i -> (ϕ[i + 1] - ϕ[i]) / dx, 5)
    # vp = ntuple(i -> (ϕ[8 - i] - ϕ[7 - i]) / dx, 5)

    @inbounds begin
        vm = @SVector [(ϕ[i + 1] - ϕ[i]) / dx for i = 1:5]
        vp = @SVector [(ϕ[8 - i] - ϕ[7 - i]) / dx for i = 1:5]

        return weno(vm), weno(vp)
    end
end

function weno(v)
    v1, v2, v3, v4, v5 = v

    # see chapter 3.5 of Osher, Fedkiw Level Set Methods and Dynamic Implicit Surfaces

    # compute the eno derivatives
    ϕ_x_1 = (1/3) * v1 + (-7/6) * v2 + (11/6) * v3 
    ϕ_x_2 = (-1/6) * v2 + (5/6) * v3 + (1/3) * v4 
    ϕ_x_3 = (1/3) * v3 + (5/6) * v4 + (-1/6) * v5

    # compute smoothness of the stencils
    S1 = (13 / 12) * (v1 - 2 * v2 + v3)^2 + (1 / 4) * (v1 - 4 * v2 + 3 * v3)^2
    S2 = (13 / 12) * (v2 - 2 * v3 + v4)^2 + (1 / 4) * (v2 - v4)^2
    S3 = (13 / 12) * (v3 - 2 * v4 + v5)^2 + (1 / 4) * (3 * v3 - 4 * v4 + v5)^2

    # get the weights
    ϵ0=1e-6
    ϵ1=1e-12
    ϵ = ϵ0 * max(v1^2, v2^2, v3^2, v4^4, v5^2) + ϵ1
    α1 = (1 / 10) / (S1 + ϵ)^2
    α2 = (6 / 10) / (S2 + ϵ)^2
    α3 = (3 / 10) / (S3 + ϵ)^2
    α = α1 + α2 + α3

    w1 = α1 / α
    w2 = α2 / α
    w3 = α3 / α

    return w1 * ϕ_x_1 + w2 * ϕ_x_2 + w3 * ϕ_x_3
end
