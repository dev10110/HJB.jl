
"""
    step(ind, dim, n)

return the index of the cell `n` steps away from `ind` in the `dim` dimension.
"""
@inline function step(ind::CartesianIndex, dim::Integer, n::Integer)
    return Base.setindex(ind, ind[dim] + n, dim)
end


"""
    GradientMethod

abstract type defining the method used to determine the spatial gradient of a grid. Given `G <: GradientMethod`, the general form is `gradient(G, data, grid, ind, dim)` to get the gradient of the array `data` at `ind` along `dim` dimension in a `grid`.  
"""
abstract type GradientMethod end

"""
    LeftGradient()

Returns the first order left gradient, (u[i] - u[i-1])/dx
"""
struct LeftGradient <: GradientMethod end

"""
    RightGradient()

Returns first order right gradient (u[i+1] - u[i])/dx 
"""
struct RightGradient <: GradientMethod end

"""
    CentralGradient()

Returns second order central gradient  (u[i+1] - u[i-1])/(2 dx)
"""
struct CentralGradient <: GradientMethod end

"""
    LeftWenoGradient(ϵ=1e-6)

Returns the fifth order left WENO gradient.
See Sec 3.4 of Osher, Fedkiw Level Set Methods and Dynamic Implicit Surfaces
"""
struct LeftWenoGradient{T} <: GradientMethod
    ϵ::T
end
LeftWenoGradient() = LeftWenoGradient(1e-6)


"""
    RightWenoGradient(ϵ=1e-6)

Returns the fifth order right WENO gradient. 
See Sec 3.4 of Osher, Fedkiw Level Set Methods and Dynamic Implicit Surfaces
"""
struct RightWenoGradient{T} <: GradientMethod
    ϵ::T
end
RightWenoGradient() = RightWenoGradient(1e-6)

"""
    CentralWenoGradient(ϵ=1e-6)

Returns the fifth order central WENO gradient, i.e., the average of the left and right Weno gradients.
"""
struct CentralWenoGradient{T} <: GradientMethod
    ϵ::T
end
CentralWenoGradient() = CentralWenoGradient(1e-6)

"""
    StencilGradient(inds)
Returns a stencil based gradient operator, using Fornberg's method. Actually implemented using https://discourse.julialang.org/t/generating-finite-difference-stencils/85876/5
"""
struct StencilGradient{T,W}
    inds::UnitRange{T}
    weights::W
end

# returns a vector with weights corresponding to using stencil at relative inds assuming a spacing of 1.
function StencilGradient(inds::UnitRange{T}) where {T}

    x = inds
    x0 = 0 // 1
    m = 1

    l = 0:length(x)-1
    m in l || throw(ArgumentError("order $m ∉ $l"))
    A = @. (x' - x0)^l / factorial(l)
    w = A \ (l .== m) # vector of weights w

    return StencilGradient(inds, w)
end

"""
    gradient(data, grid, ind, dim)
    gradient(method::GradientMethod, data, grid, ind, dim)

returns the gradient of `data` at `ind` along dimension `dim` using the `grid` and the `method` gradient method. Default method is `CentralWenoGradient` 
"""
function gradient(data, grid, ind, dim)
    return gradient(CentralWenoGradient(), data, grid, ind, dim)
end

function gradient(method::LeftGradient, data, grid, ind, dim)
    indH = ind
    indL = step(ind, dim, -1)
    return (data[indH] - data[indL]) / grid.dx[dim]
end

function gradient(method::RightGradient, data, grid, ind, dim)
    indH = step(ind, dim, 1)
    indL = ind
    return (data[indH] - data[indL]) / grid.dx[dim]
end

function gradient(method::CentralGradient, data, grid, ind, dim)
    indH = step(ind, dim, 1)
    indL = step(ind, dim, -1)
    return (data[indH] - data[indL]) / (2 * grid.dx[dim])
end

function gradient(method::StencilGradient, data, grid, ind, dim)

    # get the coordinates of the data we want to check
    stencil_inds = step(ind, dim, method.inds[1]):step(ind, dim, method.inds[end])

    # dot product of the stencil weights with the data, and divide by the spacing
    g = dot(data[stencil_inds], method.weights) / grid.dx[dim]

    return g

end

function gradient(method::LeftWenoGradient, data, grid, ind, dim)
    # refer to Section 3.4 of Osher Fedkiw Level Set Methods
    v = ntuple(i -> gradient(LeftGradient(), data, grid, step(ind, dim, i - 3), dim), 5)
    return weno(v, method.ϵ)
end

function gradient(method::RightWenoGradient, data, grid, ind, dim)
    # refer to Section 3.4 of Osher Fedkiw Level Set Methods
    v = ntuple(i -> gradient(RightGradient(), data, grid, step(ind, dim, 3 - i), dim), 5)
    return weno(v, method.ϵ)
end

function gradient(method::CentralWenoGradient, data, grid, ind, dim)
    gm = gradient(LeftWenoGradient(method.ϵ), data, grid, ind, dim)
    gp = gradient(RightWenoGradient(method.ϵ), data, grid, ind, dim)
    return (gm + gp) / 2
end

"""
    weno(v, ϵ=1e-6)
helper function to compute the WENO gradient given `v=(v1,v2,v3,v4,v5)`
See Sec 3.5 of Osher, Fedkiw, Level Set Methods and Dynamic Implicit Surfaces.
"""
function weno(v, ϵ = 1e-6)
    # see chapter 3.5 of Osher, Fedkiw Level Set Methods and Dynamic Implicit Surfaces

    # expand out the tuple of values 
    v1, v2, v3, v4, v5 = v

    # compute the eno derivatives
    ϕx_1 = (1 // 3) * v1 + (-7 // 6) * v2 + (11 // 6) * v3
    ϕx_2 = (-1 // 6) * v2 + (5 // 6) * v3 + (1 // 3) * v4
    ϕx_3 = (1 // 3) * v3 + (5 // 6) * v4 + (-1 // 6) * v5

    # compute smoothness of the stencils
    S1 = (13 // 12) * (v1 - 2 * v2 + v3)^2 + (1 // 4) * (v1 - 4 * v2 + 3 * v3)^2
    S2 = (13 // 12) * (v2 - 2 * v3 + v4)^2 + (1 // 4) * (v2 - v4)^2
    S3 = (13 // 12) * (v3 - 2 * v4 + v5)^2 + (1 // 4) * (3 * v3 - 4 * v4 + v5)^2

    # determine the weights
    # ϵ0=1e-6
    # ϵ1=1e-12
    # ϵ = ϵ0 * max(v1^2, v2^2, v3^2, v4^4, v5^2) + ϵ1
    # note, in Sec 3.5, it suggests using the above formula, but a fixed ϵ is suggested in other implementations. 
    α1 = (1 // 10) / (S1 + ϵ)^2
    α2 = (3 // 5) / (S2 + ϵ)^2
    α3 = (3 // 10) / (S3 + ϵ)^2

    α = α1 + α2 + α3

    w1 = α1 / α
    w2 = α2 / α
    w3 = α3 / α

    return w1 * ϕx_1 + w2 * ϕx_2 + w3 * ϕx_3
end
