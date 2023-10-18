
struct Grid{D,F}
    x0::NTuple{D,F} # origin of the grid
    dx::NTuple{D,F} # spacing along each dimension
    Ns::NTuple{D,Int64} # number of cells in each dimension
end

function Grid(xs::T) where {D,R<:AbstractRange,T<:NTuple{D,R}}

    x0 = Tuple(ntuple(i -> xs[i][1], D) |> collect)
    dx = Tuple(ntuple(i -> step(xs[i]), D) |> collect)
    Ns = Tuple(ntuple(i -> length(xs[i]), D) |> collect)

    return Grid(x0, dx, Ns)

end


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
