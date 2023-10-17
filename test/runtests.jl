using HJB
using Test
using StaticArrays
using LinearAlgebra

@testset "HJB.jl Double Integrator" begin


    # construct the grid
    x1s = range(0.0, 2.0, length = 128)
    x2s = range(-1.0, 1.5, length = 128)
    grid = HJB.Grid((x1s, x2s))

    @test grid isa HJB.Grid{2,Float64}
    println("grid created")

    # define the dynamics
    function f(t, x, u)
        return @SVector [x[2], u[1]]
    end

    # define the Hamiltonian
    function H(t, x, V, DxV, params)

        u1 = @SVector [1.0]
        u2 = @SVector [-1.0]

        H1 = dot(DxV, f(t, x, u1))
        H2 = dot(DxV, f(t, x, u2))

        return min(0, max(H1, H2))

    end

    # construct the terminal function
    function l(x)
        return 1.5 - x[1]
    end


    # construct the HJB Problem
    tspan = (0.0, 2.0)
    prob = HJB.HJBProblem(H, l, tspan)
    println("problem created")

    # solve the problem
    sol = HJB.solve_MOL(prob, grid; reltol = 1e-1, abstol = 1e-1)
    println("problem solved")

    @time sol = HJB.solve_MOL(prob, grid; reltol = 1e-1, abstol = 1e-1)

    @show sol.sol.retcode
    @test sol.sol.retcode == HJB.DiffEq.ReturnCode.Success


end
