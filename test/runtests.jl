using HJB
using Test
using StaticArrays
using LinearAlgebra
using Plots

@testset "HJB.jl: ind2dstate, state2ind" begin

    # create the axes
    x1s = range(-1.0, 1.0, length = 64)
    x2s = range(5.0, 100.0, length = 128)

    # create the grid
    grid = HJB.Grid((x1s, x2s))

    @test grid isa HJB.Grid{2,Float64}
    println("grid created")

    for i = 1:length(x1s), j = 1:length(x2s)
        @test collect(HJB.ind2state(grid, CartesianIndex(i, j))) ≈ [x1s[i], x2s[j]] atol =
            1e-6
        ind = HJB.state2ind(grid, [x1s[i], x2s[j]])
        @test i - 1 <= ind[1] <= i + 1 #TODO IMPROVE THIS!!
        @test j - 1 <= ind[2] <= j + 1
    end

end

@testset "HJB.jl: Gradients (1D)" begin

    for method in [HJB.Simple(), HJB.WENO()] # check both methods

        # create the axes
        xs = range(-1.0, 1.0, length = 101)
        dx = step(xs)

        # create the grid
        grid = HJB.Grid((xs,))

        @test grid isa HJB.Grid{1,Float64}
        println("grid created")

        # set some values
        data = abs.(xs)
        data[xs .> 0] = 2 * xs[xs .> 0]

        # get the gradients
        grads = [HJB.gradient(method, data, grid, ind, 1) for ind in CartesianIndices(data)]

        left_grads = first.(grads)
        right_grads = last.(grads)

        # test against expected grads
        for (i, x) in enumerate(xs)
            if i == 1
                @test left_grads[i] ≈ 0 atol = 1e-1
                @test right_grads[i] ≈ -1 atol = 1e-1
            elseif i == length(xs)
                @test left_grads[i] ≈ 2 atol = 1e-1
                @test right_grads[i] ≈ 0 atol = 1e-1
            elseif x <= -2 * dx
                @test left_grads[i] ≈ -1 atol = 1e-1
                @test right_grads[i] ≈ -1 atol = 1e-1
            elseif x >= 2 * dx
                @test left_grads[i] ≈ 2 atol = 1e-1
                @test right_grads[i] ≈ 2 atol = 1e-1
            elseif x == 0
                @test left_grads[i] ≈ -1 atol = 1e-1
                @test right_grads[i] ≈ 2 atol = 1e-1
            end
        end

    end

end


@testset "HJB.jl: Gradients (2D)" begin

    for method in [HJB.Simple(), HJB.WENO()]

        # create the axes
        xs = range(-1.0, 1.0, length = 100)

        # create the grid
        grid = HJB.Grid((xs, xs))

        @test grid isa HJB.Grid{2,Float64}
        println("grid created")

        # set some values
        data = HJB.allocate_grid(grid)
        @test size(data) == (length(xs), length(xs))

        HJB.fill_grid!(data, grid, x -> sin(x[1]) + cos(x[2]))

        # check each interior point
        for ind in CartesianIndices(data)
            if ind[1] <= 3 ||
               ind[2] <= 3 ||
               ind[1] >= size(data, 1) - 3 ||
               ind[2] >= size(data, 2) - 3
                continue
            end

            # get state
            x = HJB.ind2state(grid, ind)

            # now check the functions
            exp_grad_1 = cos(x[1])
            exp_grad_2 = -sin(x[2])

            g1 = HJB.gradient(data, grid, ind, 1)
            g2 = HJB.gradient(data, grid, ind, 2)

            # @show x, g1, exp_grad_1
            # @show x, g2, exp_grad_2

            @test g1[1] ≈ exp_grad_1 atol = 1e-1
            @test g1[2] ≈ exp_grad_1 atol = 1e-1
            @test g2[1] ≈ exp_grad_2 atol = 1e-1
            @test g2[2] ≈ exp_grad_2 atol = 1e-1

        end
    end

end

if false
    @testset "HJB.jl Double Integrator" begin


        # construct the grid
        x1s = range(0.0, 2.0, length = 64)
        x2s = range(-1.0, 1.5, length = 64)
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

        @show sol.sol.retcode
        @test sol.sol.retcode == HJB.DiffEq.ReturnCode.Success


    end

end
