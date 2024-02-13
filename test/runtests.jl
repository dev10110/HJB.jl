using HJB
using Test
using StaticArrays
using LinearAlgebra


if false
    @testset "Create Grid and Indexing 1D" begin

        x1s = range(0.0, 2.0, length = 128)

        grid = HJB.Grid((x1s,))

        @test grid isa HJB.Grid{1,Float64}

        for i = 1:length(x1s)

            state = HJB.ind2state(grid, CartesianIndex(i - 1))
            @test state[1] ≈ x1s[i]
            ind = HJB.state2ind(grid, state)

            # check that it is approximately in the right spot
            @test ind in CartesianIndices(((i-2):i,))

            # @show i, x1s[i], state, ind
        end

    end

end


@testset "gradients of abs(x)" begin

    x1s = range(-4.0, 4.0, length = 129) # must be odd

    grid = HJB.Grid((x1s,))

    data = HJB.allocate_grid(grid, x -> abs(x[1]))

    @test grid isa HJB.Grid{1,Float64}

    @test size(data) == (129 + 6,) # padded one dimensional array of data

    for i in HJB.DomainIndices(grid)

        x = HJB.ind2state(grid, i)
        # @show x, data[i]

        # test that the data was filled in correctly
        @test data[i] == abs(x[1])

    end

    # update the ghost cells
    HJB.update_ghost_nodes!(data, grid)

    # check gradients everywhere
    for i in HJB.DomainIndices(grid)

        x = HJB.ind2state(grid, i)
        grad_L1 = HJB.gradient(HJB.LeftGradient(), data, grid, i, 1)
        grad_L2 = HJB.gradient(HJB.LeftWenoGradient(), data, grid, i, 1)
        grad_R1 = HJB.gradient(HJB.RightGradient(), data, grid, i, 1)
        grad_R2 = HJB.gradient(HJB.RightWenoGradient(), data, grid, i, 1)

        if x[1] < 0
            @test grad_L1 ≈ -1.0
            @test grad_R1 ≈ -1.0
            @test grad_L2 ≈ -1.0
            @test grad_R2 ≈ -1.0
        elseif x[1] > 0
            @test grad_L1 ≈ 1.0
            @test grad_R1 ≈ 1.0
            @test grad_L2 ≈ 1.0
            @test grad_R2 ≈ 1.0
        else
            @test grad_L1 ≈ -1.0
            @test grad_R1 ≈ 1.0
            @test grad_L2 ≈ -1.0
            @test grad_R2 ≈ 1.0
            @show i, x, grad_L1, grad_L2, grad_R1, grad_R2
        end

    end


end

# @testset "HJB.jl Double Integrator" begin
# 
# 
#     # construct the grid
#     x1s = range(0.0, 2.0, length = 128)
#     x2s = range(-1.0, 1.5, length = 128)
#     grid = HJB.Grid((x1s, x2s))
# 
#     @test grid isa HJB.Grid{2,Float64}
#     println("grid created")
# 
#     # define the dynamics
#     function f(t, x, u)
#         return @SVector [x[2], u[1]]
#     end
# 
#     # define the Hamiltonian
#     function H(t, x, V, DxV, params)
# 
#         u1 = @SVector [1.0]
#         u2 = @SVector [-1.0]
# 
#         H1 = dot(DxV, f(t, x, u1))
#         H2 = dot(DxV, f(t, x, u2))
# 
#         return min(0, max(H1, H2))
# 
#     end
# 
#     # construct the terminal function
#     function l(x)
#         return 1.5 - x[1]
#     end
# 
# 
#     # construct the HJB Problem
#     tspan = (0.0, 2.0)
#     prob = HJB.HJBProblem(H, l, tspan)
#     println("problem created")
# 
#     # solve the problem
#     sol = HJB.solve_MOL(prob, grid; reltol = 1e-1, abstol = 1e-1)
#     println("problem solved")
# 
#     @show sol.sol.retcode
#     @test sol.sol.retcode == HJB.DiffEq.ReturnCode.Success
# 
# 
# end
