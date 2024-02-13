
function get_plotting_data(
    grid::G,
    data,
    idxs = (1, 2),
    at = zeros(F, D),
) where {D,F,G<:HJB.Grid{D,F}}

    # determine the axes to slice
    ax1, ax2 = idxs

    # grab the reference point index
    ref_ind_ = HJB.state2ind(grid, at)

    # Determine the indices to slice the data
    all_inds = HJB.DomainIndices(grid)
    lower_ind = all_inds[1]
    upper_ind = all_inds[end]

    # construct the indices to slice over the data
    for i = 1:D
        if (i != ax1 && i != ax2)
            lower_ind = Base.setindex(lower_ind, ref_ind_[i], i)
            upper_ind = Base.setindex(upper_ind, ref_ind_[i], i)
        end
    end

    inds = lower_ind:upper_ind

    # slice over the data
    vals = data[inds]

    # permute and dropdims
    dims = [ax1, ax2, (i for i = 1:D if (i != ax1 && i != ax2))...]
    permuted_vals = permutedims(vals, dims)
    dropped_vals = dropdims(permuted_vals; dims = tuple(3:D...))

    # finally get the x and y axes
    xs = HJB.getDomainAxes(grid, ax1)
    ys = HJB.getDomainAxes(grid, ax2)

    return xs, ys, dropped_vals'

end

# define a specialization for 2D grids
function get_plotting_data(grid::G, data) where {F,G<:HJB.Grid{2,F}}
    xs = HJB.getDomainAxes(grid, 1)
    ys = HJB.getDomainAxes(grid, 2)
    inds = HJB.DomainIndices(grid)
    vals = data[inds]
    return xs, ys, vals'
end


## Now define the main function to convert data into a series that can be plotted


@userplot HJB_Plot # so the function is called hjb_plot

@recipe function f(h::HJB_Plot)

    if !((length(h.args) == 2) || (length(h.args) == 4))
        error("hjb_plot should be given two or four args")
    end

    if !(typeof(h.args[1]) <: HJB.Grid)
        error(
            "the first argument of hjb_plot must be a `HJB.Grid`. Got $(typeof(h.args[1])) ",
        )
    end

    # parse the datastructures
    grid_ = h.args[1]
    data_ = h.args[2]

    # handle the 1D case first
    if typeof(grid_) <: HJB.Grid{1}

        xs = HJB.getDomainAxes(grid_, 1)
        ys = data_[HJB.DomainIndices(grid_)]

        @series begin
            xlabel --> "x1"
            seriestype --> :path
            xs, ys
        end
        return

        # handle the 2D case
    elseif typeof(grid_) <: HJB.Grid{2}

        xs, ys, vals = get_plotting_data(grid_, data_)
        xlabel = "x1"
        ylabel = "x2"

        @series begin
            seriestype --> :heatmap
            xlabel --> xlabel
            ylabel --> ylabel
            xs, ys, vals
        end
        return

        # handle all other dimensions
    else

        if !(length(h.args) == 4)
            error("hjb_plot for a grid of dimension > 2 should be called with four args")
        end

        idxs = h.args[3]
        at = h.args[4]

        xs, ys, vals = get_plotting_data(grid_, data_, idxs, at)

        xlabel = "x$(idxs[1])"
        ylabel = "x$(idxs[2])"

        @series begin
            seriestype --> :heatmap # default is heatmap can be overridden easily
            xlabel --> xlabel
            ylabel --> ylabel
            xs, ys, vals
        end
        return

    end

end


## provide some documentation

"""
    hjb_plot(grid::G, data; idxs=(1,2), at=zeros(D), kwargs...) where {D, F, G<:Grid{D, F}}

A plotting utility to plot the solution of a hjb. 
Provide the grid, the data, and if the data is of dim > 2, also the dimensions you want to slice in, and the reference point `at` that you want to slice through. 

Examples:

```
# create the grids
x1s = -1:0.025:1
x2s = -2:0.025:2
x3s = 0.0:0.5:2.0
grid1 = HJB.Grid((x1s,))
grid2 = HJB.Grid((x1s, x2s,))
grid3 = HJB.Grid((x1s, x2s, x3s))

# fill in some data
data1 = HJB.allocate_grid(grid1, x-> x[1]^2);
data2 = HJB.allocate_grid(grid2, x-> x[1]^2 + x[2]^2);
data3 = HJB.allocate_grid(grid3, x-> x[1]^2 + x[2]^2 - x[3]);

# plot 1D:
hjb_plot(grid1, data1)

# plot 2D:
hjb_plot(grid2, data2)

# plot 3D:
hjb_plot(grid3, data3, (1, 2), [0, 0, 1.0])

"""
hjb_plot
