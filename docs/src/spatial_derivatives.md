# Spatial Derivatives

Once we have a grid, we can take spatial derivatives using various methods. 

The syntax is 
```
  gradient(data, grid, ind, dim)
  gradient(method, data, grid, ind, dim)
```
which allows you determine the gradient of the data stored in `data`, based on the `grid` at a position `ind` along dimension `dim`. You can also specify the `method` to use. The mathematical expressions for each are listed in the docs below.  

All of the gradient methods are sufficiently general to work in arbitrary dimensions, but assumes constant spacing of grid nodes.

## Spatial Derivatives API

```@autodocs; canonical=false
Modules = [HJB]
Pages = ["spatial_derivatives.jl"]
```
