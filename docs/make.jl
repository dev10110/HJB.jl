using HJB
using Documenter

DocMeta.setdocmeta!(HJB, :DocTestSetup, :(using HJB); recursive=true)

makedocs(;
    modules=[HJB],
    authors="Devansh Ramgopal Agrawal <devansh@umich.edu> and contributors",
    repo="https://github.com/dev10110/HJB.jl/blob/{commit}{path}#{line}",
    sitename="HJB.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://dev10110.github.io/HJB.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/dev10110/HJB.jl",
    devbranch="main",
)
