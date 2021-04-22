# Generate documentation with this command:
# (cd docs && julia --color=yes make.jl)

push!(LOAD_PATH, "..")

using BlockSparseArrays
using Documenter

makedocs(; sitename="BlockSparseArrays", format=Documenter.HTML(),
         modules=[BlockSparseArrays])

deploydocs(; repo="github.com/eschnett/BlockSparseArrays.jl.git",
           devbranch="main", push_preview=true)
