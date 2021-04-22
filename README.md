# BlockSparseArrays.jl

The
[`BlockSparseArrays.jl`](https://github.com/eschnett/BlockSparseArrays.jl)
package provides block-sparse vectors and matrices.

* [![Documenter](https://img.shields.io/badge/docs-dev-blue.svg)](https://eschnett.github.io/BlockSparseArrays.jl/dev)
* [![GitHub
  CI](https://github.com/eschnett/BlockSparseArrays.jl/workflows/CI/badge.svg)](https://github.com/eschnett/BlockSparseArrays.jl/actions)
* [![Codecov](https://codecov.io/gh/eschnett/BlockSparseArrays.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/eschnett/BlockSparseArrays.jl)

## Current state

What this package needs next is a generic sparse LU decomposition
implemented in Julia. "Generic" here means that it must work with
arbitrary types, not just the standard floating-point types. Without
such an implementation, it will not be possible to be more efficient
than standard Julia sparse arrays when solving linear systems.

## Related work

This package is similar to
[BlockSparseMatrices.jl](https://github.com/KristofferC/BlockSparseMatrices.jl).
Different from that packge, this package here encodes the block size
in the type, which can create more efficient code.
