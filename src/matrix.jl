export BlockSparseMatrixCSC
@computed struct BlockSparseMatrixCSC{B,T,I} <:
                 SparseArrays.AbstractSparseMatrixCSC{T,I}
    elts::SparseMatrixCSC{SMatrix{B,B,T,B^2},I}

    function BlockSparseMatrixCSC{B,T,I}(bA::SparseMatrixCSC{SMatrix{B,B,T,L},
                                                             I}) where {B,T,
                                                                        I<:Integer,
                                                                        L}
        B::Integer
        @assert L == B^2
        @assert T ≢ Any && T ≢ Union{} # TODO
        return new{B,T,I}(bA)
    end
    function BlockSparseMatrixCSC{B,T}(bA::SparseMatrixCSC{SMatrix{B,B,T,L},I}) where {B,
                                                                                       T,
                                                                                       I<:Integer,
                                                                                       L}
        return BlockSparseMatrixCSC{B,T,I}(bA)
    end
    function BlockSparseMatrixCSC{B}(bA::SparseMatrixCSC{SMatrix{B,B,T,L},I}) where {B,
                                                                                     T,
                                                                                     I<:Integer,
                                                                                     L}
        return BlockSparseMatrixCSC{B,T,I}(bA)
    end
    function BlockSparseMatrixCSC(bA::SparseMatrixCSC{SMatrix{B,B,T,L},I}) where {B,
                                                                                  T,
                                                                                  I<:Integer,
                                                                                  L}
        return BlockSparseMatrixCSC{B,T,I}(bA)
    end
end

Base.show(io::IOContext, A::BlockSparseMatrixCSC) = show(io, SparseMatrixCSC(A))
# Base.show(io::IOContext, A::BlockSparseMatrixCSC) = show(io, A.elts)

###

export blocksparse
function blocksparse(::Val{B}, Is::AbstractVector{I}, Js::AbstractVector{I},
                     Vs::AbstractVector{T}, ni::Integer,
                     nj::Integer) where {B,T,I<:Integer}
    B::Integer
    length(Is) == length(Js) == length(Vs) ||
        throw(ArgumentError("index and value vectors must have the same length"))
    ni ≥ 0 && nj ≥ 0 ||
        throw(ArgumentError("Number of elements must be non-negative"))
    @assert ni % B == 0
    @assert nj % B == 0
    ni0 = ni ÷ B
    nj0 = nj ÷ B
    Ibs = I[]
    Jbs = I[]
    Vbs = SMatrix{B,B,T,B^2}[]
    curi0 = curj0 = 0
    v0 = zero(SMatrix{B,B,T})
    for (i, j, v) in zip(Is, Js, Vs)
        @assert 1 ≤ i ≤ ni
        @assert 1 ≤ j ≤ nj
        i0, i1 = fldmod1(i, B)
        j0, j1 = fldmod1(j, B)
        if (curi0, curj0) ≠ (i0, j0)
            if curi0 ≠ 0
                push!(Ibs, curi0)
                push!(Jbs, curj0)
                push!(Vbs, v0)
            end
            curi0, curj0 = i0, j0
            v0 = zero(SMatrix{B,B,T})
        end
        v0 = setindex(v0, v, i1, j1)
    end
    if curi0 ≠ 0
        push!(Ibs, curi0)
        push!(Jbs, curj0)
        push!(Vbs, v0)
    end
    return BlockSparseMatrixCSC(sparse(Ibs, Jbs, Vbs, ni0, nj0))
end

function blocksparse(::Val{B}, A::AbstractMatrix{T}) where {B,T}
    ni, nj = size(A)
    @assert ni % B == 0
    @assert nj % B == 0
    ni0 = ni ÷ B
    nj0 = nj ÷ B
    C = Array{SMatrix{B,B,T,B^2}}(undef, ni0, nj0)
    for j0 in 1:nj0, i0 in 1:ni0
        C1 = zero(SMatrix{B,B,T})
        for j1 in 1:B, i1 in 1:B
            C1 = setindex(C1, A[B * (i0 - 1) + i1, B * (j0 - 1) + j1], i1, j1)
        end
        C[i0, j0] = C1
    end
    return BlockSparseMatrixCSC(sparse(C))
end

export blocksize
blocksize(x::BlockSparseMatrixCSC{B}) where {B} = B

for (fun, spfun) in [(:bsprand, :sprand), (:bsprandn, :sprandn)]
    @eval begin
        export $fun
        function $fun(r::AbstractRNG, ::Val{B}, ::Type{T}, ni::Integer,
                      nj::Integer, p::AbstractFloat) where {B,T}
            @assert ni % B == 0
            @assert nj % B == 0
            ni0 = ni ÷ B
            nj0 = nj ÷ B
            return BlockSparseMatrixCSC($spfun(r, SMatrix{B,B,T,B^2}, ni0, nj0,
                                               p))
        end
        function $fun(r::AbstractRNG, ::Val{B}, ni::Integer, nj::Integer,
                      p::AbstractFloat) where {B}
            return $fun(r, Val(B), Float64, ni, nj, p)
        end
        function $fun(::Val{B}, ::Type{T}, ni::Integer, nj::Integer,
                      p::AbstractFloat) where {B,T}
            return $fun(Random.default_rng(), Val(B), T, ni, nj, p)
        end
        function $fun(::Val{B}, ni::Integer, nj::Integer,
                      p::AbstractFloat) where {B}
            return $fun(Random.default_rng(), Val(B), ni, nj, p)
        end
    end
end

export bspzeros
function bspzeros(::Val{B}, ::Type{T}, ::Type{I}, ni::Integer,
                  nj::Integer) where {B,T,I<:Integer}
    B::Integer
    @assert 0 ≤ ni && ni % B == 0
    @assert 0 ≤ nj && nj % B == 0
    return BlockSparseMatrixCSC(spzeros(SMatrix{B,B,T,B^2}, fld1(ni, B),
                                        fld1(nj, B)))
end
function bspzeros(::Val{B}, ::Type{T}, ni::Integer, nj::Integer) where {B,T}
    return bspzeros(Val(B), T, Int, ni, nj)
end
function bspzeros(::Val{B}, ni::Integer, nj::Integer) where {B}
    return bspzeros(Val(B), Float64, ni, nj)
end

function (Matrix{T})(A::BlockSparseMatrixCSC{B}) where {T,B}
    R = zeros(T, size(A))
    for j0 in 1:size(A.elts, 2), k in nzrange(A.elts, j0)
        i0 = rowvals(A.elts)[k]
        v0 = nonzeros(A.elts)[k]
        for j1 in 1:B, i1 in 1:B
            R[B * (i0 - 1) + i1, B * (j0 - 1) + j1] = v0[i1, j1]
        end
    end
    return R
end
(Array{T})(A::BlockSparseMatrixCSC) where {T} = Matrix{T}(A)
Matrix(A::BlockSparseMatrixCSC{B,T}) where {B,T} = Matrix{T}(A)
Array(A::BlockSparseMatrixCSC) = Matrix(A)

### Comparison

Base.:(==)(A::BlockSparseMatrixCSC, B::BlockSparseMatrixCSC) = A.elts == B.elts

### Collection

function Base.copy!(dst::BlockSparseMatrixCSC{B},
                    src::BlockSparseMatrixCSC{B}) where {B}
    return BlockSparseMatrixCSC(copy!(dst.elts, src.elts))
end
Base.copy(A::BlockSparseMatrixCSC) = BlockSparseMatrixCSC(copy(A.elts))

Base.eltype(A::BlockSparseMatrixCSC{B,T}) where {B,T} = T
Base.eltype(::Type{<:BlockSparseMatrixCSC{B,T}}) where {B,T} = T

function Base.getindex(A::BlockSparseMatrixCSC{B}, i::Integer,
                       j::Integer) where {B}
    i0, i1 = fldmod1(i, B)
    j0, j1 = fldmod1(j, B)
    return A.elts[i0, j0][i1, j1]
end
function Base.getindex(A::BlockSparseMatrixCSC, i::CartesianIndex{2})
    return getindex(A, i[1], i[2])
end

function Base.similar(A::BlockSparseMatrixCSC; element_type)
    return BlockSparseMatrixCSC(similar(A.elts; element_type))
end

Base.size(A::BlockSparseMatrixCSC) = blocksize(A) .* size(A.elts)

### Sparse

function blocksparse(::Val{B}, A::SparseMatrixCSC{T,I}) where {B,T,I}
    # I, J, V = findnz(A)
    # TODO: Make this more efficient
    colvals = Array{I}(undef, nnz(A))
    for j in 1:size(A, 2)
        colvals[nzrange(A, j)] .= j
    end
    return blocksparse(Val(B), rowvals(A), colvals, nonzeros(A), size(A)...)
end

function (SparseMatrixCSC{T,I})(A::BlockSparseMatrixCSC{B}) where {T,B,I}
    Is = I[]
    Js = I[]
    Vs = T[]
    for j0 in 1:size(A.elts, 2), k0 in nzrange(A.elts, j0)
        i0 = rowvals(A.elts)[k0]
        v0 = nonzeros(A.elts)[k0]
        for j1 in 1:B, i1 in 1:B
            if !iszero(v0[i1, j1])
                push!(Is, B * (i0 - 1) + i1)
                push!(Js, B * (j0 - 1) + j1)
                push!(Vs, v0[i1, j1])
            end
        end
    end
    return sparse(Is, Js, Vs, size(A)...)
end
function (SparseMatrixCSC{T})(A::BlockSparseMatrixCSC{B,U,I}) where {T,B,U,I}
    return SparseMatrixCSC{T,I}(A)
end
function SparseMatrixCSC(A::BlockSparseMatrixCSC{B,T}) where {B,T}
    return SparseMatrixCSC{T}(A)
end

#TODO SparseArrays.droptol!(x::BlockSparseVector, tol) = fkeep!(x, (i, a) -> abs(a) > tol)
#TODO 
#TODO SparseArrays.dropzeros!(x::BlockSparseVector) = fkeep!(x, (i, a) -> !iszero(a))
#TODO SparseArrays.dropzeros(x::BlockSparseVector) = dropzeros!(copy(x))
#TODO 
#TODO function SparseArrays.fkeep!(x::BlockSparseVector{B}, f) where {B}
#TODO     fkeep!(x.elts, (i0, y) -> any(f(b * (i0 - 1) + i1, y[i1]) for i1 in 1:B))
#TODO     return x
#TODO end

SparseArrays.indtype(::BlockSparseMatrixCSC{B,T}) where {B,T,I} = I
SparseArrays.indtype(::Type{<:BlockSparseMatrixCSC{B,T,I}}) where {B,T,I} = I

SparseArrays.nnz(A::BlockSparseMatrixCSC) = blocksize(A)^2 * nnz(A.elts)

### Vector space

fixeltype(::T, A::SparseMatrixCSC{T}) where {T} = A
fixeltype(::T, A::SparseMatrixCSC{Any}) where {T} = SparseMatrixCSC{T}(A)

function Base.zero(A::BlockSparseMatrixCSC{B,T,I}) where {B,T,I}
    return BlockSparseMatrixCSC{B,T,I}(spzeros(SMatrix{B,B,T}, size(A.elts)))
end
Base.:+(A::BlockSparseMatrixCSC) = A # BlockSparseMatrixCSC(+A.elts)
Base.:-(A::BlockSparseMatrixCSC) = BlockSparseMatrixCSC(-A.elts)
function Base.:+(A::BlockSparseMatrixCSC{B},
                 C::BlockSparseMatrixCSC{B}) where {B}
    return BlockSparseMatrixCSC(A.elts + C.elts)
end
function Base.:-(A::BlockSparseMatrixCSC{B},
                 C::BlockSparseMatrixCSC{B}) where {B}
    return BlockSparseMatrixCSC(A.elts - C.elts)
end
function Base.:*(a::Number, A::BlockSparseMatrixCSC)
    return BlockSparseMatrixCSC(fixeltype(a * one(eltype(A.elts)), a * A.elts))
end
function Base.:\(a::Number, A::BlockSparseMatrixCSC)
    return BlockSparseMatrixCSC(fixeltype(a \ one(eltype(A.elts)), a \ A.elts))
end
function Base.:*(A::BlockSparseMatrixCSC, a::Number)
    return BlockSparseMatrixCSC(fixeltype(one(eltype(A.elts)) * a, A.elts * a))
end
function Base.:/(A::BlockSparseMatrixCSC, a::Number)
    return BlockSparseMatrixCSC(fixeltype(one(eltype(A.elts)) / a, A.elts / a))
end

### Category
function Base.one(A::BlockSparseMatrixCSC{B,T,I}) where {B,T,I}
    @assert size(A, 1) == size(A, 2)
    return BlockSparseMatrixCSC{B,T,I}(sparse(Diagonal([one(SMatrix{B,B,T})
                                                        for i in
                                                            1:size(A.elts, 1)])))
end
function Base.:*(A::BlockSparseMatrixCSC{B},
                 C::BlockSparseMatrixCSC{B}) where {B}
    @assert size(A, 2) == size(C, 1)
    return BlockSparseMatrixCSC(fixeltype(one(eltype(A.elts)) *
                                          one(eltype(C.elts)), A.elts * C.elts))
end

### Groupoid
function Base.:/(A::BlockSparseMatrixCSC{B},
                 C::BlockSparseMatrixCSC{B}) where {B}
    # A ⋅ inv(C)
    @assert size(A, 2) == size(C, 2)
    return BlockSparseMatrixCSC(A.elts / C.elts)
end
function Base.:\(A::BlockSparseMatrixCSC{B},
                 C::BlockSparseMatrixCSC{B}) where {B}
    # inv(A) ⋅ C
    @assert size(A, 1) == size(C, 1)
    return BlockSparseMatrixCSC(A.elts \ C.elts)
end

### Operator
function Base.:*(A::BlockSparseMatrixCSC{B}, x::BlockSparseVector{B}) where {B}
    @assert size(A, 2) == size(x, 1)
    return BlockSparseVector(A.elts * x.elts)
end
function Base.:\(A::BlockSparseMatrixCSC{B}, x::BlockSparseVector{B}) where {B}
    @assert size(A, 1) == size(x, 1)
    return BlockSparseVector(A.elts \ x.elts)
end

### Linear algebra
function LinearAlgebra.norm(A::BlockSparseMatrixCSC{B,T}, p::Real=2) where {B,T}
    if p == 0
        return zero(norm(zero(T), 0)^0 + norm(one(T), 0)^0) + length(A)
    elseif p == 1
        r = zero(norm(zero(T), 1) + norm(one(T), 1))
        for v0 in nonzeros(A.elts)
            for v in v0
                r += norm(v, 1)
            end
        end
        return r
    elseif p == 2
        r = zero(norm(zero(T), 2)^2 + norm(one(T), 2)^2)
        for v0 in nonzeros(A.elts)
            for v in v0
                r += norm(v, 2)^2
            end
        end
        return sqrt(r)
    elseif p == Inf
        r = zero(max(norm(zero(T), Inf), norm(one(T), Inf)))
        for v0 in nonzeros(A.elts)
            for v in v0
                r = max(r, norm(v, Inf))
            end
        end
        return r
    else
        r = zero(zero(T) + norm(one(T), p)^p)
        for v0 in nonzeros(A.elts)
            for v in v0
                r += norm(v, p)^p
            end
        end
        return r^inv(p)
    end
end
