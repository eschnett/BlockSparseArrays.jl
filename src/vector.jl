export BlockSparseVector
struct BlockSparseVector{B,T,I} <: AbstractSparseVector{T,I}
    elts::SparseVector{SVector{B,T},I}

    function BlockSparseVector{B,T,I}(bx::SparseVector{SVector{B,T},I}) where {B,
                                                                               T,
                                                                               I<:Integer}
        B::Integer
        return new{B,T,I}(bx)
    end
    function BlockSparseVector{B,T}(bx::SparseVector{SVector{B,T},I}) where {B,
                                                                             T,
                                                                             I<:Integer}
        return BlockSparseVector{B,T,I}(bx)
    end
    function BlockSparseVector{B}(bx::SparseVector{SVector{B,T},I}) where {B,T,
                                                                           I<:Integer}
        return BlockSparseVector{B,T,I}(bx)
    end
    function BlockSparseVector(bx::SparseVector{SVector{B,T},I}) where {B,T,
                                                                        I<:Integer}
        return BlockSparseVector{B,T,I}(bx)
    end
end

Base.show(io::IOContext, x::BlockSparseVector) = show(io, SparseVector(x))

###

export blocksparse
function blocksparse(::Val{B}, Is::AbstractVector{I}, Vs::AbstractVector{T},
                     n::Integer) where {B,T,I<:Integer}
    B::Integer
    length(Is) == length(Vs) ||
        throw(ArgumentError("index and value vectors must have the same length"))
    n ≥ 0 || throw(ArgumentError("Number of elements must be non-negative"))
    @assert n % B == 0
    nb = n ÷ B
    Ibs = I[]
    Vbs = SVector{B,T}[]
    curi0 = 0
    v0 = zero(SVector{B,T})
    for (i, v) in zip(Is, Vs)
        @assert 1 ≤ i ≤ n
        i0, i1 = fldmod1(i, B)
        if curi0 ≠ i0
            if curi0 ≠ 0
                push!(Ibs, curi0)
                push!(Vbs, v0)
            end
            curi0 = i0
            v0 = zero(SVector{B,T})
        end
        v0 = setindex(v0, v, i1)
    end
    if curi0 ≠ 0
        push!(Ibs, curi0)
        push!(Vbs, v0)
    end
    return BlockSparseVector(sparsevec(Ibs, Vbs, nb))
end

function blocksparse(::Val{B}, x::AbstractVector{T}) where {B,T}
    n = length(x)
    @assert n % B == 0
    n0 = n ÷ B
    y = Array{SVector{B,T}}(undef, n0)
    for i0 in 1:n0
        y1 = zero(SVector{B,T})
        for i1 in 1:B
            y1 = setindex(y1, x[B * (i0 - 1) + i1], i1)
        end
        y[i0] = y1
    end
    return BlockSparseVector(sparsevec(y))
end

export blocksize
blocksize(x::BlockSparseVector{B}) where {B} = B

for (fun, spfun) in [(:bsprand, :sprand), (:bsprandn, :sprandn)]
    @eval begin
        export $fun
        function $fun(r::AbstractRNG, ::Val{B}, ::Type{T}, n::Integer,
                      p::AbstractFloat) where {B,T}
            @assert n % B == 0
            n0 = n ÷ B
            return BlockSparseVector($spfun(r, SVector{B,T}, n0, p))
        end
        function $fun(r::AbstractRNG, ::Val{B}, n::Integer,
                      p::AbstractFloat) where {B}
            return $fun(r, Val(B), Float64, n, p)
        end
        function $fun(::Val{B}, ::Type{T}, n::Integer,
                      p::AbstractFloat) where {B,T}
            return $fun(Random.default_rng(), Val(B), T, n, p)
        end
        function $fun(::Val{B}, n::Integer, p::AbstractFloat) where {B}
            return $fun(Random.default_rng(), Val(B), n, p)
        end
    end
end

export bspzeros
function bspzeros(::Val{B}, ::Type{T}, ::Type{I},
                  n::Integer) where {B,T,I<:Integer}
    B::Integer
    @assert 0 ≤ n && n % B == 0
    return BlockSparseVector(spzeros(SVector{B,T}, fld1(n, B)))
end
function bspzeros(::Val{B}, ::Type{T}, n::Integer) where {B,T}
    return bspzeros(Val(B), T, Int, n)
end
bspzeros(::Val{B}, n::Integer) where {B} = bspzeros(Val(B), Float64, n)

function (Vector{T})(x::BlockSparseVector{B}) where {T,B}
    r = zeros(T, size(x))
    for j in nzrange(x.elts, 1)
        i0 = rowvals(x.elts)[j]
        v0 = nonzeros(x.elts)[j]
        for i1 in 1:B
            r[B * (i0 - 1) + i1] = v0[i1]
        end
    end
    return r
end
(Array{T})(x::BlockSparseVector) where {T} = Vector{T}(x)
Vector(x::BlockSparseVector{B,T}) where {B,T} = Vector{T}(x)
Array(x::BlockSparseVector) = Vector(x)

### Comparison

Base.:(==)(x::BlockSparseVector, y::BlockSparseVector) = x.elts == y.elts

### Collection

function Base.copy!(dst::BlockSparseVector{B},
                    src::BlockSparseVector{B}) where {B}
    return BlockSparseVector(copy!(dst.elts, src.elts))
end
Base.copy(x::BlockSparseVector) = BlockSparseVector(copy(x.elts))

Base.eltype(x::BlockSparseVector{B,T}) where {B,T} = T
Base.eltype(::Type{<:BlockSparseVector{B,T}}) where {B,T} = T

function Base.getindex(x::BlockSparseVector{B}, i::Integer) where {B}
    i0, i1 = fldmod1(i, B)
    return x.elts[i0][i1]
end

function Base.similar(x::BlockSparseVector; element_type)
    return BlockSparseVector(similar(x.elts; element_type))
end

Base.size(x::BlockSparseVector) = blocksize(x) .* size(x.elts)

### Sparse

function blocksparse(::Val{B}, x::SparseVector) where {B}
    return blocksparse(Val(B), rowvals(x), nonzeros(x), length(x))
end

function (SparseVector{T,I})(x::BlockSparseVector{B}) where {T,B,I}
    Is = I[]
    Vs = T[]
    for k0 in nzrange(x.elts, 1)
        i0 = rowvals(x.elts)[k0]
        v0 = nonzeros(x.elts)[k0]
        for i1 in 1:B
            if !iszero(v0[i1])
                push!(Is, B * (i0 - 1) + i1)
                push!(Vs, v0[i1])
            end
        end
    end
    return sparsevec(Is, Vs, length(x))
end
function (SparseVector{T})(x::BlockSparseVector{B,U,I}) where {T,B,U,I}
    return SparseVector{T,I}(x)
end
SparseVector(x::BlockSparseVector{B,T}) where {B,T} = SparseVector{T}(x)

function SparseArrays.droptol!(x::BlockSparseVector, tol)
    return fkeep!(x, (i, a) -> abs(a) > tol)
end

SparseArrays.dropzeros!(x::BlockSparseVector) = fkeep!(x, (i, a) -> !iszero(a))
SparseArrays.dropzeros(x::BlockSparseVector) = dropzeros!(copy(x))

function SparseArrays.fkeep!(x::BlockSparseVector{B}, f) where {B}
    fkeep!(x.elts, (i0, y) -> any(f(b * (i0 - 1) + i1, y[i1]) for i1 in 1:B))
    return x
end

SparseArrays.indtype(::BlockSparseVector{B,T}) where {B,T,I} = I
SparseArrays.indtype(::Type{<:BlockSparseVector{B,T,I}}) where {B,T,I} = I

SparseArrays.nnz(x::BlockSparseVector) = blocksize(x) * nnz(x.elts)

### Vector space

function Base.zero(x::BlockSparseVector{B,T,I}) where {B,T,I}
    return BlockSparseVector{B,T,I}(spzeros(SVector{B,T}, size(x.elts)))
end
Base.:+(x::BlockSparseVector) = x # BlockSparseVector(+x.elts)
Base.:-(x::BlockSparseVector) = BlockSparseVector(-x.elts)
function Base.:+(x::BlockSparseVector{B}, y::BlockSparseVector{B}) where {B}
    return BlockSparseVector(x.elts + y.elts)
end
function Base.:-(x::BlockSparseVector{B}, y::BlockSparseVector{B}) where {B}
    return BlockSparseVector(x.elts - y.elts)
end
Base.:*(a::Number, x::BlockSparseVector) = BlockSparseVector(a * x.elts)
Base.:\(a::Number, x::BlockSparseVector) = BlockSparseVector(a \ x.elts)
Base.:*(x::BlockSparseVector, a::Number) = BlockSparseVector(x.elts * a)
Base.:/(x::BlockSparseVector, a::Number) = BlockSparseVector(x.elts / a)

### Linear algebra
function LinearAlgebra.norm(A::BlockSparseVector{B,T}, p::Real=2) where {B,T}
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
