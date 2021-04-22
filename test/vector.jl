Random.seed!(0)
@testset "SparseVector: Constructors B=$B T=$T" for B in [1, 2, 4, 8],
                                                    T in [Int32, Float32,
                                                     Complex{Float32},
                                                     SMatrix{2,2,Float32,2 * 2}]

    len = 10 * B
    n = bspzeros(Val(B), T, len)
    @test length(n) == len
    @test eltype(n) == T

    x = bsprand(Val(B), T, len, 0.3)
    @test length(x) == len
    @test eltype(x) == T
    if !(T <: Integer)
        y = bsprandn(Val(B), T, len, 0.3)
        @test length(y) == len
        @test eltype(y) == T
    end

    xf = Vector(x)
    @test eltype(xf) == T
    @test length(xf) == length(x)
    x′ = blocksparse(Val(B), xf)
    @test length(x′) == length(xf)
    @test eltype(x′) == T
    @test x′ == x

    xs = SparseVector(x)
    @test eltype(xs) == T
    @test length(xs) == len
    x′ = blocksparse(Val(B), xs)
    @test eltype(x′) == T
    @test x′ == x

    zf = rand(T, len)::Vector{T}
    z = blocksparse(Val(B), zf)
    @test eltype(z) == T
    @test size(z) == size(zf)
    zf′ = Array(z)
    @test eltype(zf′) == T
    @test zf′ == zf

    zs = sprand(T, len, 0.3)::SparseVector{T}
    z = blocksparse(Val(B), zs)
    @test eltype(z) == T
    @test size(z) == size(zs)
    zs′ = SparseVector(z)
    @test eltype(zs′) == T
    @test zs′ == zs
end

Random.seed!(0)
@testset "SparseVector: Collection B=$B T=$T" for B in [1, 2, 4, 8],
                                                  T in [Int32, Float32,
                                                   Complex{Float32},
                                                   SMatrix{2,2,Float32,2 * 2}]

    len = 10 * B
    x = bsprand(Val(B), T, len, 0.3)
    xs = SparseVector(x)
    @test x == xs
end

Random.seed!(0)
@testset "SparseVector: Vector space B=$B T=$T" for B in [1, 2, 4, 8],
                                                    T in [Int32, Float32,
                                                     Complex{Float32},
                                                     SMatrix{2,2,Float32,2 * 2}]

    len = 10 * B

    n = bspzeros(Val(B), T, len)
    x = bsprand(Val(B), T, len, 0.3)
    y = bsprand(Val(B), T, len, 0.3)
    z = bsprand(Val(B), T, len, 0.3)
    a = rand(eltype(T))
    b = rand(eltype(T))

    @test x == x
    @test y ≠ x

    @test +x == x
    @test -(-x) == x
    @test n + x == x
    @test x + n == x
    @test n - x == -x
    @test x - n == x
    @test x + y == y + x
    @test (x + y) + z == x + (y + z)

    @test zero(a) * x == n
    @test one(a) * x == x
    @test (-one(a)) * x == -x
    @test a * x == x * a
    @test a * (b * x) ≈ (a * b) * x

    if T <: Union{AbstractFloat,Complex{<:AbstractFloat}}
        @test inv(a) * (a * x) ≈ x
        @test x / a ≈ x * inv(a)
        @test a \ x ≈ inv(a) * x
    end

    @test (a + b) * x ≈ a * x + b * x
    @test a * (x + y) ≈ a * x + a * y
end
