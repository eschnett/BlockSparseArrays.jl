Random.seed!(0)
@testset "SparseMatrixCSC: Constructors B=$B T=$T" for B in [1, 2, 4, 8],
                                                       T in [Int32, Float32,
                                                        Complex{Float32},
                                                        SMatrix{2,2,Float32,
                                                                2 * 2}]

    leni = 9 * B
    lenj = 11 * B
    N = bspzeros(Val(B), T, leni, lenj)
    @test size(N) == (leni, lenj)
    @test eltype(N) == T

    A = bsprand(Val(B), T, leni, lenj, 0.3)
    @test size(A) == (leni, lenj)
    @test eltype(A) == T
    if !(T <: Integer)
        C = bsprandn(Val(B), T, leni, lenj, 0.3)
        @test size(C) == (leni, lenj)
        @test eltype(C) == T
    end

    Af = Matrix(A)
    @test eltype(Af) == T
    @test size(Af) == size(A)
    A′ = blocksparse(Val(B), Af)
    @test size(A′) == size(Af)
    @test eltype(A′) == T
    @test A′ == A

    As = SparseMatrixCSC(A)
    @test eltype(As) == T
    @test size(As) == (leni, lenj)
    A′ = blocksparse(Val(B), As)
    @test eltype(A′) == T
    @test A′ == A

    zf = rand(T, leni, lenj)::Matrix{T}
    z = blocksparse(Val(B), zf)
    @test size(z) == size(zf)
    @test eltype(z) == T
    zf′ = Array(z)
    @test eltype(zf′) == T
    @test zf′ == zf

    zs = sprand(T, leni, lenj, 0.3)::SparseMatrixCSC{T}
    z = blocksparse(Val(B), zs)
    @test size(z) == size(zs)
    @test Matrix(zs) == Matrix(z)
    @test eltype(z) == T
    zs′ = SparseMatrixCSC(z)
    @test eltype(zs′) == T
    @test zs′ == zs
end

Random.seed!(0)
@testset "SparseMatrixCSC: Collection B=$B T=$T" for B in [1, 2, 4, 8],
                                                     T in [Int32, Float32,
                                                      Complex{Float32},
                                                      SMatrix{2,2,Float32,
                                                              2 * 2}]

    leni = 9 * B
    lenj = 11 * B
    A = bsprand(Val(B), T, leni, lenj, 0.3)
    As = SparseMatrixCSC(A)
    # @test A == As
    for i in eachindex(A)
        @test A[i] == As[i]
    end
end

Random.seed!(0)
@testset "SparseMatrixCSC: Vector space B=$B T=$T" for B in [1, 2, 4, 8],
                                                       T in [Int32, Float32,
                                                        Complex{Float32},
                                                        SMatrix{2,2,Float32,
                                                                2 * 2}]

    leni = 9 * B
    lenj = 11 * B

    N = bspzeros(Val(B), T, leni, lenj)
    A = bsprand(Val(B), T, leni, lenj, 0.3)
    C = bsprand(Val(B), T, leni, lenj, 0.3)
    D = bsprand(Val(B), T, leni, lenj, 0.3)
    a = rand(eltype(T))
    b = rand(eltype(T))

    @test A == A
    @test C ≠ A

    @test +A == A
    @test -(-A) == A
    @test N + A == A
    @test A + N == A
    @test N - A == -A
    @test A - N == A
    @test A + C == C + A
    @test (A + C) + D == A + (C + D)

    @test zero(a) * A == N
    @test one(a) * A == A
    @test (-one(a)) * A == -A
    @test a * A == A * a
    @test a * (b * A) ≈ (a * b) * A

    if T <: Union{AbstractFloat,Complex{<:AbstractFloat}}
        @test inv(a) * (a * A) ≈ A
        @test A / a ≈ A * inv(a)
        @test a \ A ≈ inv(a) * A
    end

    @test (a + b) * A ≈ a * A + b * A
    @test a * (A + C) ≈ a * A + a * C
end

Random.seed!(0)
@testset "SparseMatrixCSC: Category B=$B T=$T" for B in [1, 2, 4, 8],
                                                   T in [Int32, Float32,
                                                    Complex{Float32},
                                                    SMatrix{2,2,Float32,2 * 2}]

    len = 9 * B

    N = bspzeros(Val(B), T, len, len)
    E = one(N)
    A = bsprand(Val(B), T, len, len, 0.3)
    C = bsprand(Val(B), T, len, len, 0.3)
    D = bsprand(Val(B), T, len, len, 0.3)
    a = rand(eltype(T))
    b = rand(eltype(T))

    @test N * A == N
    @test A * N == N

    @test E * A == A
    @test A * E == A

    @test (A * C) * D ≈ A * (C * D)

    @test a * (A * C) ≈ (a * A) * C

    @test (A + C) * D ≈ A * D + C * D
    @test A * (C + D) ≈ A * C + A * D
end

Random.seed!(0)
@testset "SparseMatrixCSC: Operator B=$B T=$T" for B in [1, 2, 4, 8],
                                                   T in [Int32, Float32,
                                                    Complex{Float32},
                                                    SMatrix{2,2,Float32,2 * 2}]

    len = 9 * B

    n = bspzeros(Val(B), T, len)
    x = bsprand(Val(B), T, len, 0.3)
    y = bsprand(Val(B), T, len, 0.3)
    N = bspzeros(Val(B), T, len, len)
    E = one(N)
    A = bsprand(Val(B), T, len, len, 0.3)
    C = bsprand(Val(B), T, len, len, 0.3)
    a = rand(eltype(T))
    b = rand(eltype(T))

    @test N * x == n
    @test E * x == x

    @test A * n == n

    @test A * (x + y) ≈ A * x + A * y
    @test (A + C) * x ≈ A * x + C * x

    @test (a * A) * x ≈ a * (A * x)
    @test (A * a) * x ≈ A * (a * x)

    if !(T <: Integer)          # doesn't make sense otherwise
        if T <: Union{AbstractFloat,Complex} # UMFPACK only supports some types
            # We can't do LU decompositions! We lost.
            # r = A \ x
            # @test A * r ≈ x
        end
    end
end
