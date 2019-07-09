using BunchKaufmanUtils, StaticArrays, LinearAlgebra
using Test

@testset "BunchKaufmanUtils.jl" begin
    θ = pi/6
    V = [cos(θ) -sin(θ); sin(θ) cos(θ)]
    D = Diagonal([-1, 1])
    A = V*D*V'
    F = bunchkaufman(A)
    v = [1,2]
    @test pseudosolve(F, v) ≈ F \ v ≈ A \ v
    AS = SMatrix{2,2}(A)
    vs = SVector{2}(v)
    FS = @inferred(bunchkaufman(AS))
    @test @inferred(pseudosolve(FS, vs)) ≈ @inferred(FS \ vs) ≈ @inferred(AS \ vs)

    A = [1 1; 1 1]
    v = [2, 2]
    F = bunchkaufman(A; check=false)
    x = pseudosolve(F, v)
    @test A*x ≈ v
end

# Tests modified from the stdlib Bunch-Kaufman tests
using Random
using LinearAlgebra: BlasComplex, BlasFloat, BlasReal, QRPivoted

@testset "stdlib tests" begin
    n = 10

    # Split n into 2 parts for tests needing two matrices
    n1 = div(n, 2)
    n2 = 2*n1

    Random.seed!(1234321)

    areal = randn(n,n)/2
    aimg  = randn(n,n)/2
    a2real = randn(n,n)/2
    a2img  = randn(n,n)/2
    breal = randn(n,2)/2
    bimg  = randn(n,2)/2

    @testset "$eltya argument A" for eltya in (Float32, Float64, Int) # , ComplexF32, ComplexF64)
        a = eltya == Int ? rand(1:7, n, n) : convert(Matrix{eltya}, eltya <: Complex ? complex.(areal, aimg) : areal)
        a2 = eltya == Int ? rand(1:7, n, n) : convert(Matrix{eltya}, eltya <: Complex ? complex.(a2real, a2img) : a2real)
        asym = transpose(a) + a                  # symmetric indefinite
        aher = a' + a                  # Hermitian indefinite
        apd  = a' * a                  # Positive-definite
        for (a, a2, aher, apd) in ((a, a2, aher, apd),
                                   (view(a, 1:n, 1:n),
                                    view(a2, 1:n, 1:n),
                                    view(aher, 1:n, 1:n),
                                    view(apd , 1:n, 1:n)))
            ε = εa = eps(abs(float(one(eltya))))

            @testset "$uplo Bunch-Kaufman factor of indefinite matrix" for uplo in (:U,) # (:U, :L)
                bc1 = bunchkaufman(Hermitian(aher, :U))
                @test bc1 \ aher ≈ Matrix(I, n, n)
                @test pseudosolve(bc1, aher) ≈ Matrix(I, n, n)
                @testset for rook in (false, true)
                    @test pseudosolve(bunchkaufman(Symmetric(transpose(a) + a, uplo), rook), transpose(a) + a) ≈ Matrix(I, n, n)
                end
            end

            @testset "$eltyb argument B" for eltyb in (Float32, Float64, Int) # ,ComplexF32, ComplexF64)
                b = eltyb == Int ? rand(1:5, n, 2) : convert(Matrix{eltyb}, eltyb <: Complex ? complex.(breal, bimg) : breal)
                for b in (b, view(b, 1:n, 1:2))
                    εb = eps(abs(float(one(eltyb))))
                    ε = max(εa,εb)

                    @testset "$uplo Bunch-Kaufman factor of indefinite matrix" for uplo in (:U,) #(:L, :U)
                        bc1 = bunchkaufman(Hermitian(aher, uplo))
                        @test aher*(bc1\b) ≈ b atol=1000ε
                        @test aher*(pseudosolve(bc1, b)) ≈ b atol=1000ε
                    end

                    @testset "$uplo Bunch-Kaufman factors of a pos-def matrix" for uplo in (:U,) #(:U, :L)
                        @testset "rook pivoting: $rook" for rook in (false, true)
                            bc2 = bunchkaufman(Hermitian(apd, uplo), rook)
                            @test pseudosolve(bc2, apd) ≈ Matrix(I, n, n)
                            @test apd*(pseudosolve(bc2, b)) ≈ b rtol=eps(cond(apd))
                        end
                    end
                end
            end
        end
    end

    @testset "test example due to @timholy in PR 15354" begin
        A = rand(6,5); A = complex(A'*A) # to avoid calling the real-lhs-complex-rhs method
        F = bunchkaufman(A);
        v6 = rand(ComplexF64, 6)
        v5 = view(v6, 1:5)
        @test pseudosolve(F, v5) ≈ F\v6[1:5]
    end
end
