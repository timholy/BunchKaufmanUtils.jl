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
