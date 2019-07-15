module BunchKaufmanUtils

using LinearAlgebra, StaticArrays

export pseudosolve, ispossemidef

# Piracy, move to StaticArrays
struct SBunchKaufman{N,T,DT,UT} <: Factorization{T}
    p::SVector{N,Int}
    D::DT
    U::UT
end

Base.size(F::SBunchKaufman{N}) where N = (N, N)
Base.size(F::SBunchKaufman{N}, dim::Integer) where N = dim < 3 ? N : 1

function Base.invperm(a::StaticVector{N,Int}) where N
    n = length(a)
    b = MVector{N,Int}(zeros(Int, n))
    @inbounds for (i, j) in enumerate(a)
        ((1 <= j <= n) && b[j] == 0) ||
            throw(ArgumentError("argument is not a permutation"))
        b[j] = i
    end
    return b
end

function LinearAlgebra.bunchkaufman(A::SMatrix{N,N,T}, rook::Bool=false; check=true) where {N,T}
    F = bunchkaufman(Matrix(A), rook; check=check)
    D, U = F.D, F.U
    return SBunchKaufman{N,eltype(F),typeof(D),typeof(U)}(SVector{N,Int}(F.p), D, U)
end

function Base.:*(F::Union{BunchKaufman,SBunchKaufman}, B::AbstractVecOrMat)
    D, U, p = F.D, F.U, F.p
    return permrows(U*(D*(U'*permrows(B, F.p))), invperm(F.p))
end

function LinearAlgebra.:\(F::SBunchKaufman{N}, B::Union{StaticVector{N},StaticMatrix{N}}) where N
    D, U, p = F.D, F.U, F.p
    X = U' \ (D \ (U \ permrows(B, invperm(p))))
    return similar_type(B, eltype(X))(permrows(X, p))
end

function pseudosolve(F::Union{BunchKaufman,SBunchKaufman}, B::AbstractVecOrMat; tol=eps(real(eltype(F)))*10*size(B, 1))
    D, U, p = F.D, F.U, F.p
    n = size(D, 1)
    Y = U \ permrows(B, p)
    dthresh = tol*max(maximum(abs.(D.d)), isempty(D.du) ? zero(eltype(D.du)) : maximum(abs.(D.du)))
    i = 1
    while i <= n
        if i == n || iszero(D.du[i])
            solve1!(Y, i, D.d[i], dthresh)
            i += 1
        else
            solve2!(Y, i, D.d[i], D.du[i], D.d[i+1], dthresh)
            i += 2
        end
    end
    X = U' \ Y
    return simtype(B, eltype(X))(permrows(X, invperm(p)))
end

function ispossemidef(F::Union{BunchKaufman,SBunchKaufman}; tol=eps(real(eltype(F)))*10*size(F, 1))
    D, U, p = F.D, F.U, F.p
    dthresh = tol*max(maximum(abs.(D.d)), maximum(abs.(D.du)))
    i, n = 1, size(F, 1)
    while i <= n
        if i == n || iszero(D.du[i])
            D.d[i] < -dthresh && return false
            i += 1
        else
            λ1, λ2, _ = symeig(D.d[i], D.du[i], D.d[i+1])
            (λ1 < -dthresh || λ2 < -dthresh) && return false
            i += 2
        end
    end
    return true
end

## Utilities
permrows(v::AbstractVector, p) = v[p]
permrows(M::AbstractMatrix, p) = M[p,:]

simtype(B, ::Type{T}) where T = identity
simtype(B::StaticArray, ::Type{T}) where T = similar_type(B, T)

function solve1!(y::AbstractVector, i, d, dthresh)
    if abs(d) > dthresh
        y[i] /= d
    else
        y[i] = 0
    end
    return y
end

function solve1!(Y::AbstractMatrix, i, d, dthresh)
    if abs(d) > dthresh
        for j in axes(Y, 2)
            Y[i,j] /= d
        end
    else
        for j in axes(Y, 2)
            Y[i,j] = 0
        end
    end
    return Y
end

function solve2!(y::AbstractVector, i, di, dui, di1, dthresh)
    λ1, λ2, V = symeig(di, dui, di1)
    y2 = SVector(y[i], y[i+1])
    vy = V*y2
    dinvvy = SVector(abs(λ1) >= dthresh ? vy[1]/λ1 : zero(eltype(vy)),
                     abs(λ2) >= dthresh ? vy[2]/λ2 : zero(eltype(vy)))
    x = V'*dinvvy
    y[i], y[i+1] = x
    return y
end

function solve2!(Y::AbstractMatrix, i, di, dui, di1, dthresh)
    λ1, λ2, V = symeig(di, dui, di1)
    for j in axes(Y, 2)
        y2 = SVector(Y[i,j], Y[i+1,j])
        vy = V*y2
        dinvvy = SVector(abs(λ1) >= dthresh ? vy[1]/λ1 : zero(eltype(vy)),
                         abs(λ2) >= dthresh ? vy[2]/λ2 : zero(eltype(vy)))
        x = V'*dinvvy
        Y[i,j], Y[i+1,j] = x
    end
    return Y
end

function symeig(a::Real, b::Real, d::Real)
    @assert !iszero(b)
    T, D = a + d, a*d-b^2
    if T >= 0
        λ1 = T/2 + sqrt((a-d)^2/4 + b^2)
        λ2 = D/λ1
    else
        λ2 = T/2 - sqrt((a-d)^2/4 + b^2)
        λ1 = D/λ2
    end
    n1, n2 = sqrt((λ1-d)^2 + b^2), sqrt(b^2 + (λ2-a)^2)
    V = @SMatrix [(λ1-d)/n1 b/n2; b/n1 (λ2-a)/n2]
    return λ1, λ2, V
end

end # module
