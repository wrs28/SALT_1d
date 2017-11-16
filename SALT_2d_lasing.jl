# Here's the status of the code: the jacobian array still needs explicit construction.
# The diagonal block piece is done, but the derivatives with respect to frequency and
# the non-zero index still need work.


function computeLasing(solve_single_mode_lasing(inputs::InputStruct, D₀::Float64,
    k_init::Array{Float64,1}, ψ_init::Array{Complex128,1}; ind::Int=14, dispOpt::Bool=false,
    iter::Int=250, ftol::Float64=2e-8)::Tuple{Array{Complex128,2},Array{Float64,1}}

    N = prod(inputs.N_ext)
    M = length(k_init)
    ind = inputs.x̄_inds[ind]
    ɛ_sm = inputs.ɛ_sm

    ∇² = laplacian(inputs.k₀, inputs)
    ɛ = sparse(1:N, 1:N, ɛ_sm[:], N, N, +)

    f!(Ψ::Array{Float64,1}, fvec) = lasing_residual(Ψ, fvec, N, M, ∇², ε, ind)
    jac!(Ψ::Array{Float64,1}, jacarray) = lasing_jacobian(Ψ, fvec, N, M, ∇², ε, ind)

    Ψ_init = Array(Float64,2N*M)
    for m in 1:M
        Ψ_init[2N*(m-1) + (1:N)] = real(ψ_init[:,m])
        Ψ_init[2N*(m-1)+N+(1:N)] = imag(ψ_init[:,m])
        Ψ_init[2N*(m-1)+N+ind] = k_init[m]
    end

    df = DifferentiableSparseMultivariateFunction(f!,jac!)
    z = nlsolve(df, Ψ_init, show_trace=dispOpt, ftol=ftol, iterations=iter)

    ψ = Array{Complex128}(N,M)
    k = Array{Float64,M}
    if converged(z)
        for m in 1:M
            ψ[:,m] = z.zero[2N*(m-1) + (1:N)] + 1im*z.zero[2N*(m-1)+N+(1:N)]
            ψ[ind,m] = z.zero[2N*(m-1) + ind]
            k[m] = z.zero[2N*(m-1)+N+ind]
        end
    else
        println("Warning, did not converge. Returning NaN.")
        ψ = NaN*ψ
        k = NaN*ψ
    end

    return ψ, k
end



################################################################################
###
################################################################################


"""
"""
function lasing_residual(Ψ::Array{Float64,1}, fvec::Array{Float64,1}, N::Int,
                                                        M::Int, ∇², ε, ind::Int)

    ψ = Array{Complex128}(N, M)
    k = Array{Float64}(M)
    for m in 1:M
        ψ[1:N, m] = Ψ[2N*(m-1)+(1:N)] + 1im*Ψ[2N*(m-1)+N+(1:N)]
        ψ[ind, m] = Ψ[2N*(m-1) + ind]
        k[m] = Ψ[2N*(m-1)+N+ind]
    end

    k² = k.^2
    for m in 1:M
        residual = (∇²+(ɛ+χ(ψ,k,m,inputs))*k²[m])*ψ[:,m]/ψ[ind,m]
        fvec[2N*(m-1) + (1:N)] = real(residual)
        fvec[2N*(m-1)+N+(1:N)] = imag(residual)
    end
end


"""
"""
function lasing_jacobian(Ψ::Array{Float64,1}, jacarray, N::Int, M::Int, ∇², ε, ind::Int)

    ψ = Array{Complex128}(N,M)
    k = Array{Float64}(M)
    for m in 1:M
        ψ[1:N, m] = Ψ[2N*(m-1)+(1:N)] + 1im*Ψ[2N*(m-1)+N+(1:N)]
        ψ[ind, m] = Ψ[2N*(m-1) + ind]
        k[m] = Ψ[2N*(m-1)+N+ind]
    end

    k² = k.^2

    W1 = Array{Int}(4N*M^2)
    W2 = Array{Int}(4N*M^2)
    T = Array{Float64}(4N*M^2)

    X1 = Array{Float64}(2N*M)
    X2 = Array{Float64}(2N*M)
    U = Array{Float64}()

    Y1 = Array{Float64}(2N*M^2)
    Y2 = Array{Float64}(2N*M^2)
    V = Array{Float64}()

    for m in 1:M, m′ in 1:M

        tempr = zeros(Float64,N)
        tempi = zeros(Float64,N)
        dχdk = zeros(Complex128,N)
        χdk²dk = zeros(Complex128,N)

        if m == m′
            temp = (∇²+(ɛ+χ(ψ,k[m],inputs))*k²[m])/ψ[ind,m]
            tempr = real(temp.nzval)
            tempi = imag(temp.nzval)
            dkχ = 2*(ɛ+χ(ψ,k,m,inputs))*k[m]*ψ[:,m]/ψ[ind,m]
        end

        W1[4N*M*(m′-1) + 4N*(m-1) + 0N + (1:N)] = N*(m-1) + 0N + (1:N)
        W1[4N*M*(m′-1) + 4N*(m-1) + 1N + (1:N)] = N*(m-1) + 1N + (1:N)
        W1[4N*M*(m′-1) + 4N*(m-1) + 2N + (1:N)] = N*(m-1) + 0N + (1:N)
        W1[4N*M*(m′-1) + 4N*(m-1) + 3N + (1:N)] = N*(m-1) + 1N + (1:N)

        W2[4N*M*(m′-1) + 4N*(m-1) + 0N + (1:N)] = N*(m-1) + 0N + (1:N)
        W2[4N*M*(m′-1) + 4N*(m-1) + 1N + (1:N)] = N*(m-1) + 0N + (1:N)
        W2[4N*M*(m′-1) + 4N*(m-1) + 2N + (1:N)] = N*(m-1) + 1N + (1:N)
        W2[4N*M*(m′-1) + 4N*(m-1) + 3N + (1:N)] = N*(m-1) + 1N + (1:N)

        T[4N*M*(m′-1) + 4N*(m-1) + 0N + (1:N)] = +tempr + χʳʳ′(ψ, k, inputs, ind, m, m′)*k²[m] + real(dχdk) + real(χdk²dk)
        T[4N*M*(m′-1) + 4N*(m-1) + 1N + (1:N)] = +tempi + χⁱʳ′(ψ, k, inputs, ind, m, m′)*k²[m] + imag(dχdk) + imag(χdk²dk)
        T[4N*M*(m′-1) + 4N*(m-1) + 2N + (1:N)] = -tempi + χʳⁱ′(ψ, k, inputs, ind, m, m′)*k²[m] + real(dχdk) + real(χdk²dk)
        T[4N*M*(m′-1) + 4N*(m-1) + 3N + (1:N)] = +tempr + χⁱⁱ′(ψ, k, inputs, ind, m, m′)*k²[m] + imag(dχdk) + imag(χdk²dk)

        T[4N*M*(m′-1) + 4N*(m-1) + 0N + ind] = 0
        T[4N*M*(m′-1) + 4N*(m-1) + 1N + ind] = 0
        T[4N*M*(m′-1) + 4N*(m-1) + 2N + ind] = 0
        T[4N*M*(m′-1) + 4N*(m-1) + 3N + ind] = 0

        X1[2N*(m-1) + (1:2N)] = 2N*(m-1) + (1:2N)
        X2[2N*(m-1) + (1:2N)] = fill(2N*(m-1)+ind,2N)

        Y1[2N*M*(m-1) + (1:2N)] = 2N*(m-1) + (1:2N)
        Y2[2N*M*(m-1) + (1:2N)] = 2N*(m-1) + (1:2N)

        W1[]



        tempj = [tempr+χʳʳ′(ψ,k,ind,inputs)*k²+ψ_normʳʳ′(ψ,ω) -tempi+χʳⁱ′(ψ,ω)*ω²+ψ_normʳⁱ′(ψ,ω); tempi+χⁱʳ′(ψ,ω)*ω²+ψ_normⁱʳ′(ψ,ω) tempr+χⁱⁱ′(ψ,ω)*ω²+ψ_normⁱⁱ′(ψ,ω)]
        wemp = 2*ω*(ɛ+χ(ψ,ω)+Γ)*ψ/ψ_norm(ψ,inds) + χʷ′(ψ,ω)*ω²*ψ/ψ_norm(ψ,inds)

        tempj[:,1] = [real(wemp); imag(wemp)]
        jacarray = sparse(INDS1,INDS2,V,4*N^2*M^2,4*N^2*M^2,+)
    end
end


"""
"""
function χ(ψ::Array{Complex128,2}, k::Array{Float64,1}, m::Int,
                    inputs::InputStruct)::SparseMatrixCSC{Complex{Float64},Int64}
    N = prod(inputs.N_ext); D₀ = inputs.D₀; F_sm = inputs.F_sm
    h = hb(ψ,k,inputs)
    V = D₀*F_sm[:].*γ(k[m],inputs)./(1+h)
    return sparse(1:N, 1:N, V, N, N, +)
end


"""
"""
function hb(ψ::Array{Complex128,2}, k::Array{Float64,1},
                    inputs::InputStruct)::Array{Float64,1}
    N = prod(inputs.N_ext); M = length(k)
    h = zeros(Float64,N)
    for m in 1:M
        h += abs2.(γ.(k[m],inputs)*ψ[:,m])
    end
    return h
end


"""
"""
function hbʳ′(ψ::Array{Complex128,2}, k::Array{Float64,1}, m::Int,
                    inputs::InputStruct)::Array{Float64,1}
    h′ = 2*abs2(γ(k[m],inputs))*real(ψ[:,m])
    return h′
end


"""
"""
function hbⁱ′(ψ::Array{Complex128,2}, k::Array{Float64,1}, m::Int,
                    inputs::InputStruct)::Array{Float64,1}
    h′ = 2*abs2(γ(k[m],inputs))*imag(ψ[:,m])
    return h′
end


"""
"""
function χʳʳ′(ψ::Array{Complex128,2}, k::Array{Complex128,1}, ind::Int, m::Int, m′::Int,
                    inputs::InputStruct)::Array{Float64,1}
    N = prod(inputs.N_ext); M = length(k); D₀ = inputs.D₀; F_sm = inputs.F_sm; γt = γ.(k, inputs)
    h = hb(ψ,k,inputs)
    hʳ′ = hbʳ′(ψ,k,m′,inputs)
    V = -D₀.*F_sm[:].*hʳ′.*(real(γt).*real.(ψ[:,m]) - imag(γt).*imag.(ψ[:,m]))./(1+h).^2/ψ[ind,m]
end


"""
"""
function χⁱʳ′(ψ::Array{Complex128,2}, k::Array{Complex128,1}, ind::Int, m::Int, m′::Int,
                     inputs::InputStruct)
    N = prod(inputs.N_ext); M = length(k); D₀ = inputs.D₀; F_sm = inputs.F_sm; γt = γ.(k, inputs)
    h = hb(ψ,k,inputs)
    hʳ′ = hbʳ′(ψ,k,m′,inputs)
    V = -D₀.*F_sm[:].*hʳ′.*(imag(γt).*real.(ψ[:,m]) + real(γt).*imag.(ψ[:,m]))./(1+h).^2/ψ[ind,m]
end


"""
"""
function χʳⁱ′(ψ::Array{Complex128,2}, k::Array{Complex128,1}, ind::Int, m::Int, m′::Int,
                     inputs::InputStruct)
    N = prod(inputs.N_ext); M = length(k); D₀ = inputs.D₀; F_sm = inputs.F_sm; γt = γ.(k, inputs)
    h = hb(ψ,k,inputs)
    hⁱ′ = hbⁱ′(ψ,k,m′,inputs)
    V = -D₀.*F_sm[:].*hⁱ′.*(real(γt).*real.(ψ[:,m]) - imag(γt).*imag.(ψ[:,m]))./(1+h).^2/ψ[ind,m]
end


"""
"""
function χⁱⁱ′(ψ::Array{Complex128,2}, k::Array{Complex128,1}, ind::Int, m::Int, m′::Int,
                     inputs::InputStruct)
     N = prod(inputs.N_ext); M = length(k); D₀ = inputs.D₀; F_sm = inputs.F_sm; γt = γ.(k, inputs)
     h = hb(ψ,k,inputs)
     hⁱ′ = hbⁱ′(ψ,k,m′,inputs)
     V = -D₀.*F_sm[:].*hⁱ′.*(imag(γt).*real.(ψ[:,m]) + real(γt).*imag.(ψ[:,m]))./(1+h).^2/ψ[ind,m]
end


"""
"""
function χᵏ′(ψ::Array{Complex128,2}, k::Array{Float64,1}, m::Int, m′::Int,
                    inputs::InputStruct)::Array{Complex128,1}
    N = prod(inputs.N_ext); M = length(k); F_sm = inputs.F_sm; D₀ = inputs.D₀
    γt = γ(k[m],inputs); γt′ = γ′(k[m′],inputs); Γt′= Γ′(k[m′],inputs)
    h = hb(ψ,k,inputs)
    V = (m==m′)*D₀.*F_sm[:].*γ′./(1+h) - D₀.*F_sm[:].*γt.*Γt′.*abs2.(ψ[:,m′])./(1+h).^2
end


"""
"""
function γ′(k::Complex128, inputs::InputStruct)::Complex128
    return -inputs.γ⟂/(k-inputs.k₀+1im*inputs.γ⟂)^2
end


"""
"""
function Γ′(k::Complex128, inputs::InputStruct)::Complex128
    return -2*(inputs.γ⟂^2)*(k-inputs.k₀)/((k-inputs.k₀)^2 + inputs.γ⟂^2)^2
end



function norm_indʳʳ′(Ψ, k, ind, inputs)
    k² = k^2; N = prod(inputs.N_ext)
    rows = zeros(Int64,length(inds)*N)
    cols = zeros(Int64,length(inds)*N)
    V = zeros(Complex128,length(inds)*N)
    for i in 1:length(inds)
        rows[N_ext*(i-1) + (1:N_ext)] = 1:N_ext
        cols[N_ext*(i-1) + (1:N_ext)] = fill(inds[i],N_ext)
        V[N_ext*(i-1) + (1:N_ext)] = -2.*real((∇²+ɛ*ω²+χ(Ψ,ω)*ω²+Γ*ω²)*Ψ).*real(Ψ[inds[i]])./(abs2.(Ψ[ind]))^2
    end
    return sparse(rows, cols, V, N_ext, N_ext, +)
end

function ψ_normⁱʳ′(Ψ,ω)
    ω² = ω^2
    rows = zeros(Int64,length(inds)*N_ext)
    cols = zeros(Int64,length(inds)*N_ext)
    V = zeros(Complex128,length(inds)*N_ext)
    for i in 1:length(inds)
        rows[N_ext*(i-1) + (1:N_ext)] = 1:N_ext
        cols[N_ext*(i-1) + (1:N_ext)] = fill(inds[i],N_ext)
        V[N_ext*(i-1) + (1:N_ext)] = -2.*imag((∇²+ɛ*ω²+χ(Ψ,ω)*ω²+Γ*ω²)*Ψ).*real(Ψ[inds[i]])./(abs2.(Ψ[ind]))^2
    end
    return sparse(rows, cols, V, N_ext, N_ext, +)
end

function ψ_normʳⁱ′(Ψ,ω)
    ω² = ω^2
    rows = zeros(Int64,length(inds)*N_ext)
    cols = zeros(Int64,length(inds)*N_ext)
    V = zeros(Complex128,length(inds)*N_ext)
    for i in 1:length(inds)
        rows[N_ext*(i-1) + (1:N_ext)] = 1:N_ext
        cols[N_ext*(i-1) + (1:N_ext)] = fill(inds[i],N_ext)
        V[N_ext*(i-1) + (1:N_ext)] = -2.*real((∇²+ɛ*ω²+χ(Ψ,ω)*ω²+Γ*ω²)*Ψ).*imag(Ψ[inds[i]])./(abs2.(Ψ[ind]))^2
    end
    return sparse(rows, cols, V, N_ext, N_ext, +)
end

function ψ_normⁱⁱ′(Ψ,ω)
    ω² = ω^2
    rows = zeros(Int64,length(inds)*N_ext)
    cols = zeros(Int64,length(inds)*N_ext)
    V = zeros(Complex128,length(inds)*N_ext)
    for i in 1:length(inds)
        rows[N_ext*(i-1) + (1:N_ext)] = 1:N_ext
        cols[N_ext*(i-1) + (1:N_ext)] = fill(inds[i],N_ext)
        V[N_ext*(i-1) + (1:N_ext)] = -2.*imag((∇²+ɛ*ω²+χ(Ψ,ω)*ω²+Γ*ω²)*Ψ).*imag(Ψ[inds[i]])./(abs2.(Ψ[ind]))^2
    end
    return sparse(rows, cols, V, N_ext, N_ext, +)
end
