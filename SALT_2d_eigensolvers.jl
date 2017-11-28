"""
k,ψ =  computeK_L_core(inputs, k; nk=1, F=[1.], truncate=false, ψ_init=[])

    Compute eigenfrequencies k w/o line pulling, according to boundary conditions
    set in inputs.bc

    Set D or F to zero to get passive cavity.

    k[# of poles]

    ψ[cavity size, # of poles]
"""
function computeK_L_core(inputs::InputStruct, k::Complex128;
    nk::Int=1, F::Array{Float64,1}=[1.], truncate::Bool=false,
    ψ_init::Array{Complex128,1}=Complex128[])::Tuple{Array{Complex128,1},Array{Complex128,2}}

    ∇² = laplacian(k,inputs)

    N = prod(inputs.N_ext); ε_sm = inputs.ε_sm; D₀ = inputs.D₀; F_sm = inputs.F_sm
    ɛ⁻¹ = sparse(1:N, 1:N, 1./(ɛ_sm[:]-1im*D₀.*F.*F_sm[:]), N, N, +)
    if isempty(ψ_init)
        (k²,ψ,nconv,niter,nmult,resid) = eigs(-ɛ⁻¹*∇²,which = :LM, nev = nk,
            sigma = k^2)
    else
        (k²,ψ,nconv,niter,nmult,resid) = eigs(-ɛ⁻¹*∇²,which = :LM, nev = nk,
            sigma = k^2, v0 = ψ_init)
    end

    r = inputs.r_ext; inds = inputs.x̄_inds
    if length(F)>1
        F_temp = F[r[inds]]
    else
        F_temp = F
    end

     for ii = 1:nk
        ψ[:,ii] = ψ[:,ii]/sqrt( trapz((ɛ_sm[inds]-1im*D₀.*F_temp.*F_sm[inds]).*abs2.(ψ[inds,ii]), inputs.dx̄) )
    end

    if truncate
        return sqrt.(k²),ψ[inds,:]
    else
        return sqrt.(k²),ψ
    end
end #end of function computeK_L_core


"""
k,ψ =  computePole_L(inputs, k; np=1, F=1, truncate = false, ψ_init = [])

    Compute poles w/o line pulling, using purely outoing PML's.

    Set D or F to zero to get passive cavity.

    k[# of poles]

    ψ[cavity size, # of poles]
"""
function computePole_L(inputs1::InputStruct, k::Union{Complex128,Float64,Int64}; np::Int=1,
    F::Array{Float64,1}=[1.], truncate::Bool = false,
    ψ_init::Array{Complex128,1}=Complex128[])::Tuple{Array{Complex128,1},Array{Complex128,2}}

    inputs = open_to_pml_out(inputs1)

    K, ψ = computeK_L_core(inputs, complex(1.0*k); nk=np, F=F, truncate=truncate, ψ_init = ψ_init)
end #end of function computePole_L


"""
k,ψ = computeZero_L(inputs, k; nz=1, F=1, truncate = false, ψ_init = [])

    Compute zeros w/o line pulling, using purely incoming PML's.

    Set D or F to zero to get passive cavity.

    k[# of poles]

    ψ[cavity size, # of poles]
"""
function computeZero_L(inputs1::InputStruct, k::Union{Complex128,Float64,Int}; nz::Int=1,
    F::Array{Float64,1}=[1.], truncate::Bool=false,
    ψ_init::Array{Complex128,1}=Complex128[])::Tuple{Array{Complex128,1},Array{Complex128,2}}

    inputs = open_to_pml_in(inputs1)

    K, ψ = computeK_L_core(inputs, complex(1.0*k); nk=nz, F=F, truncate=truncate, ψ_init=ψ_init)
end # end of function computeZero_L


"""
k,ψ = computeUZR_L(inputs,k; nu=1, F=1, truncate = false, ψ_init = [], direction = [1,0])

    Compute UZRs w/o line pulling, using purely PML's.

    If direction[1] = +1, have right-going. If -1, left-going. If 0, leave bc's
    as indicated in inputs.

    Set D or F to zero to get passive cavity.

    k[# of poles]

    ψ[cavity size, # of poles]
"""
function computeUZR_L(inputs1::InputStruct, k::Union{Complex128,Float64,Int}; nu::Int=1,
    F::Array{Float64,1}=[1.], truncate::Bool=false,
    ψ_init::Array{Complex128,1}=Complex128[],
    direction::Array{Int,1}=[1,0])::Tuple{Array{Complex128,1},Array{Complex128,2}}

    inputs = deepcopy(inputs1)
    if (direction[1]==+1) && (inputs.bc[1:2] !== ["I", "O"])
        updateInputs!(inputs, :bc, ["pml_in", "pml_out", inputs.bc[3], inputs.bc[4]])
    elseif (direction[1]==-1) && (inputs.bc[1:2] !== ["O", "I"])
        updateInputs!(inputs, :bc, ["pml_out", "pml_in", inputs.bc[3], inputs.bc[4]])
    end

    if (direction[2]==+1) && (inputs.bc[3:4] !== ["I", "O"])
        updateInputs!(inputs, :bc, [inputs.bc[1], inputs.bc[2], "pml_in", "pml_out"])
    elseif (direction[2]==-1) && (inputs.bc[3:4] !== ["O", "I"])
        updateInputs!(inputs, :bc, [inputs.bc[1], inputs.bc[2], "pml_out", "pml_in"])
    end

    K, ψ = computeK_L_core(inputs, complex(1.0*k); nk=nu, F=F, truncate=truncate, ψ_init = ψ_init)
end # end of function computeUZR_L


"""
η,u = computeCF(inputs, k; nCF=1, F=[1.], η_init=0., u_init=[], truncate=false)

    Compute CF state according to bc set in inputs.
"""
function computeCF(inputs::InputStruct,k::Complex128; nCF::Int=1,
    F::Array{Float64,1}=[1.], η_init::Complex128=complex(0.),
    u_init::Array{Complex128,1}=Complex128[],
    truncate::Bool=false)::Tuple{Array{Complex128,1},Array{Complex128,2}}

    k²= k^2

    ∇² = laplacian(k, inputs)

    N = inputs.N_ext; ε_sm = inputs.ε_sm; F_sm = inputs.F_sm
    ɛk² = sparse(1:prod(N), 1:prod(N), ɛ_sm[:]*k²       , prod(N), prod(N), +)
    sF  = sparse(1:prod(N), 1:prod(N), sign.(F.*F_sm[:]), prod(N), prod(N), +)
    FF  = sparse(1:prod(N), 1:prod(N), abs.(F.*F_sm[:]) , prod(N), prod(N), +)

    if isempty(u_init)
        (η,u,nconv,niter,nmult,resid) = eigs(-sF*(∇²+ɛk²)./k²,FF, which = :LM,
            nev = nCF, sigma = η_init)
    elseif !isempty(u_init)
        (η,u,nconv,niter,nmult,resid) = eigs(-sF*(∇²+ɛk²)./k²,FF, which = :LM,
            nev = nCF, sigma = η_init, v0 = u_init)
    end

    for ii = 1:nCF
        u[:,ii] = u[:,ii]/sqrt( trapz( u[:,ii].*F.*F_sm[:].*u[:,ii], inputs.dx̄ ) )
    end

    if truncate
        return η,u[inputs.x̄_inds,:]
    else
        return η,u
    end
end #end of function computeCF


"""
k, u, η, conv = computeK_NL1(inputs, k_init; F=[1.], dispOpt=false,η_init = 0., u_init = [],
    k_avoid = 0., tol = .5, max_count = 15, max_iter = 50)

    compute eigenfrequency k with dispersion via root-finding of η(k) - D₀γ(k).

    max_count: the max number of CF states to compute at a time.
    max_iter: maximum number of iterations to include in nonlinear solve
"""
function computeK_NL1(inputs::InputStruct, k_init::Union{Complex128,Float64,Int}; F::Array{Float64,1}=[1.],
    dispOpt::Bool=false,η_init::Complex128=complex(0.),
    u_init::Array{Complex128,1}=Complex128[],
    k_avoid::Array{Complex128,1}=Complex128[0], tol::Float64=.5, max_count::Int = 15,
    max_iter::Int=50)::Tuple{Array{Complex128,1},Array{Complex128,2},Array{Complex128,1},Bool}

    ηt, ut = computeCF(inputs, complex(k_init); nCF = 1, η_init=η_init, u_init = u_init)

    γ⟂ = inputs.γ⟂; k₀ = inputs.k₀; D₀ = inputs.D₀; dx̄ = inputs.dx̄
    function f!(z, fvec)

        k = z[1]+1im*z[2]

        flag = true
        count = 1
        M = 1
        ind = Int

        while flag

            η_temp, u_temp = computeCF(inputs, k; nCF = M, F=F, η_init=ηt[1], u_init = ut[:,1])

            overlap = zeros(Float64,M)
            for i in 1:M
                overlap[i] = abs(trapz(ut[:,1].*inputs.F_sm[:].*F.*u_temp[:,i],dx̄))
            end

            max_overlap, ind = findmax(overlap)

            if (max_overlap > (1-tol))
                flag = false
                ηt[1] = η_temp[ind]
                ut[:,1] = u_temp[:,ind]
            elseif  (count < max_count) & (max_overlap ≤ (1-tol))
                M += 1
            else
                flag = false
                ηt[1] = η_temp[ind]
                ut[:,1] = u_temp[:,ind]
                println("Warning: overlap less than tolerance, skipped to neighboring TCF.")
            end

            count += 1

        end

        fvec[1] = real((ηt[1]-D₀*γ(k,inputs))/prod(k-k_avoid))
        fvec[2] = imag((ηt[1]-D₀*γ(k,inputs))/prod(k-k_avoid))

    end

    z = nlsolve(f!,[real(k_init),imag(k_init)]; iterations = max_iter, show_trace = dispOpt)
    k = z.zero[1]+1im*z.zero[2]
    conv = converged(z)

    ηt[1] = D₀*γ(k,inputs)
    η, ψ = computeCF(inputs, k; nCF = 1, η_init=ηt[1], F=F, u_init = ut[:,1])

    return [k], ψ, η, conv
end # end of function computeK_NL1


"""
k, ψ, η, conv = computePole_NL1(inputs, k_init; F=[1.], dispOpt=false, η_init = 0., u_init = [],
    k_avoid = 0., tol = .5, max_count = 15, max_iter = 50)

    compute pole with dispersion via root-finding of η(k) - D₀γ(k).

    max_count: the max number of CF states to compute at a time.
    max_iter: maximum number of iterations to include in nonlinear solve
"""
function computePole_NL1(inputs1::InputStruct, k_init::Union{Complex128,Float64,Int}; F::Array{Float64,1}=[1.],
    dispOpt::Bool=false, η_init::Complex128=complex(0.),
    u_init::Array{Complex128,1}=Complex128[],
    k_avoid::Array{Complex128,1}=Complex128[0], tol::Float64=.5, max_count::Int=15,
    max_iter::Int=50)::Tuple{Array{Complex128,1},Array{Complex128,2},Array{Complex128,1},Bool}

    inputs = open_to_pml_out(inputs1)

    k, ψ, η, conv = computeK_NL1(inputs, complex(k_init); F=F, dispOpt=dispOpt, η_init=η_init,
        u_init=u_init, k_avoid=k_avoid, tol=tol, max_count=max_count, max_iter=max_iter)
end # end of function computePole_NL1


"""
k, ψ, η, conv = computeZero_NL1(inputs, k_init;F=[1.], dispOpt=false, η_init=0., u_init=[],
    k_avoid = 0., tol = .5, max_count = 15, max_iter = 50)

    compute zero with dispersion via root-finding of η(k) - D₀γ(k).

    max_count: the max number of CF states to compute at a time.
    max_iter: maximum number of iterations to include in nonlinear solve
"""
function computeZero_NL1(inputs1::InputStruct, k_init::Union{Complex128,Float64,Int}; F::Array{Float64,1}=[1.],
    dispOpt::Bool=false, η_init::Complex128=complex(0.),
    u_init::Array{Complex128,1}=Complex128[],
    k_avoid::Array{Complex128,1}=Complex128[0], tol::Float64=.5, max_count::Int=15,
    max_iter::Int=50)::Tuple{Array{Complex128,1},Array{Complex128,2},Array{Complex128,1},Bool}

    inputs = open_to_pml_in(inputs1)

    k, ψ, η, conv = computeK_NL1(inputs, complex(k_init); F=F, dispOpt=dispOpt, η_init=η_init,
        u_init=u_init, k_avoid=k_avoid, tol=tol, max_count=max_count, max_iter=max_iter)
end # end of function computeZero_NL1


"""
k, ψ, η, conv = computeUZR_NL1(inputs,k_init;F=[1.],dispOpt=false,η_init = 0., u_init = [],
    k_avoid = 0., tol = .5, max_count = 15, max_iter = 50, direction = [1,0])

    compute UZR with dispersion via root-finding of η(k) - D₀γ(k).

    If direction[1] = +1, have right-going. If -1, left-going. If 0, leave bc's
    as indicated in inputs.

    max_count: the max number of CF states to compute at a time.
    max_iter: maximum number of iterations to include in nonlinear solve
"""
function computeUZR_NL1(inputs1::InputStruct, k_init::Union{Complex128,Float64,Int}; F::Array{Float64,1}=[1.],
    dispOpt::Bool=false, η_init::Complex128=complex(0.),
    u_init::Array{Complex128,1}=Complex128[], k_avoid::Array{Complex128,1}=Complex128[0],
    tol::Float64=.5, max_count::Int=15, max_iter::Int=50,
    direction::Array{Int,1}=[1,0])::Tuple{Array{Complex128,1},Array{Complex128,2},Array{Complex128,1},Bool}

    inputs = deepcopy(inputs1)
    if direction[1]==1 && inputs1.bc[[1,2]] !== ["I", "O"]
        updateInputs!(inputs, :bc, ["pml_in", "pml_out", inputs.bc[3], inputs.bc[4]])
    elseif direction[1]==-1 && inputs1.bc[[1,2]] !== ["O", "I"]
        updateInputs!(inputs, :bc, ["pml_out", "pml_in", inputs.bc[3], inputs.bc[4]])
    end

    if direction[2]==1 && inputs1.bc[[3,4]] !== ["I", "O"]
        updateInputs!(inputs, :bc, [inputs.bc[1], inputs.bc[2], "pml_in", "pml_out"])
    elseif direction[2]==-1 && inputs1.bc[[3,4]] !== ["O", "I"]
        updateInputs!(inputs, :bc, [inputs.bc[1], inputs.bc[2], "pml_out", "pml_in"])
    end

    k, ψ, η, conv = computeK_NL1(inputs, k_init; F=F, dispOpt=dispOpt, η_init=η_init,
        u_init=u_init, k_avoid=k_avoid, tol=tol, max_count=max_count, max_iter=max_iter)
end # end of function computeUZR_NL1


"""
k = computeK_NL2(inputs, kc, Radii; nk=3, Nq=100, F=[1.], R_min=.01, rank_tol=1e-8)

    Compute eigenfrequency with dispersion, using contour integration. BC's set
    by inputs.bc

    Contour is centered on kc, Radii = (x-diameter, y-diameter).

    nk is an upper bound on the number of eigenfrequencies contained in the contour.

    Nq is the number of contour quadrature points.
"""
function computeK_NL2(inputs::InputStruct, kc::Union{Complex128,Float64,Int},
    Radii::Tuple{Float64,Float64}; nk::Int=3, Nq::Int=100, F::Array{Float64,1}=[1.],
    R_min::Float64=.01, rank_tol::Float64=1e-8)::Array{Complex{Float64},1}

    k = Complex128(kc)

    ∇² = laplacian(k,inputs)

    N_ext = prod(inputs.N_ext)
    A  = zeros(Complex128,N_ext,nk)
    A₀ = zeros(Complex128,N_ext,nk)
    A₁ = zeros(Complex128,N_ext,nk)
    M = rand(N_ext,nk)

    ϕ = 2π*(0:1/Nq:(1-1/Nq))
    Ω = k + Radii[1]*cos.(ϕ) + 1im*Radii[2]*sin.(ϕ)

    θ =  angle(inputs.k₀ - 1im*inputs.γ⟂-k)
    flag = abs(inputs.k₀ - 1im*inputs.γ⟂-k) < rad(Radii[1],Radii[2],θ)

    if flag
        AA  = zeros(Complex128,N_ext,nk)
        AA₀ = zeros(Complex128,N_ext,nk)
        AA₁ = zeros(Complex128,N_ext,nk)
        RR = 2*R_min
        ΩΩ = k₀-1im*γ⟂ + (RR/2)*cos.(ϕ) + 1im*(RR/2)*sin.(ϕ)
    end

    ε_sm = inputs.ε_sm; F_sm = inputs.F_sm; D₀ = inputs.D₀; dx̄ = inputs.dx̄
    for i in 1:Nq

        k′ = Ω[i]
        k′² = k′^2

        if (i > 1) & (i < Nq)
            dk′ = (Ω[i+1] - Ω[i-1]  )/2
        elseif i == Nq
            dk′ = (Ω[1]   - Ω[end-1])/2
        elseif i == 1
            dk′ = (Ω[2]   - Ω[end]  )/2
        end

        ɛk′² = sparse(1:N_ext, 1:N_ext, ɛ_sm[:]*k′², N_ext, N_ext, +)
        χk′² = sparse(1:N_ext, 1:N_ext, D₀*γ(k′,inputs)*F.*F_sm[:]*k′², N_ext, N_ext, +)

        A = (∇²+ɛk′²+χk′²)\M
        A₀ += A*dk′/(2π*1im)
        A₁ += A*k′*dk′/(2π*1im)

        if flag
            kk′ = ΩΩ[i]
            kk′² = kk′^2
            if (i > 1) & (i < Nq)
                dkk′ = (ΩΩ[i+1] - ΩΩ[i-1]  )/2
            elseif i == Nq
                dkk′ = (ΩΩ[1]   - ΩΩ[end-1])/2
            elseif i == 1
                dkk′ = (ΩΩ[2]   - ΩΩ[end]  )/2
            end
            ɛkk′² = sparse(1:N_ext, 1:N_ext, ɛ_sm[:]*kk′², N_ext, N_ext, +)
            χkk′² = sparse(1:N_ext, 1:N_ext, D₀*γ(kk′,inputs)*F.*F_sm[:]*kk′², N_ext, N_ext, +)

            AA = (∇²+ɛkk′²+χkk′²)\M
            AA₀ += AA*dkk′/(2π*1im)
            AA₁ += AA*kk′*dkk′/(2π*1im)

       end

    end

    if flag
        A₀ = A₀-AA₀
        A₁ = A₁-AA₁
    end

    P = svdfact(A₀,thin = true)
    temp = find(P[:S] .< rank_tol)
    if isempty(temp)
        println("Error. Need more nevals")
        return Complex128[NaN]
    else
        k = temp[1]-1
    end

    B = (P[:U][:,1:k])'*A₁*(P[:Vt][1:k,:])'*diagm(1./P[:S][1:k])

    D,V = eig(B)

    return D
end # end of function computeK_NL2


"""
computePole_NL2(inputs, kc, Radii; np=3, Nq=100, F=[1.], R_min=.01, rank_tol=1e-8)

    Compute poles with dispersion, using contour integration and outgoing PML's.

    Contour is centered on kc, Radii = (x-diameter, y-diameter).

    np is an upper bound on the number of eigenfrequencies contained in the contour.

    Nq is the number of contour quadrature points.
"""
function computePole_NL2(inputs1::InputStruct, kc::Union{Complex128,Float64,Int},
    Radii::Tuple{Float64,Float64}; np::Int=3, Nq::Int=100, F::Array{Float64,1}=[1.],
    R_min::Float64=.01, rank_tol::Float64=1e-8)::Array{Complex{Float64},1}

    inputs = open_to_pml_out(inputs1)

    k = computeK_NL2(inputs, kc, Radii; nk=np, Nq=Nq, F=F, R_min=R_min, rank_tol=rank_tol)
end # end of function computePole_NL2


"""
computeZero_NL2(inputs, kc, Radii; nz=3, Nq=100, F=[1.], R_min=.01, rank_tol=1e-8)

    Compute zeros with dispersion, using contour integration and incoming PML's.

    Contour is centered on kc, Radii = (x-diameter, y-diameter).

    nz is an upper bound on the number of eigenfrequencies contained in the contour.

    Nq is the number of contour quadrature points.
"""
function computeZero_NL2(inputs1::InputStruct, kc::Union{Complex128,Float64,Int},
    Radii::Tuple{Float64,Float64}; nz::Int=3, Nq::Int=100, F::Array{Float64,1}=[1.],
    R_min::Float64=.01, rank_tol::Float64=1e-8)::Array{Complex{Float64},1}

    inputs = open_to_pml_in(inputs1)

    k = computeK_NL2(inputs, kc, Radii; nk=nz, Nq=Nq, F=F, R_min=R_min, rank_tol=rank_tol)
end # end of function computeZero_NL2


"""
computeUZR_NL2(inputs, kc, Radii; nz=3, Nq=100, F=[1.], R_min=.01, rank_tol=1e-8, direction = [1,0])

    Compute UZR's with dispersion, using contour integration and outoing/incoming PML's.

    Contour is centered on kc, Radii = (x-diameter, y-diameter).

    nz is an upper bound on the number of eigenfrequencies contained in the contour.

    Nq is the number of contour quadrature points.

    If direction[1] = +1, have right-going. If -1, left-going. If 0, leave bc's
    as indicated in inputs.
"""
function computeUZR_NL2(inputs1::InputStruct, kc::Union{Complex128,Float64,Int},
    Radii::Tuple{Float64,Float64}; nu::Int=3, Nq::Int=100, F::Array{Float64,1}=[1.],
    R_min::Float64=.01, rank_tol::Float64=1e-8,
    direction::Array{Int,1}=[1,0])::Array{Complex{Float64},1}

    inputs = deepcopy(inputs1)
    if direction[1]==1 && inputs1.bc[[1,2]] !== ["I", "O"]
        updateInputs!(inputs, :bc, ["pml_in", "pml_out", inputs.bc[3], inputs.bc[4]])
    elseif direction[1]==-1 && inputs1.bc[[1,2]] !== ["O", "I"]
        updateInputs!(inputs, :bc, ["pml_out", "pml_in", inputs.bc[3], inputs.bc[4]])
    end

    if direction[2]==1 && inputs1.bc[[3,4]] !== ["pml_in", "pml_out"]
        updateInputs!(inputs, :bc, [inputs.bc[1], inputs.bc[2], "pml_in", "pml_out"])
    elseif direction[2]==-1 && inputs1.bc[[3,4]] !== ["pml_out", "pml_in"]
        updateInputs!(inputs, :bc, [inputs.bc[1], inputs.bc[2], "pml_out", "pml_in"])
    end

    k = computeK_NL2(inputs, kc, Radii; nk=nu, Nq=Nq, F=F, R_min=R_min, rank_tol=rank_tol)
end # end of function computeUZR_NL2


"""
rad(a,b,θ) is the complex radius as a function of angle for an ellipse with
    semi-major axis a and semi-minor axis b.

    For use in contour eigensolver.
"""
function rad(a::Float64, b::Float64, θ::Float64)::Float64
    return b./sqrt.(sin(θ).^2+(b/a)^2.*cos(θ).^2)
end
