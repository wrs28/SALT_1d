module SALT_1d

export processInputs, updateInputs!, computeCF, computePole_L, computeZero_L, computeUZR_L, computeK_NL1, computePole_NL1, computeZero_NL1, computeUZR_NL1, computeK_NL2, computePole_NL2, computeZero_NL2, computeUZR_NL2, computePsi, computeG, computeS
#solve_SPA, solve_scattered, solve_single_mode_lasing, solve_CPA, computeS, solve_CPA,
# bootstrap

include("SALT_1d_Core.jl")

using .Core

using NLsolve
using Formatting

####################################################################################


"""
η,u =  computeCF(inputs, k, nCFs; F=1., η_init = [], u_init = [])

Compute CF states w/o line pulling. Uses bc's specified in inputs file.

η[ # of CF states]

u[ cavity size, # of CF states]

"""
function computeCF(inputs::InputStruct, k::Complex128, nCF::Int; F=1., η_init = [], u_init = [])::Tuple{Array{Complex128,1},Array{Complex128,2}}

    ## DEFINITIONS BLOCK ##
    N_ext = inputs.N_ext
    dx = inputs.dx
    x_ext = inputs.x_ext
    x_inds = inputs.x_inds
    ∂_ext = inputs.∂_ext
    ɛ_ext = inputs.ɛ_sm
    F_ext = inputs.F_sm
    Γ_ext = inputs.Γ_ext
    Γ = zeros(Complex128,N_ext,1)
    for dd in 3:length(∂_ext)-2
        δ,dummy1 = dirac_δ(x_ext,∂_ext[dd])
        Γ = Γ[:]  + full(δ)/Γ_ext[dd]
    end
    k²= k^2
    ##

    if isempty(size(F))
        F = [F]
    end

    ∇² = laplacian(k, inputs)

    ɛk² = sparse( 1:N_ext, 1:N_ext, ɛ_ext[:]*k²      , N_ext, N_ext, +)
    Γk² = sparse( 1:N_ext, 1:N_ext, Γ*k²          , N_ext, N_ext, +)
    sF  = sparse( 1:N_ext, 1:N_ext, sign.(F.*F_ext), N_ext, N_ext, +)
    FF  = sparse( 1:N_ext, 1:N_ext, abs.(F.*F_ext) , N_ext, N_ext, +)

    ## eigenvalue solve
    if isempty(η_init) & isempty(u_init)
        (η,u_ext,nconv,niter,nmult,resid) = eigs(-sF*(∇²+ɛk²+Γk²)/k²,FF,
            which = :LM, nev = nCF, sigma = 1e-8)
    elseif !isempty(η_init) & isempty(u_init)
        (η,u_ext,nconv,niter,nmult,resid) = eigs(-sF*(∇²+ɛk²+Γk²)/k²,FF,
            which = :LM, nev = nCF, sigma = η_init)
    elseif !isempty(η_init) & !isempty(u_init)
        (η,u_ext,nconv,niter,nmult,resid) = eigs(-sF*(∇²+ɛk²+Γk²)/k²,FF,
            which = :LM, nev = nCF, sigma = η_init, v0 = u_init)
    end

    u = zeros(Complex128,inputs.N_ext,nCF)

    ## normalization
    for ii = 1:nCF
        N = trapz(u_ext[:,ii].*F[:].*F_ext[:].*u_ext[:,ii],dx)
        u[:,ii] = u_ext[:,ii]/sqrt(N)
    end

    return η,u

end # end of function computeCFs




"""
k,ψ =  computePole_L(inputs, k, np; F=1, truncate = false, ψ_init = [])

Compute poles w/o line pulling, using purely outoing PML's.

Set D or F to zero to get passive cavity.

k[ # of poles]

ψ[ cavity size, # of poles]

"""
function computePole_L(inputs1::InputStruct, k::Complex128, np::Int; F=1., truncate = false, ψ_init = [])

    # set outgoing boundary conditions, using PML implementation
    if !inputs1.bc==["pml_out", "pml_out"]
        inputs = deepcopy(inputs1)
        updateInputs!(inputs, :bc, ["pml_out", "pml_out"])
    else
        inputs = deepcopy(inputs1)
    end

    ## DEFINITIONS BLOCK ##
    D₀ = inputs.D₀
    N_ext = inputs.N_ext
    dx = inputs.dx
    x_ext = inputs.x_ext
    x_inds = inputs.x_inds
    ∂_ext = inputs.∂_ext
    ɛ_ext = inputs.ɛ_sm
    F_ext = inputs.F_sm
    Γ_ext = inputs.Γ_ext
    Γ = zeros(N_ext,1)
    for dd in 3:length(∂_ext)-2
        δ,dummy1 = dirac_δ(x_ext,∂_ext[dd])
        Γ = Γ[:]  + full(δ)/Γ_ext[dd]
    end
    ## END OF DEFINITIONS BLOCK ##

    if isempty(size(F))
        F = [F]
    end
    if length(F)==1
        r1 = 1
    else
        r1 = inputs.x_inds
    end

    ∇² = laplacian(k, inputs)

    ɛΓ⁻¹ = sparse( 1:N_ext, 1:N_ext, 1./(ɛ_ext+Γ-1im*D₀*F.*F_ext), N_ext, N_ext, +)

    if isempty(ψ_init)
        (k²,ψ_ext,nconv,niter,nmult,resid) = eigs(-ɛΓ⁻¹*∇²,which = :LM, nev = np, sigma = k^2+1.e-5)
    else
        (k²,ψ_ext,nconv,niter,nmult,resid) = eigs(-ɛΓ⁻¹*∇²,which = :LM, nev = np, sigma = k^2+1.e-5, v0 = ψ_init)
    end

    ψ = zeros(Complex128,length(inputs.x_ext),np)

    r = inputs.x_inds
    for ii = 1:np
        N = trapz( (ɛ_ext[r] + Γ[r] - 1im*D₀.*F_ext[r].*F[r1]) .*abs2.(ψ_ext[r,ii]),dx)
        ψ[:,ii] = ψ_ext[:,ii]/sqrt(N)
    end

    if truncate
        return sqrt.(k²), ψ[inputs.x_inds,:]
    else
        return sqrt.(k²), ψ
    end

end
# end of function computePole_L




"""
k,ψ =  computeZero_L(inputs, k, nz; F=1, truncate = false, ψ_init = [])

Compute zeros w/o line pulling, using purely incoming PML's.

Set D or F to zero to get passive cavity.

k[ # of zeros]

ψ[ cavity size, # of zeros]

"""
function computeZero_L(inputs::InputStruct, k::Complex128, nz::Int; F=1., truncate::Bool = false, ψ_init = [])

    inputs1 = deepcopy(inputs)

    conj!(inputs1.ɛ_sm)
    conj!(inputs1.ɛ_ext)
    conj!(inputs1.ɛ)

    conj!(inputs1.Γ)
    conj!(inputs1.Γ_ext)

    inputs1.D₀ = -inputs.D₀

    kz,ψz = computePole_L(inputs1, conj(k), nz; F=F, truncate = truncate, ψ_init = conj(ψ_init) )

    return conj!(kz),conj!(ψz)

end
# end of function computeZero_L





"""
k,ψ =  computeUZR_L(inputs, k, nu; F=1., direction = "R", truncate = false, ψ_init = [])

Compute UZR (unidirectional zero reflection) states w/o line pulling. Sets bc's to incident PML on left and outgoing on right if DIRECTION = "R", and vice versa if "L"

k[ # of UZRs]

ψ[ cavity size, # of UZRs]

"""
function computeUZR_L(inputs1::InputStruct, k::Complex128, nu::Int; F=1., direction::String = "R", truncate::Bool = false, ψ_init = [])

    # set outgoing boundary conditions, using PML implementation
    inputs = deepcopy(inputs1)
    if direction in ["R" "r" "right" "Right" "->" ">" "=>"]
        updateInputs!(inputs,:bc, ["pml_in", "pml_out"])
    elseif direction in ["L" "l" "left" "Left" "<-" "<" "<="]
        updateInputs!(inputs,:bc, ["pml_out", "pml_in"])
    else
        println("error. invalid direction")
        return
    end

    ## DEFINITIONS BLOCK ##
    D₀ = inputs.D₀
    N_ext = inputs.N_ext
    dx = inputs.dx
    x_ext = inputs.x_ext
    x_inds = inputs.x_inds
    ∂_ext = inputs.∂_ext
    ɛ_ext = inputs.ɛ_sm
    F_ext = inputs.F_sm
    Γ_ext = inputs.Γ_ext
    ## END OF DEFINITIONS BLOCK ##

    if isempty(size(F))
        F = [F]
    end
    if length(F)==1
        r1 = 1
    else
        r1 = inputs.x_inds
    end

    Γ = zeros(N_ext,1)
    for dd in 3:length(∂_ext)-2
        δ,dummy1 = dirac_δ(x_ext,∂_ext[dd])
        Γ = Γ[:]  + full(δ)/Γ_ext[dd]
    end

    ∇² = laplacian(k, inputs)

    ɛΓ⁻¹ = sparse( 1:N_ext, 1:N_ext, 1./(ɛ_ext+Γ-1im*D₀*F.*F_ext), N_ext, N_ext, +)

    if isempty(ψ_init)
        (k²,ψ_ext,nconv,niter,nmult,resid) = eigs(-ɛΓ⁻¹*∇²,which = :LM, nev = nu, sigma = k^2+1.e-6)
    else
        (k²,ψ_ext,nconv,niter,nmult,resid) = eigs(-ɛΓ⁻¹*∇²,which = :LM, nev = nu, sigma = k^2+1.e-6, v0 = ψ_init)
    end

    ψ = zeros(Complex128,length(inputs.x_ext),nu)

    r = inputs.x_inds
    for ii = 1:nu
        N = trapz( (ɛ_ext[r] + Γ[r] - 1im*D₀.*F_ext[r].*F[r1]) .*abs2.(ψ_ext[r,ii]),dx)
        ψ[:,ii] = ψ_ext[:,ii]/sqrt(N)
    end

    if truncate
        return sqrt.(k²),ψ[inputs.x_inds,:]
    else
        return sqrt.(k²),ψ
    end

end
# end of function computeUZR_L




"""
k,ψ =  computeK_NL1(inputs, k_init; F=1., dispOpt = false, η_init = 1e-13+0.0im, u_init = [], k_avoid = [.0], tol = .5, max_count = 15, max_iter = 50)

Compute K's at S-matrix evec specified in input file, with line-pulling, using CF method.

k::Complex128

ψ[cavity size]

"""
function computeK_NL1(inputs::InputStruct, k_init::Complex128; F=1., dispOpt::Bool = false, η_init::Complex128 = 1e-13+0.0im, u_init = [], k_avoid::Array{Complex128,1} = [complex(.0)], tol::Float64 = .5, max_count::Int = 15, max_iter::Int = 50)
    # With Line Pulling. Nonlinear solve using TCF evecs and evals
    # max_count: the max number of TCF states to compute
    # max_iter: maximum number of iterations to include in nonlinear solve
    # η: should be a number, not an array (even a singleton array)
    # Uses bc's specified in input file.


    ## definitions block
    D₀ = inputs.D₀
    dx = inputs.dx
    ##

    if isempty(size(F))
        F = [F]
    end

    u = NaN*ones(Complex128,inputs.N_ext,1)
    η_init = [η_init]
    η_init[:,1],u[:,1] = computeCF(inputs, k_init, 1; η_init=η_init[1], u_init = u_init, F=F)

    function f!(z::Array{Float64,1}, fvec::Array{Float64,1})

        k = z[1]+1im*z[2]

        flag::Bool = true
        count::Int = 1
        M::Int = 1
        ind = Int
        println("while")
        @time while flag

            η_temp,u_temp = computeCF(inputs, k, M; η_init=η_init[1], u_init = u[:,1], F=F)
            overlap = zeros(Float64,M)

            for i in 1:M
                 overlap[i] = abs(trapz(u[:,1].*inputs.F_sm.*F.*u_temp[:,i],dx))
            end

            ind = findmax(overlap)[2]

              if (abs(overlap[ind]) > (1-tol))
                flag = false
                η_init[:,1] = η_temp[ind]
                u[:,1] = u_temp[:,ind]
            elseif  (count < max_count) & (abs(overlap[ind]) ≤ (1-tol))
                M += 1
            else
                flag = false
                η_init[:,1] = η_temp[ind]
                u[:,1] = u_temp[:,ind]
                println("Warning: overlap less than tolerance, skipped to neighboring TCF.")
            end

            count += 1

        end

        η = η_init[1]

        fvec[1] = real((η-D₀*γ(inputs,k))/prod(k-k_avoid))
        fvec[2] = imag((η-D₀*γ(inputs,k))/prod(k-k_avoid))

    end

    @time    z = nlsolve(f!,[real(k_init),imag(k_init)]; iterations = max_iter, show_trace = dispOpt)
    k = z.zero[1]+1im*z.zero[2]
    conv = converged(z)

    η_init[1] = inputs.D₀*γ(inputs,k)
    η,u[:,1] = computeCF(inputs, k, 1; η_init=η_init[1], F=F)

    return k,u[:,1],η[1],conv

end
# end of function computeK_NL1




"""
k,ψ =  computePole_NL1(inputs, k_init; F=1., dispOpt = false, η_init = 1e-13+0.0im,
    u_init = [], k_avoid = [.0], tol = .5, max_count = 15, max_iter = 50)

    Compute pole, with line-pulling, using CF method.

    k::Complex128

    ψ[cavity size]
"""
function computePole_NL1(inputs1::InputStruct, k_init::Complex128; F=1., dispOpt::Bool = false, η_init::Complex128 = 1e-13+0.0im, u_init = [], k_avoid::Array{Complex128,1} = [complex(.0)], tol::Float64 = .5, max_count::Int = 15, max_iter::Int = 50)
    # With Line Pulling. Nonlinear solve using TCF evecs and evals
    # max_count: the max number of TCF states to compute
    # max_iter: maximum number of iterations to include in nonlinear solve
    # η: should be a number, not an array (even a singleton array)
    # uses outgoing PML implementation

    # set outgoing boundary conditions, using PML implementation
    inputs = deepcopy(inputs1)
    updateInputs!(inputs,:bc, ["out", "out"])

    k,u,η,conv = computeK_NL1(inputs, k_init; F=F, dispOpt = dispOpt, η_init = η_init, u_init = u_init, k_avoid = k_avoid, tol = tol, max_count = max_count, max_iter = max_iter)

    return k,u,η,conv

end
# end of function computePole_NL1




"""
k,ψ =  computeZero_NL1(inputs, k_init; F=1., dispOpt = false, η_init = 1e-13+0.0im, u_init = [], k_avoid = [.0], tol = .5, max_count = 15, max_iter = 50)

Compute zero, with line-pulling, using CF method.

k::Complex128

ψ[cavity size]

F should be an array either of length N_ext or of length 1.

"""
function computeZero_NL1(inputs1::InputStruct, k_init::Complex128; F=1., dispOpt::Bool = false, η_init::Complex128 = 1e-13+0.0im, u_init = [], k_avoid::Array{Complex128,1} = [complex(.0)], tol::Float64 = .5, max_count::Int = 15, max_iter::Int = 50)
    # With Line Pulling. Nonlinear solve using TCF evecs and evals
    # max_count: the max number of TCF states to compute
    # max_iter: maximum number of iterations to include in nonlinear solve
    # η: should be a number, not an array (even a singleton array)
    # uses outgoing PML implementation

    # set outgoing boundary conditions, using PML implementation
    inputs = deepcopy(inputs1)
    updateInputs!(inputs,:bc, ["in", "in"])

    k,u,η,conv = computeK_NL1(inputs, k_init; F=F, dispOpt = dispOpt, η_init = η_init, u_init = u_init, k_avoid = k_avoid, tol = tol, max_count = max_count, max_iter = max_iter)

    return k,u,η,conv

end
# end of function computeZero_NL1





"""
k,ψ =  computeUZR_NL1(inputs, k_init; F=1., dispOpt = false, η_init = 1e-13+0.0im, u_init = [], k_avoid = [.0], tol = .5, max_count = 15, max_iter = 50)

Compute unidirectional zero, with line-pulling, using CF method.

k::Complex128

ψ[cavity size]

"""
function computeUZR_NL1(inputs1::InputStruct, k_init::Complex128; direction::String = "R", F=1., dispOpt::Bool = false, η_init::Complex128 = 1e-13+0.0im, u_init = [], k_avoid::Array{Complex128,1} = [complex(.0)], tol::Float64 = .5, max_count::Int = 15, max_iter::Int = 50)
    # With Line Pulling. Nonlinear solve using TCF evecs and evals
    # max_count: the max number of TCF states to compute
    # max_iter: maximum number of iterations to include in nonlinear solve
    # η: should be a number, not an array (even a singleton array)
    # uses outgoing PML implementation


    # set outgoing boundary conditions, using PML implementation
    inputs = deepcopy(inputs1)
    if direction in ["R" "r" "right" "Right" "->" ">" "=>"]
        updateInputs!(inputs,:bc, ["in", "out"])
    elseif direction in ["L" "l" "left" "Left" "<-" "<" "<="]
        updateInputs!(inputs,:bc, ["out", "in"])
    else
        println("error. invalid direction")
        return
    end

    k,u,η,conv = computeK_NL1(inputs, k_init; F=F, dispOpt = dispOpt, η_init = η_init, u_init = u_init, k_avoid = k_avoid, tol = tol, max_count = max_count, max_iter = max_iter)

    return k,u,η,conv

end
# end of function computeUZR_NL1





"""
k =  computeK_NL2(inputs, k, Radii; Nq=100, nks=3, F=[1.], R_min = .01, rank_tol = 1e-8) = .5, max_count = 15, max_iter = 50)

Compute K's at S-matrix evec specified in input file, with line-pulling, using contour method.

Inputs: k is centroid of contour
Radii = (real radius, imag radius) are the semi-diameters of the contour
Nq are the number of quadrature points.
nKs is an upper-limit on the number of frequencies expected inside the contour.
R_min is the radius of subtracted contour if the atomic point is contained inside the contour

"""
function computeK_NL2(inputs::InputStruct, k::Complex128, Radii::Tuple{Real,Real}; Nq::Int =100, nk::Int=3, F=1., R_min::Float64 = .01, rank_tol::Float64 = 1e-8)
    # With Line Pulling, using contour integration


    ## definitions block
    nevals = nk
    D₀ = inputs.D₀
    N_ext = inputs.N_ext
    dx = inputs.dx
    x_ext = inputs.x_ext
    x_inds = inputs.x_inds
    ∂_ext = inputs.∂_ext
    ɛ_ext = inputs.ɛ_sm
    F_ext = inputs.F_sm
    Γ_ext = inputs.Γ_ext
    Γ = zeros(N_ext,1)
    for dd in 3:length(∂_ext)-2
        δ,dummy1 = dirac_δ(x_ext,∂_ext[dd])
        Γ = Γ[:] + full(δ)/Γ_ext[dd]
    end
    ## end of definitions block

    if isempty(size(F))
        F = [F]
    end

    ɛ = sparse(1:N_ext, 1:N_ext, ɛ_ext, N_ext, N_ext, +)
    Γ = sparse(1:N_ext, 1:N_ext, Γ    , N_ext, N_ext, +)

    A  = zeros(Complex128,N_ext,nevals)
    A₀ = zeros(Complex128,N_ext,nevals)
    A₁ = zeros(Complex128,N_ext,nevals)

    rad(a,b,θ) = b/sqrt(sin(θ)^2+(b/a)^2*cos(θ)^2)
    θ = angle(inputs.k₀-1im*inputs.γ⟂-k)
    flag::Bool = abs(inputs.k₀-1im*inputs.γ⟂-k) < rad.(Radii[1],Radii[2],θ)

    M = rand(N_ext,nevals)
    ϕ = 2π*(0:1/Nq:(1-1/Nq))
    Ω = k + Radii[1]*cos.(ϕ) + 1im*Radii[2]*sin.(ϕ)

    if flag
        AA  = zeros(Complex128,N_ext,nevals)
        AA₀ = zeros(Complex128,N_ext,nevals)
        AA₁ = zeros(Complex128,N_ext,nevals)
        RR = 2*R_min
        ΩΩ = inputs.k₀-1im*inputs.γ⟂ + (RR/2)*cos.(ϕ) + 1im*(RR/2)*sin.(ϕ)
    end

    for i in 1:Nq

        k′ = Ω[i]
        k′² = k′^2

        if (i > 1) & (i < Nq)
            dk′ = (Ω[i+1]-Ω[i-1]  )/2
        elseif i == Nq
            dk′ = (Ω[1]  -Ω[end-1])/2
        elseif i == 1
            dk′ = (Ω[2]  -Ω[end]  )/2
        end

        ∇² = laplacian(k′, inputs)
        χk′² = sparse(1:N_ext, 1:N_ext, D₀*γ(inputs,k′)*F.*F_ext*k′², N_ext, N_ext, +)

        A = (∇²+(ɛ+Γ)*k′²+χk′²)\M
        A₀ += A*dk′/(2π*1im)
        A₁ += A*k′*dk′/(2π*1im)

        if flag
            kk′ = ΩΩ[i]
            kk′² = kk′^2
            if (i > 1) & (i < Nq)
                dkk′ = (ΩΩ[i+1]-ΩΩ[i-1]  )/2
            elseif i == Nq
                dkk′ = (ΩΩ[1]  -ΩΩ[end-1])/2
            elseif i == 1
                dkk′ = (ΩΩ[2]  -ΩΩ[end]  )/2
            end
            χkk′² = sparse(1:N_ext, 1:N_ext, D₀*γ(inputs,kk′)*F.*F_ext*kk′², N_ext, N_ext, +)

            AA = (∇²+(ɛ+Γ)*kk′²+χkk′²)\M
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
        println("error. need more nevals")
        return NaN
    else
        k = temp[1]-1
    end

    B = (P[:U][:,1:k])'*A₁*(P[:Vt][1:k,:])'*diagm(1./P[:S][1:k])

    D,V = eig(B)

    return D

end
# end of function computeK_NL2




"""
k =  computePole_NL2(inputs, k, Radii; Nq=100, nps=3, F=[1.], R_min = .01, rank_tol = 1e-8) = .5, max_count = 15, max_iter = 50)

Compute poles, with line-pulling, using contour method.

Inputs: k is centroid of contour
Radii = (real radius, imag radius) are the semi-diameters of the contour
Nq are the number of quadrature points.
nps is an upper-limit on the number of frequencies expected inside the contour.
R_min is the radius of subtracted contour if the atomic point is contained inside the contour

"""
function computePole_NL2(inputs1::InputStruct, k::Complex128, Radii::Tuple{Real,Real}; Nq::Int =100, np::Int=3, F=1., R_min::Float64 = .01, rank_tol::Float64 = 1e-8)
    # With Line Pulling, using contour integration

    # set outgoing boundary conditions, using exact implementation
    inputs = deepcopy(inputs1)
    updateInputs!(inputs,:bc,["out", "out"])

    k = computeK_NL2(inputs, k, Radii; Nq=Nq, nk=np, F=F, R_min = R_min, rank_tol = rank_tol)

    return k

end
# end of function computePole_NL2



"""
k =  computeZero_NL2(inputs, k, Radii; Nq=100, nz=3, F=[1.], R_min = .01, rank_tol = 1e-8) = .5, max_count = 15, max_iter = 50)

Compute zeros, with line-pulling, using contour method.

Inputs: k is centroid of contour
Radii = (real radius, imag radius) are the semi-diameters of the contour
Nq are the number of quadrature points.
nzs is an upper-limit on the number of frequencies expected inside the contour.
R_min is the radius of subtracted contour if the atomic point is contained inside the contour

"""
function computeZero_NL2(inputs1::InputStruct, k::Complex128, Radii::Tuple{Real,Real}; Nq::Int =100, nz::Int=3, F=1., R_min::Float64 = .01, rank_tol::Float64 = 1e-8)
    # With Line Pulling, using contour integration

    # set outgoing boundary conditions, using exact implementation
    inputs = deepcopy(inputs1)
    updateInputs!(inputs,:bc,["in", "in"])

    k = computeK_NL2(inputs, k, Radii; Nq=Nq, nk=nz, F=F, R_min = R_min, rank_tol = rank_tol)

    return k

end
# end of function computeZero_NL2




"""
k =  computeUZR_NL2(inputs, k, Radii::Tuple{Real,Real}; direction = "R" Nq=100, nus=3, F=[1.], R_min = .01, rank_tol = 1e-8) = .5, max_count = 15, max_iter = 50)

Compute UZR, with line-pulling, using contour method.

Inputs: k is centroid of contour
Radii = (real radius, imag radius) are the semi-diameters of the contour
Nq are the number of quadrature points.
nus is an upper-limit on the number of frequencies expected inside the contour.
R_min is the radius of subtracted contour if the atomic point is contained inside the contour

"""
function computeUZR_NL2(inputs1::InputStruct, k::Complex128, Radii::Tuple{Real,Real}; direction::String = "R", Nq::Int =100, nu::Int=3, F=1., R_min::Float64 = .01, rank_tol::Float64 = 1e-8)
    # With Line Pulling, using contour integration

    # set outgoing boundary conditions, using PML implementation
    inputs = deepcopy(inputs1)
    if direction in ["R" "r" "right" "Right" "->" ">" "=>"]
        updateInputs!(inputs,:bc, ["in", "out"])
    elseif direction in ["L" "l" "left" "Left" "<-" "<" "<="]
        updateInputs!(inputs,:bc, ["out", "in"])
    else
        println("error. invalid direction")
        return
    end

    k = computeK_NL2(inputs, k, Radii; Nq=Nq, nk=nu, F=F, R_min = R_min, rank_tol = rank_tol)

    return k

end
# end of function computeUZR_NL2




"""
ψ, A =  computePsi(inputs::Dict, k::Number; isNonLinear=false, ψ_init=0, F=1., dispOpt = false, truncate = false, fileName = "", ftol=1e-6, iter=150)

Solves using whatever boundary conditions are specified. Not necessarily scattering in the usual sense.

Outputs: ψ is solution, A is factorized linear wave operator
"""
function computePsi(inputs::InputStruct, k::Complex128; isNonLinear::Bool=false, ψ_init=0, F=1., dispOpt::Bool = false, truncate::Bool = false, fileName::String = "", ftol::Float64 = 1e-6, iter::Int = 150, A = [])::Tuple{ Array{ Complex{Float64},1 } , Base.SparseArrays.UMFPACK.UmfpackLU{Complex{Float64},Int64} }

    ## definitions block
    x_ext = inputs.x_ext
    ∂_ext = inputs.∂_ext
    Γ_ext = inputs.Γ_ext
    N_ext = inputs.N_ext
    x_inds = inputs.x_inds
    D₀ = inputs.D₀
    k₀ = inputs.k₀
    a = inputs.a
    ɛ_ext = inputs.ɛ_sm
    F_ext = inputs.F_sm
    ## end of definitions block

    if isempty(size(F))
        F = [F]
    end

    if A==[] || isNonLinear

        k²= k^2

        ∇² = laplacian(k, inputs)

        Γ = zeros(N_ext,1)
        for dd in 3:length(∂_ext)-2
            δ,dummy1 = dirac_δ(x_ext,∂_ext[dd])
            Γ = Γ[:] + full(δ)/Γ_ext[dd]
        end

        ɛk² = sparse(1:N_ext,1:N_ext, k²*ɛ_ext              ,N_ext, N_ext, +)
        χk² = sparse(1:N_ext,1:N_ext, k²*D₀*γ(inputs,k)*(F.*F_ext) ,N_ext, N_ext, +)
        Γk² = sparse(1:N_ext,1:N_ext, k²*Γ                  ,N_ext, N_ext, +)

        A = factorize(∇²+ɛk²+χk²+Γk²)

    end

    δᵨ₊,∇δᵨ₊ = dirac_δ(x_ext,inputs.xᵨ₊)
    Θᵨ₊,∇Θᵨ₊,∇²Θᵨ₊ = heaviside_Θ(x_ext,inputs.xᵨ₊)
    x₀ = inputs.x[1]
    j₊ = a[1]*(+∇δᵨ₊.*exp.(+1im*k*(x_ext-x₀)) + 2*1im*k*δᵨ₊.*exp.(+1im*k*(x_ext-x₀)))

    δᵨ₋,∇δᵨ₋ = dirac_δ(x_ext,inputs.xᵨ₋)
    Θᵨ₋,∇Θᵨ₋,∇²Θᵨ₋ = heaviside_Θ(x_ext,inputs.xᵨ₋)
    Θᵨ₋ = 1-Θᵨ₋
    x₀ = inputs.x[end]
    j₋ = a[2]*(-∇δᵨ₋.*exp.(-1im*k*(x_ext-x₀)) + 2*1im*k*δᵨ₋.*exp.(-1im*k*(x_ext-x₀)))

    j = j₊ + j₋

    if ψ_init == 0 || !isNonLinear
        ψ_ext = A\full(j)
    else
        ψ_ext = ψ_init
    end

    if isNonLinear

        K::Float64 = real(k)
        K² = K^2

        function f!(Ψ::AbstractVector,fvec::AbstractVector)
            ψ = zeros(Complex128,N_ext)
            ψ = Ψ[1:N_ext]+1im*Ψ[N_ext+(1:N_ext)]
            temp = (∇².+ɛk².+Γk².+χ(inputs,k,F,ψ)*k²)*ψ .- j
            fvec[1:N_ext] = real(temp)
            fvec[N_ext+(1:N_ext)] = imag(temp)
        end

        function jac!(Ψ::AbstractVector, jacarr::AbstractMatrix)
            ψ = zeros(Complex128,N_ext)
            ψ = Ψ[1:N_ext]+1im*Ψ[N_ext+(1:N_ext)]
            temp = ∇².+ɛk².+Γk².+χ(inputs,k,F,ψ)*K²
            tempr = similar(temp,Float64)
            tempi = similar(temp,Float64)
            tr = nonzeros(tempr)
            ti = nonzeros(tempi)
            tr[:] = real.((nonzeros(temp)))
            ti[:] = imag.((nonzeros(temp)))
            tempj = [tempr+χʳʳ′(inputs,k,F,ψ)*K² -tempi+χʳⁱ′(inputs,k,F,ψ)*K²; tempi+χⁱʳ′(inputs,k,F,ψ)*K² tempr+χⁱⁱ′(inputs,k,F,ψ)*K²]
            jacarr[:,:] = tempj[:,:]
        end

        Ψ_init = Array{Float64}(2*N_ext)
        Ψ_init[1:N_ext] = real(ψ_ext)
        Ψ_init[N_ext+(1:N_ext)] = imag(ψ_ext)

        df = DifferentiableSparseMultivariateFunction(f!, jac!)
        z = nlsolve(df, Ψ_init; show_trace = dispOpt , ftol=ftol, iterations=iter)

        if converged(z)
            ψ_ext = z.zero[1:N_ext] + 1im*z.zero[N_ext+(1:N_ext)]
        else
            ψ_ext = NaN*ψ_ext;
            println("Warning, computePsi did not converge. Returning NaN.")
        end

    end

    if !isempty(fileName)
        if truncate
            foo(fid) = serialize(fid,(inputs,ψ_ext[inputs.x_inds]))
            open(foo,fileName,"w")
        else
            foo1(fid) = serialize(fid,(inputs,ψ_ext))
            open(foo1,fileName,"w")
        end
    end

    if truncate
        return ψ_ext[inputs.x_inds], A
    else
        return ψ_ext, A
    end

end
# end of function solve_scattered



"""
G = computeG(inputs, k, x₀; F=1., fileName = "", truncate = false)

computes Greens fn G(k;x₀,x), using whatever boundary condition is specified in the input file.
"""
function computeG(inputs::InputStruct, k::Complex128, x₀::Float64; F=1., fileName::String = "", truncate::Bool = false)::Tuple{ Array{ Complex{Float64},1 } , Base.SparseArrays.UMFPACK.UmfpackLU{Complex{Float64},Int64} }

    ## definitions block
    N_ext = inputs.N_ext
    x_ext = inputs.x_ext
    ∂_ext = inputs.∂_ext
    Γ_ext = inputs.Γ_ext
    D₀ = inputs.D₀
    ɛ_ext = inputs.ɛ_sm
    F_ext = inputs.F_sm
    ## end of definitions block

    if isempty(size(F))
        F = [F]
    end

    k²= k^2

    ∇² = laplacian(k, inputs)

    Γ = zeros(N_ext,1)
    for dd in 3:length(∂_ext)-2
        δ,dummy1 = dirac_δ(x_ext,∂_ext[dd])
        Γ = Γ[:] + full(δ)/Γ_ext[dd]
    end

    ɛk² = sparse(1:N_ext,1:N_ext, k²*ɛ_ext                     ,N_ext, N_ext, +)
    χk² = sparse(1:N_ext,1:N_ext, k²*D₀*γ(inputs,k)*(F.*F_ext) ,N_ext, N_ext, +)
    Γk² = sparse(1:N_ext,1:N_ext, k²*Γ                         ,N_ext, N_ext, +)

    δ₀,∇δ₀ = dirac_δ(x_ext,x₀)

    A = factorize(∇²+ɛk²+χk²+Γk²)

    G_ext = A\full(δ₀)

    if !isempty(fileName)
        if truncate
            foo(fid) = serialize(fid,(inputs,G_ext[inputs.x_inds]))
            open(foo,fileName,"w")
        else
            foo1(fid) = serialize(fid,(inputs,G_ext))
            open(foo1,fileName,"w")
        end
    end

    if truncate
        return G_ext[inputs.x_inds], A
    else
        return G_ext, A
    end

end
# end of function ComputeG




"""
S =  computeS(inputs::Dict; N=10, N_Type="D", isNonLinear=false, F=1., dispOpt = true, ψ_init = [], fileName = "")
"""
function computeS(inputs1::InputStruct; N::Int=10, N_Type::String="D", isNonLinear::Bool=false, F=1., dispOpt::Bool = true, ψ_init = [], fileName::String = "")
    # N is the number of steps to go from D0 = 0 to given D0

    inputs = deepcopy(inputs1)
    updateInputs!(inputs,:bc, ["out", "out"])

    x_inds = inputs.x_inds

    ψ₊ = Array{Complex128}(length(inputs.x_ext))
    ψ₋ = Array{Complex128}(length(inputs.x_ext))
    ψ  = Array{Complex128}(length(inputs.x_ext))

    if isempty(ψ_init) & (N>1)
        S = NaN*ones(Complex128,2,2,length(inputs.k),N)
    else
        S = NaN*ones(Complex128,2,2,length(inputs.k),1)
    end

    D₀ = deepcopy(inputs.D₀)
    A = deepcopy(inputs.a)

    if N > 1
        D = linspace(0,D₀,N)
        A_vec = linspace(0.001,1,N)
    else
        D = [inputs.D₀]
        A_vec = [1]
    end

    for ii in 1:length(inputs.k)

        k = inputs.k[ii]

        if (ii/1 == round(ii/1)) & dispOpt
            if typeof(k)<:Real
                printfmtln("Solving for frequency {1:d} of {2:d}, ω = {3:2.3f}.",ii,length(inputs.k),k)
            else
                printfmtln("Solving for frequency {1:d} of {2:d}, ω = {3:2.3f}{4:+2.3f}i.",ii,length(inputs.k),real(k),imag(k))
            end
        end

        if isempty(ψ_init)

            for j in 1:N

                if N_Type == "D"
                    inputs.D₀ = D[j]
                elseif N_Type == "A"
                    inputs.a = A_vec[j]*A
                end

                if isNonLinear
                    if isnan(ψ[1]) | j==1
                        ψ, W = computePsi(inputs,k,isNonLinear = true, F = F)
                    else
                        ψ, W = computePsi(inputs,k,isNonLinear = true, F = F, ψ_init = ψ)
                    end
                else
                    ψ = 0.
                end

                inputs.a = [1.0,0.0]
                ψ₊, W = computePsi(inputs,k; isNonLinear = false, F = F./(1+abs2.(γ(inputs,k)*ψ)))

                inputs.a = [0.0,1.0]
                ψ₋, W = computePsi(inputs,k; isNonLinear = false, F = F./(1+abs2.(γ(inputs,k)*ψ)), A = W)

                S[1,1,ii,j] = ψ₊[x_inds[1]]*exp(+1im*inputs.dx*k)
                S[2,1,ii,j] = ψ₊[x_inds[end]]
                S[1,2,ii,j] = ψ₋[x_inds[1]]
                S[2,2,ii,j] = ψ₋[x_inds[end]]*exp(-1im*inputs.dx*k)

                inputs.D₀ = D₀
                inputs.a  = A

                if !isempty(fileName)
                    foo1(fid) = serialize(fid,(inputs,D,S,ii,j))
                    open(foo1,fileName,"w")
                end

            end

        else
            if isNonLinear
                ψ = computePsi(inputs,k,isNonLinear = true, F = F, ψ_init = ψ_init)
            else
                ψ = 0.
            end

            inputs.a = [1.0,0.0]
            ψ₊, W = computePsi(inputs,k,isNonLinear = false, F = F./(1+abs2.(γ(inputs,k)*ψ)))

            inputs.a = [0.0,1.0]
            ψ₋, W = computePsi(inputs,k,isNonLinear = false, F = F./(1+abs2.(γ(inputs,k)*ψ)), A = W)

            S[1,1,ii,1] = ψ₊[x_inds[1]]*exp(+1im*inputs.dx*k)
            S[2,1,ii,1] = ψ₊[x_inds[end]]
            S[1,2,ii,1] = ψ₋[x_inds[1]]
            S[2,2,ii,1] = ψ₋[x_inds[end]]*exp(-1im*inputs.dx*k)

            inputs.D₀ = D₀
            inputs.a = A

            if !isempty(fileName)
                foo2(fid) = serialize(fid,(inputs,D,S,ii,1))
                open(foo2,fileName,"w")
            end

        end

    end

    return S

end
# end of fuction computeS



#### NOT CHECKED FROM HERE ON ####

function solve_SPA(inputs::Dict, ω; z₀₊=0.001im, z₀₋=0.001im, F = 1., u=[], η=[], φ₊=[], φ₋=[])

    x = inputs["x_ext"]
    dx = inputs["dx"]
    ∂_ext = inputs["∂_ext"]
    N_ext = inputs["N_ext"]
    D₀ = inputs["D₀"]

    function γ()
        return inputs["γ⟂"]/(ω-inputs["ω₀"]+1im*inputs["γ⟂"])
    end

    a = deepcopy(inputs["a"])

    if isempty(φ₊) | isempty(φ₋)
        inputs["a"] = [1.0,0.0]
        φ₊ = solve_scattered(inputs,ω; isNonLinear = false, F = 0.)
        inputs["a"] = [0.0,1.0]
        φ₋ = solve_scattered(inputs,ω; isNonLinear = false, F = 0.)
    end
    inputs["a"] = deepcopy(a)

    if isempty(η) | isempty(u)
        η,u = computeCFs(inputs,ω,1)
    end

    r = whichRegion(x,∂_ext)
    F_ext = inputs["F_ext"][r]

    function f₊!(z, fvec)

        b = inputs["a"][1]
        Z = z[1]+1im*z[2]

        numerator = u.*(F.*F_ext).*(Z*u+b*φ₊)
        denominator = 1+abs2.(γ())*abs2.(Z*u+b*φ₊)
        term = (D₀*γ()./η).*trapz(numerator./denominator,dx)

        fvec[1] = real(term[1]/Z - 1)
        fvec[2] = imag(term[1]/Z - 1)

        return term,Z

    end

    function f₋!(z,fvec)

        b = inputs["a"][2]
        Z = z[1]+1im*z[2]

        numerator = u.*(F.*F_ext).*(Z*u+b*φ₋)
        denominator = 1+abs2.(γ())*abs2.(Z*u+b*φ₋)
        term = (D₀*γ()./η).*trapz(numerator./denominator,dx)

        fvec[1] = real(term[1]/Z - 1)
        fvec[2] = imag(term[1]/Z - 1)

        return term,Z

    end

    result₊ = nlsolve(f₊!,[real(z₀₊),imag(z₀₊)])
    result₋ = nlsolve(f₋!,[real(z₀₋),imag(z₀₋)])

    z₊ = result₊.zero[1]+1im*result₊.zero[2]
    z₋ = result₋.zero[1]+1im*result₋.zero[2]

    ψ₊ = inputs["a"][1]*φ₊+z₊*u
    ψ₋ = inputs["a"][2]*φ₋+z₋*u

    return ψ₊,ψ₋,(z₊,z₋,φ₊,φ₋,η,u)

end
# end of function solve_SPA



function solve_single_mode_lasing(inputs::Dict, D₀::Float64; ψ_init=0.000001, ω_init=inputs["ω₀"], inds=14,dispOpt=false)

    dx = inputs["dx"]
    x_ext = inputs["x_ext"]
    ∂_ext = inputs["∂_ext"]
    ℓ_ext = inputs["ℓ_ext"]
    N_ext = inputs["N_ext"]
    λ = inputs["λ"]
    x_inds = inputs["x_inds"]
    F_ext = inputs["F_ext"]
    Γ_ext = inputs["Γ_ext"]
    ɛ_ext = inputs["ɛ_ext"]
    ω₀ = inputs["ω₀"]
    γ⟂ = inputs["γ⟂"]

    inds = inputs["x_inds"][inds]

    ɛ_ext, F_ext = subpixelSmoothing(inputs)
    ∇² = laplacian(inputs["k₀"], inputs)

    Γ = zeros(N_ext,1)
    for dd in 2:length(∂_ext)-1
        δ,dummy1 = dirac_δ(x_ext,∂_ext[dd])
        Γ = Γ[:] + full(δ)/Γ_ext[dd]
    end

    ɛ = sparse(1:N_ext,1:N_ext,ɛ_ext[:],N_ext,N_ext,+)
    Γ = sparse(1:N_ext,1:N_ext,Γ[:]    ,N_ext,N_ext,+)

    function γ(ω)
        return γ⟂/(ω-ω₀+1im*γ⟂)
    end

    function γ′(ω)
        return -γ⟂/(ω-ω₀+1im*γ⟂)^2
    end

    function Γ′(ω)
        return -2*(γ⟂^2)*(ω-ω₀)/((ω-ω₀)^2 + γ⟂^2)^2
    end

    function ψ_norm(Ψ,inds)
        return sum(abs2.(Ψ[inds]))
    end

    function χ(Ψ,ω)
        V = F_ext[:].*γ(ω)*D₀./(1+abs2.(γ(ω)*Ψ))
        return sparse(1:N_ext, 1:N_ext, V, N_ext, N_ext,+)
    end

    function χʷ′(Ψ,ω)
        V = F_ext[:].*γ′(ω).*D₀./(1+abs2.(γ(ω).*Ψ)) - Γ′(ω).*abs2.(Ψ).*F_ext[:].*γ(ω).*D₀./((1+abs2.(γ(ω).*Ψ)).^2)
        return sparse(1:N_ext, 1:N_ext, V, N_ext, N_ext, +)
    end

    function χʳʳ′(Ψ,ω)
        V = -2.*(abs2.(γ(ω))*F_ext[:].*real(γ(ω).*Ψ).*D₀.*real(Ψ)./((1+abs2.(γ(ω)*Ψ)).^2))/ψ_norm(Ψ,inds)
        return sparse(1:N_ext, 1:N_ext, V, N_ext, N_ext, +)
    end

    function χⁱʳ′(Ψ,ω)
        V = -2.*(abs2.(γ(ω))*F_ext[:].*imag(γ(ω).*Ψ).*D₀.*real(Ψ)./((1+abs2.(γ(ω)*Ψ)).^2))/ψ_norm(Ψ,inds)
        return sparse(1:N_ext, 1:N_ext, V, N_ext, N_ext, +)
    end

    function χʳⁱ′(Ψ,ω)
        V = -2.*(abs2.(γ(ω))*F_ext[:].*real(γ(ω).*Ψ).*D₀.*imag(Ψ)./((1+abs2.(γ(ω)*Ψ)).^2))/ψ_norm(Ψ,inds)
        return sparse(1:N_ext, 1:N_ext, V, N_ext, N_ext, +)
    end

    function χⁱⁱ′(Ψ,ω)
        V = -2*(abs2.(γ(ω)).*F_ext[:].*imag(γ(ω).*Ψ).*D₀.*imag(Ψ)./((1+abs2.(γ(ω)*Ψ)).^2))/ψ_norm(Ψ,inds)
        return sparse(1:N_ext, 1:N_ext, V, N_ext, N_ext, +)
    end

    function ψ_normʳʳ′(Ψ,ω)
        ω² = ω^2
        rows = zeros(Int64,length(inds)*N_ext)
        cols = zeros(Int64,length(inds)*N_ext)
        V = zeros(Complex128,length(inds)*N_ext)
        for i in 1:length(inds)
            rows[N_ext*(i-1) + (1:N_ext)] = 1:N_ext
            cols[N_ext*(i-1) + (1:N_ext)] = fill(inds[i],N_ext)
            V[N_ext*(i-1) + (1:N_ext)] = -2.*real((∇²+ɛ*ω²+χ(Ψ,ω)*ω²+Γ*ω²)*Ψ).*real(Ψ[inds[i]])./(ψ_norm(Ψ,inds)^2)
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
            V[N_ext*(i-1) + (1:N_ext)] = -2.*imag((∇²+ɛ*ω²+χ(Ψ,ω)*ω²+Γ*ω²)*Ψ).*real(Ψ[inds[i]])./(ψ_norm(Ψ,inds)^2)
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
            V[N_ext*(i-1) + (1:N_ext)] = -2.*real((∇²+ɛ*ω²+χ(Ψ,ω)*ω²+Γ*ω²)*Ψ).*imag(Ψ[inds[i]])./(ψ_norm(Ψ,inds)^2)
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
            V[N_ext*(i-1) + (1:N_ext)] = -2.*imag((∇²+ɛ*ω²+χ(Ψ,ω)*ω²+Γ*ω²)*Ψ).*imag(Ψ[inds[i]])./(ψ_norm(Ψ,inds)^2)
        end
        return sparse(rows, cols, V, N_ext, N_ext, +)
    end


    function f!(Ψ,fvec)
        ψ = zeros(Complex128,N_ext)
        ψ[2:N_ext] = (Ψ[2:N_ext]+1im*Ψ[N_ext+(2:N_ext)])
        ψ[1] = 1im*Ψ[N_ext+1]
        ω = Ψ[1]; ω² = ω^2

        temp = (∇²+ɛ*ω²+χ(ψ,ω)*ω²+Γ*ω²)*ψ/ψ_norm(ψ,inds)

        fvec[1:N_ext]         = real(temp)
        fvec[N_ext+(1:N_ext)] = imag(temp)
    end


    function jac!(Ψ,jacarr)

        ψ = zeros(Complex128,N_ext)
        ψ[2:N_ext] = (Ψ[2:N_ext]+1im*Ψ[N_ext+(2:N_ext)])
        ψ[1] = 1im*Ψ[N_ext+1]
        ω = Ψ[1]; ω² = ω^2

        temp = (∇²+ɛ*ω²+χ(ψ,ω)*ω²+Γ*ω²)/ψ_norm(ψ,inds)

        tempr = similar(temp,Float64)
        tempi = similar(temp,Float64)
        tr = nonzeros(tempr)
        ti = nonzeros(tempi)
        tr[:] = real((nonzeros(temp)))
        ti[:] = imag((nonzeros(temp)))

        tempj = [tempr+χʳʳ′(ψ,ω)*ω²+ψ_normʳʳ′(ψ,ω) -tempi+χʳⁱ′(ψ,ω)*ω²+ψ_normʳⁱ′(ψ,ω); tempi+χⁱʳ′(ψ,ω)*ω²+ψ_normⁱʳ′(ψ,ω) tempr+χⁱⁱ′(ψ,ω)*ω²+ψ_normⁱⁱ′(ψ,ω)]
        wemp = 2*ω*(ɛ+χ(ψ,ω)+Γ)*ψ/ψ_norm(ψ,inds) + χʷ′(ψ,ω)*ω²*ψ/ψ_norm(ψ,inds)

        tempj[:,1] = [real(wemp); imag(wemp)]
        jacarr[:,:] = tempj[:,:]

    end

    Ψ_init = Array(Float64,2*N_ext)
    Ψ_init[1:N_ext]         = real(ψ_init)
    Ψ_init[N_ext+(1:N_ext)] = imag(ψ_init)
    Ψ_init[1] = ω_init

    df = DifferentiableSparseMultivariateFunction(f!,jac!)
    z = nlsolve(df,Ψ_init,show_trace = dispOpt, ftol = 2e-8, iterations = 250)

    ψ_ext = zeros(Complex128,N_ext)
    if converged(z)
        ψ_ext = z.zero[1:N_ext] + 1im*z.zero[N_ext+(1:N_ext)]
    else
        ψ_ext = NaN*ψ_ext
    end

    ω = z.zero[1]
    ψ_ext[1] = imag(ψ_ext[1])

    return ψ_ext,ω

end
# end of function solve_lasing



function solve_CPA(inputs::Dict, D₀::Float64; ψ_init=0.000001, k_init=inputs["ω₀"], inds=14, dispOpt=false)

    inputs1 = deepcopy(inputs)

    inputs1["ɛ_ext"] = conj(inputs1["ɛ_ext"])
    inputs1["Γ_ext"] = conj(inputs["Γ_ext"])
    inputs1["γ⟂"] = -inputs["γ⟂"]

    ψ_ext,k = solve_single_mode_lasing(inputs1, -D₀; ψ_init=conj(ψ_init), ω_init=k_init, inds=inds, dispOpt=dispOpt)

    return conj(ψ_ext),k

end # solve_CPA



############################################################################################

function γ(inputs::InputStruct,k::Number)::Complex128
    inputs.γ⟂./(k-inputs.k₀+1im*inputs.γ⟂)
end

function χ(inputs::InputStruct, k::Complex128, F, Ψ::Array{Complex128,1})::SparseMatrixCSC{Complex128,Int64}
    V = inputs.D₀.*(F.*inputs.F_sm).*γ(inputs,k)./(1+abs2.(γ(inputs,k).*Ψ))
    return sparse(1:inputs.N_ext, 1:inputs.N_ext, V, inputs.N_ext, inputs.N_ext, +)
end

function χʳʳ′(inputs::InputStruct, K::Complex128, F, Ψ::Array{Complex128,1})::SparseMatrixCSC{Float64,Int64}
    k = real(K)
    V = -2.*abs2.(γ(inputs,k)).*(F.*inputs.F_sm).*real.(γ(inputs,k).*Ψ).*inputs.D₀.*real(Ψ)./((1+abs2.(γ(inputs,k).*Ψ)).^2)
    return sparse(1:inputs.N_ext, 1:inputs.N_ext, V, inputs.N_ext, inputs.N_ext, +)
end

function χⁱʳ′(inputs::InputStruct, K::Complex128, F, Ψ::Array{Complex128,1})::SparseMatrixCSC{Float64,Int64}
    k = real(K)
    V = -2.*abs2.(γ(inputs,K)).*(F.*inputs.F_sm).*imag.(γ(inputs,k).*Ψ).*inputs.D₀.*real(Ψ)./((1+abs2.(γ(inputs,K).*Ψ)).^2)
    return sparse(1:inputs.N_ext, 1:inputs.N_ext, V, inputs.N_ext, inputs.N_ext, +)
end

function χʳⁱ′(inputs::InputStruct, K::Complex128, F, Ψ::Array{Complex128,1})::SparseMatrixCSC{Float64,Int64}
    k = real(K)
    V = -2.*abs2.(γ(inputs,K)).*(F.*inputs.F_sm).*real.(γ(inputs,K).*Ψ).*inputs.D₀.*imag(Ψ)./((1+abs2.(γ(inputs,K).*Ψ)).^2)
    return sparse(1:inputs.N_ext, 1:inputs.N_ext, V, inputs.N_ext, inputs.N_ext, +)
end

function χⁱⁱ′(inputs::InputStruct, K::Complex128, F, Ψ::Array{Complex128,1})::SparseMatrixCSC{Float64,Int64}
    k = real(K)
    V = -2.*abs2.(γ(inputs,K)).*(F.*inputs.F_sm).*imag.(γ(inputs,K).*Ψ).*inputs.D₀.*imag(Ψ)./((1+abs2.(γ(inputs,K).*Ψ)).^2)
    return sparse(1:inputs.N_ext, 1:inputs.N_ext, V, inputs.N_ext, inputs.N_ext, +)
end

end
# end of Module SALT_1d
