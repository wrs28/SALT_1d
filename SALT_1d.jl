module SALT_1d

export processInputs, computePolesL, computePolesNL1, computePolesNL2, computeZerosL, computeZerosNL1, computeZerosNL2, computeCFs, solve_SPA, solve_scattered, solve_single_mode_lasing, solve_CPA, computeS, bootstrap, CF_analysis, CF_synthesis, computeZerosL2

include("SALT_1d_Core.jl")

using .Core
using NLsolve
using Formatting

include("SALT_1d_Parallel.jl")

####################################################################################

function computePolesL(inputs::Dict, ω, nTCFs::Int; F=1.) 
    # No Line Pulling. Set D or F to zero to get transparent cavity.

    dx = inputs["dx"]
    x_ext = inputs["x_ext"]
    ∂_ext = inputs["∂_ext"]
    ℓ_ext = inputs["ℓ_ext"]
    N_ext = inputs["N_ext"]
    λ = inputs["λ"]
    x_inds = inputs["x_inds"]
    Γ_ext = inputs["Γ_ext"]
    ɛ_ext = inputs["ɛ_ext"]
    D₀ = inputs["D₀"]
    F_ext = inputs["F_ext"]

    r = whichRegion(x_ext, ∂_ext)

    ∇² = laplacian(ℓ_ext, N_ext, 1+1im*σ(x_ext,∂_ext,λ)/ω)

    Γ = zeros(N_ext,1)
    for dd in 2:length(∂_ext)-1
        δ,dummy1 = dirac_δ(x_ext,∂_ext[dd])
        Γ = Γ[:] + full(δ)/Γ_ext[dd]
    end

    ɛΓ⁻¹ = sparse(1:N_ext, 1:N_ext, 1./(ɛ_ext[r]+Γ[:]-1im*D₀.*F.*F_ext[r]), N_ext, N_ext, +)

    (ω²,ψ_ext,nconv,niter,nmult,resid) = eigs(-ɛΓ⁻¹*∇²,which = :LM, nev = nTCFs, sigma = ω^2+1.e-5)

    ψ = zeros(Complex128,length(inputs["x_ext"]),nTCFs)

    for ii = 1:length(ω²)
        N = trapz(ψ_ext[:,ii].*(ɛ_ext[r]+Γ[:]-1im*D₀.*F.*F_ext[r]).*ψ_ext[:,ii],dx)
        ψ_ext[:,ii] = ψ_ext[:,ii]/sqrt(N)
        ψ[:,ii] = ψ_ext[:,ii]
    end

    return sqrt(ω²),ψ

end 
# end of function computePolesL



function computePolesNL1(inputs::Dict, ω_init; F=1., dispOpt = false, η_init = 1e-13+0.0im, u_init = [], ω_avoid = [.0],tol = .7)
    # With Line Pulling. Nonlinear solve using TCF evecs and evals
    
    function γ(ω)
        inputs["γ⟂"]./(ω-inputs["ω₀"]+1im*inputs["γ⟂"])
    end
    
    u = NaN*ones(Complex128,inputs["N_ext"],1)
    η_init = [η_init]
    η_init[:,1],u[:,1] = computeCFs(inputs, ω_init, 1, η_init=η_init[1], ψ_init = u_init)

    dx = inputs["dx"]
    r = whichRegion(inputs["x_ext"],inputs["∂_ext"])
    F = inputs["F_ext"][r]
    
    function f!(z, fvec)

        ω = z[1]+1im*z[2]
    
        flag = true
        count = 1
        max_count = 15
        M = 1
        ind = Int
        while flag
            
            η_temp,u_temp = computeCFs(inputs, ω, M; η_init=η_init[1], ψ_init = u[:,1])
            overlap = zeros(Float64,M)
            
            for i in 1:M
                overlap[i] = abs(trapz(u[:,1].*F.*u_temp[:,i],dx))
            end
            
            ind = findmax(overlap)[2]
    
            if (abs(overlap[ind]-1) < (1-tol))
                flag = false
                η_init[:,1] = η_temp[ind]
                u[:,1] = u_temp[:,ind]
            elseif  (count < max_count) & (abs(overlap[ind]-1) ≥ (1-tol))
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
    
        fvec[1] = real((η-inputs["D₀"]*γ(ω))/prod(ω-ω_avoid))
        fvec[2] = imag((η-inputs["D₀"]*γ(ω))/prod(ω-ω_avoid))

    end

    z = nlsolve(f!,[real(ω_init),imag(ω_init)]; iterations = 50, show_trace = dispOpt)
    ω = z.zero[1]+1im*z.zero[2]
    conv = converged(z)
    
    η_init[1] = inputs["D₀"]*γ(ω)
    η,u[:,1] = computeCFs(inputs, ω, 1; η_init=η_init[1])
    
    return ω,u[:,1],η[1],conv

end
# end of function computePolesNL1




function computePolesNL2(inputs::Dict, ω, Radii; Nq=100, nevals=3, F=1., R_min = .01)
    # With Line Pulling, using contour integration

    rank_tol = 2e-4

    dx = inputs["dx"]
    x_ext = inputs["x_ext"]
    ∂_ext = inputs["∂_ext"]
    ℓ_ext = inputs["ℓ_ext"]
    N_ext = inputs["N_ext"]
    λ = inputs["λ"]
    x_inds = inputs["x_inds"]
    F_ext = inputs["F_ext"]
    D₀ = inputs["D₀"]
    ɛ_ext = inputs["ɛ_ext"]
    Γ_ext = inputs["Γ_ext"]

    r = whichRegion(x_ext,∂_ext)

    Γ = zeros(N_ext,1)
    for dd in 2:length(∂_ext)-1
        δ,dummy1 = dirac_δ(x_ext,∂_ext[dd])
        Γ = Γ[:] + full(δ)/Γ_ext[dd]
    end

    ∇² = laplacian(ℓ_ext, N_ext, 1+1im*σ(x_ext,∂_ext,λ)/ω)
    ɛ = sparse(1:N_ext, 1:N_ext, ɛ_ext[r], N_ext, N_ext, +)
    Γ = sparse(1:N_ext, 1:N_ext, Γ[:]    , N_ext, N_ext, +)

    function γ(ω′)
        return inputs["γ⟂"]/(ω′-inputs["ω₀"]+1im*inputs["γ⟂"])
    end

    A  = zeros(Complex128,N_ext,nevals)
    A₀ = zeros(Complex128,N_ext,nevals)
    A₁ = zeros(Complex128,N_ext,nevals)

    rad(a,b,θ) = b./sqrt(sin(θ).^2+(b/a)^2.*cos(θ).^2)
    θ = angle(inputs["ω₀"]-1im*inputs["γ⟂"]-ω)
    flag = abs(inputs["ω₀"]-1im*inputs["γ⟂"]-ω) < rad(Radii[1],Radii[2],θ)
    
    M = rand(N_ext,nevals)
    ϕ = 2π*(0:1/Nq:(1-1/Nq))
    Ω = ω + Radii[1]*cos(ϕ) + 1im*Radii[2]*sin(ϕ)

    if flag
        AA  = zeros(Complex128,N_ext,nevals)
        AA₀ = zeros(Complex128,N_ext,nevals)
        AA₁ = zeros(Complex128,N_ext,nevals)
        RR = 2*R_min
        ΩΩ = inputs["ω₀"]-1im*inputs["γ⟂"] + (RR/2)*cos(ϕ) + 1im*(RR/2)*sin(ϕ)
    end
    
    for i in 1:Nq

        ω′ = Ω[i]
        ω′² = ω′^2

        if (i > 1) & (i < Nq)
            dω′ = (Ω[i+1]-Ω[i-1]  )/2
        elseif i == Nq
            dω′ = (Ω[1]  -Ω[end-1])/2
        elseif i == 1
            dω′ = (Ω[2]  -Ω[end]  )/2
        end

        χω′² = sparse(1:N_ext, 1:N_ext, D₀*γ(ω′)*F.*F_ext[r]*ω′², N_ext, N_ext, +)

        A = (∇²+(ɛ+Γ)*ω′²+χω′²)\M
        A₀ += A*dω′/(2π*1im)
        A₁ += A*ω′*dω′/(2π*1im)

        if flag
            ωω′ = ΩΩ[i]
            ωω′² = ωω′^2
            if (i > 1) & (i < Nq)
                dωω′ = (ΩΩ[i+1]-ΩΩ[i-1]  )/2
            elseif i == Nq
                dωω′ = (ΩΩ[1]  -ΩΩ[end-1])/2
            elseif i == 1
                dωω′ = (ΩΩ[2]  -ΩΩ[end]  )/2
            end
            χωω′² = sparse(1:N_ext, 1:N_ext, D₀*γ(ωω′)*F.*F_ext[r]*ωω′², N_ext, N_ext, +)
            
            AA = (∇²+(ɛ+Γ)*ωω′²+χωω′²)\M
            AA₀ += AA*dωω′/(2π*1im)
            AA₁ += AA*ωω′*dωω′/(2π*1im)

       end
        
    end

    if flag
        A = A-AA
        A₀ = A₀-AA₀
        A₁ = A₁-AA₁
    end
    
    P = svdfact(A₀,thin = false)
    temp = find(P[:S] .< rank_tol)
    if isempty(temp)
        println("error. need more nevals")
        return
    else
        k = temp[1]-1
    end

    B = (P[:U][:,1:k])'*A₁*(P[:Vt][1:k,:])'*diagm(1./P[:S][1:k])

    D,V = eig(B)

    return D

end 
# end of function computePolesNL2



function computeZerosL(inputs::Dict, ω, nTCFs::Int; F=1.)
    # No Line Pulling. Set D or F to zero to get transparent cavity.
    
    inputs1 = copy(inputs)
    
    inputs1["ɛ_ext"] = conj(inputs1["ɛ_ext"])
    inputs1["Γ_ext"] = conj(inputs["Γ_ext"])
    inputs1["γ⟂"] = -inputs["γ⟂"]
    
    ωz,ψz = computePolesL(inputs1, conj(ω), nTCFs; F=F)
    
    return conj(ωz),conj(ψz)

end 
# end of function computeZerosL



function computeZerosNL1(inputs::Dict, ω_init; F=1., dispOpt = false, β_init = 1e-13+0.0im, v_init = [], ω_avoid = [0.])
    # With Line Pulling. Nonlinear solve using TCF evecs and evals
    
    inputs1 = copy(inputs)
    
    inputs1["ɛ_ext"] = conj(inputs1["ɛ_ext"])
    inputs1["Γ_ext"] = conj(inputs["Γ_ext"])
    inputs1["γ⟂"] = -inputs["γ⟂"]
    inputs1["D₀"] = -inputs["D₀"]
    
    ω,u,η,conv = computePolesNL1(inputs1, conj(ω_init); F=F, dispOpt=dispOpt, η_init=conj(β_init), u_init=conj(v_init), ω_avoid = conj(ω_avoid))
    
    return conj(ω),conj(u),conj(η),conv
    
end
# end of function computeZerosNL1


function computeZerosNL2(inputs::Dict, ω, Radii; Nq=100, nevals=3, F=1., R_min = .01)
    
    inputs1 = copy(inputs)
    
    inputs1["ɛ_ext"] = conj(inputs1["ɛ_ext"])
    inputs1["Γ_ext"] = conj(inputs["Γ_ext"])
    inputs1["γ⟂"] = -inputs["γ⟂"]
    inputs1["D₀"] = -inputs["D₀"]
    
    ω = computePolesNL2(inputs1, conj(ω), Radii; Nq=Nq, nevals=nevals, F=F, R_min=R_min)
    
    return conj(ω)
    
end



function computeCFs(inputs::Dict, ω, nTCFs::Int; F=1., η_init = [], ψ_init = [])
    # No Line Pulling, calculation independent of pump D

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

    ω²= ω^2

    r = whichRegion(x_ext,∂_ext)

    ∇² = laplacian(ℓ_ext, N_ext, 1+1im*σ(x_ext,∂_ext,λ)/ω)

    Γ = zeros(N_ext,1)
    for dd in 2:length(∂_ext)-1
        δ,dummy1 = dirac_δ(x_ext,∂_ext[dd])
        Γ = Γ[:] + full(δ)/Γ_ext[dd]
    end

    ɛω² = sparse(1:N_ext, 1:N_ext, ɛ_ext[r]*ω²   ,N_ext, N_ext, +)
    Γω² = sparse(1:N_ext, 1:N_ext, Γ[:]*ω²       , N_ext, N_ext, +)
    sF = sparse(1:N_ext, 1:N_ext, sign(F.*F_ext[r]), N_ext, N_ext, +)
    FF = sparse(1:N_ext, 1:N_ext, abs(F.*F_ext[r]), N_ext, N_ext, +)

    if isempty(η_init) & isempty(ψ_init)
        (η,ψ_ext,nconv,niter,nmult,resid) = eigs(-sF*(∇²+ɛω²+Γω²)/ω²,FF, which = :LM, nev = nTCFs, sigma = 1e-8)
    elseif !isempty(η_init) & isempty(ψ_init)
        (η,ψ_ext,nconv,niter,nmult,resid) = eigs(-sF*(∇²+ɛω²+Γω²)/ω²,FF, which = :LM, nev = nTCFs, sigma = η_init)
    elseif !isempty(η_init) & !isempty(ψ_init)
        (η,ψ_ext,nconv,niter,nmult,resid) = eigs(-sF*(∇²+ɛω²+Γω²)/ω²,FF, which = :LM, nev = nTCFs, sigma = η_init, v0 = ψ_init)
    end

    ψ = zeros(Complex128,length(inputs["x_ext"]),nTCFs)

    for ii = 1:length(η)
        N = trapz(ψ_ext[:,ii].*F.*F_ext[r].*ψ_ext[:,ii],dx)
        ψ_ext[:,ii] = ψ_ext[:,ii]/sqrt(N)
        ψ[:,ii] = ψ_ext[:,ii]
    end

    return η,ψ

end 
# end of function computeCFs



function solve_SPA(inputs::Dict, ω; z₀₊=0.001im, z₀₋=0.001im, F = 1., u=[], η=[], φ₊=[], φ₋=[])

    x = inputs["x_ext"]
    dx = inputs["dx"]
    ∂_ext = inputs["∂_ext"]
    N_ext = inputs["N_ext"]
    D₀ = inputs["D₀"]

    function γ()
        return inputs["γ⟂"]/(ω-inputs["ω₀"]+1im*inputs["γ⟂"])
    end

    a = copy(inputs["a"])

    if isempty(φ₊) | isempty(φ₋)
        inputs["a"] = [1.0,0.0]
        φ₊ = solve_scattered(inputs,ω; isNonLinear = false, F = 0.)
        inputs["a"] = [0.0,1.0]
        φ₋ = solve_scattered(inputs,ω; isNonLinear = false, F = 0.)
    end
    inputs["a"] = copy(a)

    if isempty(η) | isempty(u)
        η,u = computeCFs(inputs,ω,1)
    end

    r = whichRegion(x,∂_ext)
    F_ext = inputs["F_ext"][r]

    function f₊!(z, fvec)

        b = inputs["a"][1]
        Z = z[1]+1im*z[2]

        numerator = u.*(F.*F_ext).*(Z*u+b*φ₊)
        denominator = 1+abs2(γ())*abs2(Z*u+b*φ₊)
        term = (D₀*γ()./η).*trapz(numerator./denominator,dx)

        fvec[1] = real(term[1]/Z - 1)
        fvec[2] = imag(term[1]/Z - 1)

        return term,Z

    end

    function f₋!(z,fvec)

        b = inputs["a"][2]
        Z = z[1]+1im*z[2]

        numerator = u.*(F.*F_ext).*(Z*u+b*φ₋)
        denominator = 1+abs2(γ())*abs2(Z*u+b*φ₋)
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



function solve_scattered(inputs::Dict, ω; isNonLinear=false, ψ_init=0, F=1., dispOpt = false)

    dx = inputs["dx"]
    x_ext = inputs["x_ext"]
    ∂_ext = inputs["∂_ext"]
    ℓ_ext = inputs["ℓ_ext"]
    N_ext = inputs["N_ext"]
    λ = inputs["λ"]
    x_inds = inputs["x_inds"]
    F_ext = inputs["F_ext"]
    D₀ = inputs["D₀"]
    a = inputs["a"]
    Γ_ext = inputs["Γ_ext"]
    ɛ_ext = inputs["ɛ_ext"]

    function γ()
        return inputs["γ⟂"]/(ω-inputs["ω₀"]+1im*inputs["γ⟂"])
    end

    ω²= ω^2

    r = whichRegion(x_ext,∂_ext)

    ∇² = laplacian(ℓ_ext,N_ext,1+1im*σ(x_ext,∂_ext,real(λ))/real(ω))

    Γ = zeros(N_ext,1)
    for dd in 2:length(∂_ext)-1
        δ,dummy1 = dirac_δ(x_ext,∂_ext[dd])
        Γ = Γ[:] + full(δ)/Γ_ext[dd]
    end

    ɛω² = sparse(1:N_ext,1:N_ext, ɛ_ext[r]*ω²            ,N_ext, N_ext, +)
    χω² = sparse(1:N_ext,1:N_ext, D₀*γ()*(F.*F_ext[r])*ω²,N_ext, N_ext, +)
    Γω² = sparse(1:N_ext,1:N_ext, Γ[:]*ω²                ,N_ext, N_ext, +)

    δᵨ₊,∇δᵨ₊ = dirac_δ(x_ext,inputs["xᵨ₊"])
    Θᵨ₊,∇Θᵨ₊,∇²Θᵨ₊ = heaviside_Θ(x_ext,inputs["xᵨ₊"])
    x₀ = inputs["x"][1]
    j₊ = a[1]*(+∇δᵨ₊.*exp(+1im*ω*(x_ext-x₀)) + 2*1im*ω*δᵨ₊.*exp(+1im*ω*(x_ext-x₀)))
    
    δᵨ₋,∇δᵨ₋ = dirac_δ(x_ext,inputs["xᵨ₋"])
    Θᵨ₋,∇Θᵨ₋,∇²Θᵨ₋ = heaviside_Θ(x_ext,inputs["xᵨ₋"])
    Θᵨ₋ = 1-Θᵨ₋
    x₀ = inputs["x"][end]
    j₋ = a[2]*(-∇δᵨ₋.*exp(-1im*ω*(x_ext-x₀)) + 2*1im*ω*δᵨ₋.*exp(-1im*ω*(x_ext-x₀)))
 
    j = j₊ + j₋

    if ψ_init == 0 || !isNonLinear
        ψ_ext = (∇²+ɛω²+χω²+Γω²)\full(j)
    else
        ψ_ext = ψ_init
    end


    function χ(Ψ)
        V = D₀*(F.*F_ext[r]).*γ()./(1+abs2(γ()*Ψ))
        return sparse(1:N_ext, 1:N_ext, V, N_ext, N_ext, +)
    end

    function χʳʳ′(Ψ)
        V = -2.*abs2(γ()).*(F.*F_ext[r]).*real(γ().*Ψ).*D₀.*real(Ψ)./((1+abs2(γ()*Ψ)).^2)
        return sparse(1:N_ext, 1:N_ext, V, N_ext, N_ext, +)
    end

    function χⁱʳ′(Ψ)
        V = -2.*abs2(γ()).*(F.*F_ext[r]).*imag(γ().*Ψ).*D₀.*real(Ψ)./((1+abs2(γ()*Ψ)).^2)
        return sparse(1:N_ext, 1:N_ext, V, N_ext, N_ext, +)
    end

    function χʳⁱ′(Ψ)
        V = -2.*abs2(γ()).*(F.*F_ext[r]).*real(γ().*Ψ).*D₀.*imag(Ψ)./((1+abs2(γ()*Ψ)).^2)
        return sparse(1:N_ext, 1:N_ext, V, N_ext, N_ext, +)
    end

    function χⁱⁱ′(Ψ)
        V = -2.*abs2(γ()).*(F.*F_ext[r]).*imag(γ().*Ψ).*D₀.*imag(Ψ)./((1+abs2(γ()*Ψ)).^2)
        return sparse(1:N_ext, 1:N_ext, V, N_ext, N_ext, +)
    end

    function f!(Ψ,fvec)
        ψ = zeros(Complex128,N_ext)
        ψ = Ψ[1:N_ext]+1im*Ψ[N_ext+(1:N_ext)]
        temp = (∇²+ɛω²+Γω²+χ(ψ)*ω²)*ψ - j
        fvec[1:N_ext] = real(temp)
        fvec[N_ext+(1:N_ext)] = imag(temp)
    end

    function jac!(Ψ,jacarr)
        ψ = zeros(Complex128,N_ext)
        ψ = Ψ[1:N_ext]+1im*Ψ[N_ext+(1:N_ext)]
        temp = ∇²+ɛω²+Γω²+χ(ψ)*ω²
        tempr = similar(temp,Float64)
        tempi = similar(temp,Float64)
        tr = nonzeros(tempr)
        ti = nonzeros(tempi)
        tr[:] = real((nonzeros(temp)))
        ti[:] = imag((nonzeros(temp)))
        tempj = [tempr+χʳʳ′(ψ)*ω² -tempi+χʳⁱ′(ψ)*ω²; tempi+χⁱʳ′(ψ)*ω² tempr+χⁱⁱ′(ψ)*ω²]
        jacarr[:,:] = tempj[:,:]
    end

    if isNonLinear
        Ψ_init = Array(Float64,2*N_ext)
        Ψ_init[1:N_ext] = real(ψ_ext)
        Ψ_init[N_ext+(1:N_ext)] = imag(ψ_ext)

        df = DifferentiableSparseMultivariateFunction(f!, jac!)
        z = nlsolve(df,Ψ_init, show_trace = dispOpt ,ftol = 1e-6, iterations = 150)

        if converged(z)
            ψ_ext = z.zero[1:N_ext] + im*z.zero[N_ext+(1:N_ext)]
        else
            ψ_ext = NaN*ψ_ext;
            println("Warning, solve_scattered returned NaN")
        end

    end

    return ψ_ext

end 
# end of function solve_scattered



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

    r = whichRegion(x_ext, ∂_ext)
    ∇² = laplacian(ℓ_ext, N_ext,1+1im*σ(x_ext,∂_ext,real(λ))/real(inputs["ω₀"]))

    Γ = zeros(N_ext,1)
    for dd in 2:length(∂_ext)-1
        δ,dummy1 = dirac_δ(x_ext,∂_ext[dd])
        Γ = Γ[:] + full(δ)/Γ_ext[dd]
    end

    ɛ = sparse(1:N_ext,1:N_ext,ɛ_ext[r],N_ext,N_ext,+)
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
        return sum(abs2(Ψ[inds]))
    end

    function χ(Ψ,ω)
        V = F_ext[r].*γ(ω)*D₀./(1+abs2(γ(ω)*Ψ))
        return sparse(1:N_ext, 1:N_ext, V, N_ext, N_ext,+)
    end

    function χʷ′(Ψ,ω)
        V = F_ext[r].*γ′(ω).*D₀./(1+abs2(γ(ω).*Ψ)) - Γ′(ω).*abs2(Ψ).*F_ext[r].*γ(ω).*D₀./((1+abs2(γ(ω).*Ψ)).^2)
        return sparse(1:N_ext, 1:N_ext, V, N_ext, N_ext, +)
    end

    function χʳʳ′(Ψ,ω)
        V = -2.*(abs2(γ(ω))*F_ext[r].*real(γ(ω).*Ψ).*D₀.*real(Ψ)./((1+abs2(γ(ω)*Ψ)).^2))/ψ_norm(Ψ,inds)
        return sparse(1:N_ext, 1:N_ext, V, N_ext, N_ext, +)
    end

    function χⁱʳ′(Ψ,ω)
        V = -2.*(abs2(γ(ω))*F_ext[r].*imag(γ(ω).*Ψ).*D₀.*real(Ψ)./((1+abs2(γ(ω)*Ψ)).^2))/ψ_norm(Ψ,inds)
        return sparse(1:N_ext, 1:N_ext, V, N_ext, N_ext, +)
    end

    function χʳⁱ′(Ψ,ω)
        V = -2.*(abs2(γ(ω))*F_ext[r].*real(γ(ω).*Ψ).*D₀.*imag(Ψ)./((1+abs2(γ(ω)*Ψ)).^2))/ψ_norm(Ψ,inds)
        return sparse(1:N_ext, 1:N_ext, V, N_ext, N_ext, +)
    end

    function χⁱⁱ′(Ψ,ω)
        V = -2*(abs2(γ(ω)).*F_ext[r].*imag(γ(ω).*Ψ).*D₀.*imag(Ψ)./((1+abs2(γ(ω)*Ψ)).^2))/ψ_norm(Ψ,inds)
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


function solve_CPA(inputs::Dict, D₀::Float64; ψ_init=0.000001, ω_init=inputs["ω₀"], inds=14,dispOpt = false)
   
    inputs1 = copy(inputs)

    inputs1["ɛ_ext"] = conj(inputs1["ɛ_ext"])
    inputs1["Γ_ext"] = conj(inputs["Γ_ext"])
    inputs1["γ⟂"] = -inputs["γ⟂"]
    inputs1["D₀"] = -inputs["D₀"]
    
    ψ_ext,ω = solve_single_mode_lasing(inputs1, -D₀; ψ_init=conj(ψ_init), ω_init=ω_init, inds=inds, dispOpt = dispOpt)
    
    return conj(ψ_ext),conj(ω)
    
end


function computeS(inputs::Dict; N=10, N_Type="D", isNonLinear=false, F=1., dispOpt = true, ψ_init = [], fileName = [])
    # N is the number of steps to go from D0 = 0 to given D0

    x_inds = inputs["x_inds"]

    function γ(ω)
        return inputs["γ⟂"]/(ω-inputs["ω₀"]+1im*inputs["γ⟂"])
    end

    ψ₊ = Array(Complex128,length(inputs["x_ext"]))
    ψ₋ = Array(Complex128,length(inputs["x_ext"]))
    ψ  = Array(Complex128,length(inputs["x_ext"]))
    
    if isempty(ψ_init) & (N>1)
        S = NaN*ones(Complex128,2,2,length(inputs["ω"]),N)
    else
        S = NaN*ones(Complex128,2,2,length(inputs["ω"]),1)
    end
        
    D₀ = copy(inputs["D₀"])
    A = copy(inputs["a"])
    
    if N > 1
        D = linspace(0,D₀,N)
        A_vec = linspace(0.001,1,N)
    else
        D = [inputs["D₀"]]
        A_vec = [1]
    end

    for ii in 1:length(inputs["ω"])

        ω = inputs["ω"][ii]

        if (ii/1 == round(ii/1)) & dispOpt
            printfmtln("Solving for frequency {1:d} of {2:d}, ω = {3:2.3f}.",ii,length(inputs["ω"]),ω)
        end

        if isempty(ψ_init)

            for j in 1:N

                if N_Type == "D"
                    inputs["D₀"] = D[j]
                elseif N_Type == "A"
                    inputs["a"] = A_vec[j]*A
                end
                    
                if isNonLinear
                    if isnan(ψ[1]) | j==1
                        ψ = solve_scattered(inputs,ω,isNonLinear = true, F = F)
                    else
                        ψ = solve_scattered(inputs,ω,isNonLinear = true, F = F, ψ_init = ψ)
                    end
                else
                    ψ = 0.
                end

                inputs["a"] = [1.0,0.0]
                ψ₊ = solve_scattered(inputs,ω,isNonLinear = false, F = F./(1+abs2(γ(ω)*ψ)))

                inputs["a"] = [0.0,1.0]
                ψ₋ = solve_scattered(inputs,ω,isNonLinear = false, F = F./(1+abs2(γ(ω)*ψ)))

                S[1,1,ii,j] = ψ₊[x_inds[1]]*exp(+1im*inputs["dx"]*ω)
                S[2,1,ii,j] = ψ₊[x_inds[end]]
                S[1,2,ii,j] = ψ₋[x_inds[1]]
                S[2,2,ii,j] = ψ₋[x_inds[end]]*exp(-1im*inputs["dx"]*ω)
                
                inputs["D₀"] = D₀
                inputs["a"] = A
                
                if !isempty(fileName)
                    foo1(fid) = serialize(fid,(inputs,D,S,ii,j))
                    open(foo1,fileName,"w")
                end
                
            end
            
        else
            if isNonLinear
                ψ = solve_scattered(inputs,ω,isNonLinear = true, F = F, ψ_init = ψ_init)
            else
                ψ = 0.
            end

            inputs["a"] = [1.0,0.0]
            ψ₊ = solve_scattered(inputs,ω,isNonLinear = false, F = F./(1+abs2(γ(ω)*ψ)))

            inputs["a"] = [0.0,1.0]
            ψ₋ = solve_scattered(inputs,ω,isNonLinear = false, F = F./(1+abs2(γ(ω)*ψ)))

            S[1,1,ii,1] = ψ₊[x_inds[1]]*exp(+1im*inputs["dx"]*ω)
            S[2,1,ii,1] = ψ₊[x_inds[end]]
            S[1,2,ii,1] = ψ₋[x_inds[1]]
            S[2,2,ii,1] = ψ₋[x_inds[end]]*exp(-1im*inputs["dx"]*ω)
       
            inputs["D₀"] = D₀
            inputs["a"] = A
            
            if !isempty(fileName)
                foo2(fid) = serialize(fid,(inputs,D,S,ii,1))
                open(foo2,fileName,"w")
            end
            
        end
        
    end
    
    return S

end 
# end of fuction computeS 



function CF_analysis(ψ,inputs,ω,nCF)
    
    η1,u1 = computeCFs(inputs,ω,2*nCF)
    perm = sortperm(η1, by = abs)
    η = η1[perm]
    u = u1[:,perm]
    a = NaN*zeros(Complex128,nCF)
    b = copy(a)
    
    r = SALT_1d.Core.whichRegion(inputs["x_ext"],inputs["∂_ext"])
    F = inputs["F_ext"][r]
    
    for i in 1:nCF
        a[i] = SALT_1d.Core.trapz(F.*ψ.*u[:,i],inputs["dx"]) 
        b[i] = SALT_1d.Core.trapz(F.*ψ.*conj(u[:,i]),inputs["dx"]) 
    end
    
    return a,b
    
end 
# end of function CF_analysis


function CF_synthesis(a,inputs,ω)
    
    nCF = length(a)
    
    η1,u1 = computeCFs(inputs,ω,2*nCF)
    perm = sortperm(η1, by = abs)
    η = η1[perm]
    u = u1[:,perm]
    
    ψ = zeros(Complex128,inputs["N_ext"])
    
    for i in 1:nCF
        ψ += a[i].*u[:,i]
    end
    
    return ψ
    
end 
# end of function CF_synthesis

############################################################################################

end 
# end of Module SALT_1D