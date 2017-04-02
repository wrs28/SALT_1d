module SALT_2d

export processInputs, updateInputs, computePolesL
#, updateInputs, computePolesL, computePolesNL1, computePolesNL2, computeZerosL, computeZerosNL1, computeZerosNL2, computeCFs, solve_SPA, solve_scattered, solve_single_mode_lasing, solve_CPA, computeS, bootstrap, CF_analysis, CF_synthesis, computeZerosL2

include("SALT_2d_Core.jl")

using .Core

using NLsolve
using Formatting


#######################################################################################################

using .Core
using NLsolve
using Formatting



function computePolesL(inputs::Dict, k::Number, nPoles::Int; F=1., truncate = false) 
    # No Line Pulling. Set D or F to zero to get transparent cavity.

    ## DEFINITIONS BLOCK ##
    x_ext = inputs["x_ext"]
    y_ext = inputs["u_ext"]
    ∂_ext = inputs["∂_ext"]
    N_ext = inputs["N_ext"]; Nₓ = N_ext[1]; Nᵤ = N_ext[2]
    ɛ_ext = inputs["ɛ_ext"]
    D₀ = inputs["D₀"]
    F_ext = inputs["F_ext"]
    ## END OF DEFINITIONS BLOCK ##

    r = whichRegion((x_ext,y_ext),∂_ext,inputs["geometry"])
    ɛ_ext = subpixelSmoothing(inputs; ext = true, r = r)
    
    ∇² = laplacian(k,inputs)
        
#    ɛ⁻¹ = sparse(1:Nₓ*Nᵤ, 1:Nₓ*Nᵤ, 1./(ɛ_ext[r[:]]-1im*D₀.*F.*F_ext[r[:]]), Nₓ*Nᵤ, Nₓ*Nᵤ, +)
    ɛ⁻¹ = sparse(1:Nₓ*Nᵤ, 1:Nₓ*Nᵤ, 1./(ɛ_ext[:]-1im*D₀.*F.*F_ext[r[:]]), Nₓ*Nᵤ, Nₓ*Nᵤ, +)    
    
    (k²,ψ_ext,nconv,niter,nmult,resid) = eigs(-ɛ⁻¹*∇²,which = :LM, nev = nPoles, sigma = k^2)
    
    
    ψ = zeros(Complex128, Nₓ*Nᵤ, nPoles)

    inds = inputs["xu_inds"]
    r1 = r[inds]
    if length(F)>1
        F_temp = F[r1]
    else
        F_temp = F
    end
    for ii = 1:length(k²)
        N = trapz((ɛ_ext[inputs["xu_inds"]]-1im*D₀.*F_temp.*inputs["F_ext"][r1]).*abs2(ψ_ext[inds,ii]),inputs["dr"])
        ψ[:,ii] = ψ_ext[:,ii]/sqrt(N)
    end
   
    if truncate
        return sqrt(k²),ψ[inputs["xu_inds"],:]
    else
        return sqrt(k²),ψ
    end
    
end



    function computePolesNL(inputs, ω, Radii; Nq=100, nevals=3, F=1.) #With Line Pulling

        rank_tol = 2e-4
    
        dr = inputs["dr"]; dx = dr[1]; dy = dr[2]
        x_ext = inputs["x_ext"]
        y_ext = inputs["u_ext"]
        ∂_ext = inputs["∂_ext"]
        ℓ_ext = inputs["ℓ_ext"]
        N_ext = inputs["N_ext"]; Nₓ = N_ext[1]; Nᵤ = N_ext[2]
        F_ext = inputs["F_ext"]
        D₀ = inputs["D₀"]
        ɛ_ext = inputs["ɛ_ext"]

        r = whichRegion((x_ext,y_ext),∂_ext,inputs["geometry"])

        ∇² = laplacian(ω,inputs)
    
        ɛ = sparse(1:Nₓ*Nᵤ, 1:Nₓ*Nᵤ, ɛ_ext[r[:]], Nₓ*Nᵤ, Nₓ*Nᵤ, +)


        function γ(ω′)
            return inputs["γ⊥"]/(ω′-inputs["ω₀"]+1im*inputs["γ⊥"])
        end
    
        A  = zeros(Complex128,Nₓ*Nᵤ,nevals)
        A₀ = zeros(Complex128,Nₓ*Nᵤ,nevals)
        A₁ = zeros(Complex128,Nₓ*Nᵤ,nevals)
        M = rand(Nₓ*Nᵤ,nevals)
        Ω = ω + Radii[1]*cos(2π*(0:1/Nq:(1-1/Nq))) + 1im*Radii[2]*sin(2π*(0:1/Nq:(1-1/Nq)))
    
        for i in 1:Nq
            ω′ = Ω[i]
            ω′² = ω′^2
            if i > 1
                dω′ = Ω[i]-Ω[i-1]
            else
                dω′ = Ω[1]-Ω[end]
            end
        
            χω′² = sparse(1:Nₓ*Nᵤ, 1:Nₓ*Nᵤ, D₀*γ(ω′)*F.*F_ext[r[:]]*ω′², Nₓ*Nᵤ, Nₓ*Nᵤ, +)

            A = (∇²+ɛ*ω′²+χω′²)\M
            A₀ += A*dω′/(2π*1im)
            A₁ += A*ω′*dω′/(2π*1im)

        end

        P = svdfact(A₀,thin = true)
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




    function solve_SPA1(inputs, ω; z₀₊=0.001im, z₀₋=0.001im, F = 1., u=[], η=[], φ₊=[], φ₋=[])

        x = inputs["x_ext"][inputs["x_inds"]]
        y = inputs["u_ext"][inputs["u_inds"]]
        ∂_ext = inputs["∂_ext"]
        N_ext = inputs["N_ext"]

        D₀ = copy(inputs["D₀"])
        a = copy(inputs["a"])
        inputs["D₀"] = 0.
        inputs["a"] = 1.

        ϕ,dummy = scatteredFields(inputs,inputs["ω₀"],isNonLinear = false)
        X_inds = [i for i in inputs["x_inds"], j in inputs["u_inds"]]
        Y_inds = [j for i in inputs["x_inds"], j in inputs["u_inds"]]
        φ = ϕ[Y_inds[:]*N_ext[1] + X_inds[:]]

        inputs["D₀"] = copy(D₀)
        inputs["a"] = copy(a)

        η,u = computeCF(inputs,ω,1);

        function γ()
            return inputs["γ⊥"]/(ω-inputs["ω₀"]+1im*inputs["γ⊥"])
        end

        r = whichRegion((x,y),∂_ext,inputs["geometry"])
        F = inputs["F_ext"][r[:]]

        function f!(z,fvec)

            b = inputs["a"]
            Z = z[1]+1im*z[2]

            numerator = u.*F.*(Z*u+b*φ)
            denominator = 1+abs2(Z*u+b*φ)
            term = (inputs["D₀"]*γ()./η).*trapz(numerator./denominator,inputs["dr"])

            fvec[1] = real(term[1] - Z)
            fvec[2] = imag(term[1] - Z)

            return term,Z

        end

        result = nlsolve(f!,[real(z₀),imag(z₀)])

        z = result.zero[1]+1im*result.zero[2]

        ψ = inputs["a"]*φ+z*u

        return ψ

    end



    function computeCFS(inputs,ω,nTCFs;F=1)

        dr = inputs["dr"]
        dx = dr[1];dy = dr[2]
        x_ext = inputs["x_ext"]
        y_ext = inputs["u_ext"]
        ∂_ext = inputs["∂_ext"]
        ℓ_ext = inputs["ℓ_ext"]
        N_ext = inputs["N_ext"]
        Nₓ=inputs["N"][1]; Nᵤ = inputs["N"][2]
        λ = inputs["λ"]
        x_inds = inputs["x_inds"]
        y_inds = inputs["u_inds"]
        F_ext = inputs["F_ext"]
        D₀ = inputs["D₀"]
        a = inputs["a"]
        ɛ_ext = inputs["ɛ_ext"]

        ω²= ω^2
        dx²=dx^2
        dy²=dy^2

        r = whichRegion((x_ext,y_ext),∂_ext,inputs["geometry"])

        ∇² = laplacian(ω,inputs)

        ɛω² = sparse(1:prod(N_ext),1:prod(N_ext),ɛ_ext[r[:]]*ω²   ,prod(N_ext),prod(N_ext),+)
        Fω² = sparse(1:prod(N_ext),1:prod(N_ext),F.*F_ext[r[:]]*ω²,prod(N_ext),prod(N_ext),+)

        (η,ψ_ext,nconv,niter,nmult,resid) = eigs(-(∇²+ɛω²),Fω², which = :LM, nev = nTCFs, sigma = 1e-12)
    
        ψ = zeros(Complex128,Nₓ*Nᵤ,nTCFs)
        
#        X_inds = [i for i in inputs["x_inds"], j in inputs["u_inds"]]
#        Y_inds = [j for i in inputs["x_inds"], j in inputs["u_inds"]]
    
        for ii = 1:length(η)
            N = trapz(ψ_ext[:,ii].*F_ext[r[:]].*ψ_ext[:,ii],dr)
            ψ_ext[:,ii] = ψ_ext[:,ii]/sqrt(N)
#            ψ[:,ii] = ψ_ext[Y_inds[:]*N_ext[1] + X_inds[:],ii]
            ψ[:,ii] = ψ_ext[:,ii]
        end
                
        return η,ψ

    end

 
    function solve_scattered(inputs,ω; isNonLinear = false, ψ_init = 0)

        dr = inputs["dr"]
        dx = dr[1]; dy = dr[2]
        x_ext = inputs["x_ext"]
        y_ext = inputs["u_ext"]
        ∂_ext = inputs["∂_ext"]
        ℓ_ext = inputs["ℓ_ext"]
        N_ext = inputs["N_ext"]; Nₓ = N_ext[1]; Nᵤ = N_ext[2]
        λ = inputs["λ"]
        x_inds = inputs["x_inds"]
        y_inds = inputs["u_inds"]
        F_ext = inputs["F_ext"]
        D₀ = inputs["D₀"]
        a = inputs["a"]
        ɛ_ext = inputs["ɛ_ext"]
    
        function γ()
            return inputs["γ⊥"]/(ω-inputs["ω₀"]+1im*inputs["γ⊥"])
        end

        ω²= ω^2
        dx²=dx^2
        dy²=dy^2

        r = whichRegion((x_ext,y_ext),∂_ext,inputs["geometry"])

        ∇² = laplacian(ω,inputs)

        ɛω² = sparse(1:Nₓ*Nᵤ,1:Nₓ*Nᵤ,ɛ_ext[r[:]]*ω²        ,Nₓ*Nᵤ,Nₓ*Nᵤ,+)
        χω² = sparse(1:Nₓ*Nᵤ,1:Nₓ*Nᵤ,F_ext[r[:]]*ω².*γ()*D₀,Nₓ*Nᵤ,Nₓ*Nᵤ,+)
        Ω²  = sparse(1:Nₓ*Nᵤ,1:Nₓ*Nᵤ,fill(ω²,Nₓ*Nᵤ),Nₓ*Nᵤ,Nₓ*Nᵤ,+)

        φ = inputs["incidentWave"](find(r.>9),ω,inputs)
        j = (∇²+Ω²)*φ

        if ψ_init == 0 || !isNonLinear
            ψ_ext = (∇²+ɛω²+χω²)\j
        else
            ψ_ext = ψ_init
        end

        if isNonLinear
            Ψ_init = Array(Float64,2*length(j))
            Ψ_init[1:length(j)]     = real(ψ_ext)
            Ψ_init[length(j)+1:2*length(j)] = imag(ψ_ext)

            function χ(Ψ)
                V = inputs["F_ext"][r[:]].*γ()*inputs["D₀"]./(1+abs2(Ψ))
                return sparse(1:Nₓ*Nᵤ,1:Nₓ*Nᵤ,V,Nₓ*Nᵤ,Nₓ*Nᵤ,+)
            end

            function χʳʳ′(Ψ)
                V = -2.*inputs["F_ext"][r[:]].*real(γ().*Ψ).*inputs["D₀"].*real(Ψ)./((1+abs2(Ψ)).^2)
                return sparse(1:Nₓ*Nᵤ,1:Nₓ*Nᵤ,V,Nₓ*Nᵤ,Nₓ*Nᵤ,+)
            end

            function χⁱʳ′(Ψ)
                V = -2.*inputs["F_ext"][r[:]].*imag(γ().*Ψ).*inputs["D₀"].*real(Ψ)./((1+abs2(Ψ)).^2)
                return sparse(1:Nₓ*Nᵤ,1:Nₓ*Nᵤ,V,Nₓ*Nᵤ,Nₓ*Nᵤ,+)
            end

            function χʳⁱ′(Ψ)
                V = -2.*inputs["F_ext"][r[:]].*real(γ().*Ψ).*inputs["D₀"].*imag(Ψ)./((1+abs2(Ψ)).^2)
                return sparse(1:Nₓ*Nᵤ,1:Nₓ*Nᵤ,V,Nₓ*Nᵤ,Nₓ*Nᵤ,+)
            end

            function χⁱⁱ′(Ψ)
                V = -2.*inputs["F_ext"][r[:]].*imag(γ().*Ψ).*inputs["D₀"].*imag(Ψ)./((1+abs2(Ψ)).^2)
                return sparse(1:Nₓ*Nᵤ,1:Nₓ*Nᵤ,V,Nₓ*Nᵤ,Nₓ*Nᵤ,+)
            end

            function f!(Ψ,fvec)
                ψ = similar(j,Complex128)
                ψ = Ψ[1:length(j)]+im*Ψ[length(j)+1:2*length(j)]
                temp = (∇²+ɛω²+χ(ψ)*ω²)*ψ - j
                fvec[1:length(j)]     = real(temp)
                fvec[length(j)+1:2*length(j)] = imag(temp)
            end

            function jac!(Ψ,jacarr)
                ψ = similar(j,Complex128)
                ψ = Ψ[1:length(j)]+im*Ψ[length(j)+1:2*length(j)]
                temp = ∇²+ɛω²+χ(ψ)*ω²
                tempr = similar(temp,Float64)
                tempi = similar(temp,Float64)
                tr = nonzeros(tempr)
                ti = nonzeros(tempi)
                tr[:] = real((nonzeros(temp)))
                ti[:] = imag((nonzeros(temp)))
                tempj = [tempr+χʳʳ′(ψ) -tempi+χʳⁱ′(ψ); tempi+χⁱʳ′(ψ) tempr+χⁱⁱ′(ψ)]
                jacarr[:,:] = tempj[:,:]
            end

            df = DifferentiableSparseMultivariateFunction(f!, jac!)
            z = nlsolve(df,Ψ_init,show_trace = false,ftol = 2e-8, iterations = 750)

            if converged(z)
                ψ_ext = z.zero[1:length(j)] + im*z.zero[length(j)+1:2*length(j)]
            else
                ψ_ext = NaN*ψ_ext;
            end

        end

        return ψ_ext,φ

    end



function solve_SPA(inputs,ω; z₀ = 0.0im)

        x = inputs["x_ext"][inputs["x_inds"]]
        y = inputs["u_ext"][inputs["u_inds"]]
        ∂_ext = inputs["∂_ext"]
        N_ext = inputs["N_ext"]

        D₀ = copy(inputs["D₀"])
        a = copy(inputs["a"])
        inputs["D₀"] = 0.
        inputs["a"] = 1.

        ϕ,dummy = solve_scattered(inputs,inputs["ω₀"],isNonLinear = false)
        X_inds = [i for i in inputs["x_inds"], j in inputs["u_inds"]]
        Y_inds = [j for i in inputs["x_inds"], j in inputs["u_inds"]]
        φ = ϕ[Y_inds[:]*N_ext[1] + X_inds[:]]

        inputs["D₀"] = copy(D₀)
        inputs["a"] = copy(a)

        η,u = computeCFS(inputs,ω,1);

        function γ()
            return inputs["γ⊥"]/(ω-inputs["ω₀"]+1im*inputs["γ⊥"])
        end

        r = whichRegion((x,y),∂_ext,inputs["geometry"])
        F = inputs["F_ext"][r[:]]

        function f!(z,fvec)

            b = inputs["a"]
            Z = z[1]+1im*z[2]

            numerator = u.*F.*(Z*u+b*φ)
            denominator = 1+abs2(Z*u+b*φ)
            term = (inputs["D₀"]*γ()./η).*trapz(numerator./denominator,inputs["dr"])

            fvec[1] = real(term[1]/Z - 1)
            fvec[2] = imag(term[1]/Z - 1)

            return term,Z

        end

        result = nlsolve(f!,[real(z₀),imag(z₀)])

        z = result.zero[1]+1im*result.zero[2]

        ψ = inputs["a"]*φ+z*u

        return ψ

    end




    function computeS(inputs;isNonLinear = false)

        x_inds = inputs["x_inds"]
        a = inputs["a"]
        ψ₊ = Array(Complex128,length(inputs["x_ext"]))
        ψ₋ = Array(Complex128,length(inputs["x_ext"]))
        S = Array(Complex128,2,2,length(inputs["ω"]))

        for ii in 1:length(inputs["ω"])

            ω = inputs["ω"][ii]

            if ii/10 == round(ii/10)
                printfmtln("Solving for frequency {1:d} of {2:d}, ω = {3:2.3f}.",ii,length(inputs["ω"]),ω)
            end
        
            if ii == 1 || isnan(ψ₊[1]) || isnan(ψ₋[1])
                if ii == 1 || isnan(ψ₊[1])
                    ψ₊ = scatteredFields(inputs,ω,inputs["xᵨ₊"],+,isNonLinear = isNonLinear)
                end
                if ii == 1 || isnan(ψ₋[1])
                    ψ₋ = scatteredFields(inputs,ω,inputs["xᵨ₋"],-,isNonLinear = isNonLinear)
                end
            else
                ψ₊ = scatteredFields(inputs,ω,inputs["xᵨ₊"],+,isNonLinear = isNonLinear, ψ_init = ψ₊)
                ψ₋ = scatteredFields(inputs,ω,inputs["xᵨ₋"],-,isNonLinear = isNonLinear, ψ_init = ψ₋)
            end

            S[:,1,ii] = ψ₊[x_inds[[1, end]]]/a
            S[:,2,ii] = ψ₋[x_inds[[1, end]]]/a

        end

        return(S)

    end



end