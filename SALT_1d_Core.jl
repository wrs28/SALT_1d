module Core

export laplacian, σ, whichRegion, trapz, dirac_δ, heaviside_Θ, processInputs, updateInputs, computeCFs_Core

#############################################################

function grad(N::Int,dx::Float64)

    I₁ = collect(1:N)
    J₁ = collect(1:N)
    V₁ = fill(Complex(-1/dx),N)

    I₂ = collect(1:N)
    J₂ = collect(2:(N+1))
    V₂ = fill(Complex(+1/dx),N)

    ∇ = sparse([I₁;I₂],[J₁;J₂],[V₁;V₂],N,N+1,+)

end # end of function grad


function laplacian(ℓ::Float64,N::Int,Σ)

    dx = ℓ/(N-1); dx² = dx^2
    ∇ = grad(N-1,dx)

    s₁ = sparse(1:N-1,1:N-1,2./( Σ[1:end-1] + Σ[2:end] ),N-1,N-1)
    s₂ = sparse(1:N,1:N,1./Σ,N,N)

    ∇² = -s₂*∇.'*s₁*∇

    ∇²[1,1]     += -2/dx²
    ∇²[end,end] += -2/dx²

    return ∇²

end # end of function laplacian


function σ(x,∂)

    r = whichRegion(x,∂)
    s = similar(r,Float64)

    for i in 1:length(r)
        if r[i] == 1
            s[i] = abs2(x[i]-∂[2])/abs2(∂[1]-∂[2])
        elseif r[i] == length(∂)-1
            s[i] = abs2(x[i]-∂[end-1])/abs2(∂[end]-∂[end-1])
        else
            s[i] = 0
        end
    end

    return s

end # end of function σ


function whichRegion(x,∂)

    region = similar(x,Int);

    for i in 1:length(region), k in 1:length(∂)-1

        if k+1 == length(∂)
            if  ∂[k] ≤ x[i]
                region[i] = k
            end
        elseif k == 1
            if  x[i] ≤ ∂[k+1]
                region[i] = k
            end
        else
            if  ∂[k] ≤ x[i] ≤ ∂[k+1]
                region[i] = k
            end
        end

    end

    return region

end 
# end of function whichRegionn


function trapz(z,dx::Float64)

    integral = dx*sum(z)-dx*(z[1]+z[end])/2

    return integral

end 
# end of function trapz


function dirac_δ(x,x₀::Float64)

    ind₁ = findmin(abs2(x-x₀))[2]
    ind₂ = ind₁ + (2*mod(findmin([findmin(abs2(x[ind₁+1]-x₀))[1] findmin(abs2(x[ind₁-1]-x₀))[1]])[2],2)[1] -1)
    min_ind = min(ind₁,ind₂)
    max_ind = max(ind₁,ind₂)


    x₁ = x[min_ind]
    x₂ = x[max_ind]
    dx = abs(x₂-x₁)
    dx² = dx^2

    w₁ = abs(x₂-x₀)./dx²
    w₂ = abs(x₀-x₁)./dx²

    δ = sparsevec([min_ind,max_ind],[w₁,w₂],length(x),+)

    δ1 = sparsevec([min_ind,max_ind]  ,[w₁,w₂],length(x)+1,+)
    δ2 = sparsevec([min_ind,max_ind]+1,[w₁,w₂],length(x)+1,+)
    ∇ = grad(length(x),dx)

    ∇δ = ∇*(δ1+δ2)/2

    return δ,∇δ

end 
# end of function dirac_δ


function heaviside_Θ(x,x₀::Float64)

    ind₁ = findmin(abs2(x-x₀))[2]
    ind₂ = ind₁ + (2*mod(findmin([findmin(abs2(x[ind₁+1]-x₀))[1] findmin(abs2(x[ind₁-1]-x₀))[1]])[2],2)[1] -1)
    min_ind = min(ind₁,ind₂)
    max_ind = max(ind₁,ind₂)

    x₁ = x[min_ind]
    x₂ = x[max_ind]
    dx = x₂-x₁

    Θ = zeros(length(x),1)
    Θ[x .≥ x₀,1] = 1.
    w₁ = (x₂-x₀)./dx
    w₂ = (x₀-x₁)./dx
    Θ[min_ind,1] = w₁
    Θ[max_ind,1] = w₂

    ∇Θ  = zeros(length(x),1)
    ∇Θ[1,1]    = (Θ[2,1]-Θ[1,1])/dx
    ∇Θ[end,1]  = (Θ[end,1]-Θ[end-1,1])/dx
    ∇Θ[2:end-1,1]  = (Θ[3:end,1]-Θ[1:end-2,1])/2dx

    ∇²Θ = zeros(length(x),1)
    ∇²Θ[2:end-1,1] = (Θ[1:end-2,1] - 2Θ[2:end-1,1] + Θ[3:end,1])/dx^2

    return (Θ,∇Θ,∇²Θ)

end
# end of function heaviside_Θ



function processInputs()

    (N, λ₀, λ, ∂, Γ, F, ɛ, γ⟂, D₀, a) = evalfile("SALT_1d_Inputs.jl")

    ω₀ = 2π./λ₀
    ω  = 2π./λ
    k  = ω
    k₀ = ω₀
    ℓ = ∂[end] - ∂[1]

    xᵨ₊ = (∂[1]+∂[2])/2
    xᵨ₋ = (∂[end-1]+∂[end])/2
    
    ##########################

    dx = ℓ/(N-1);
    n = 1/dx
    dN1 = ceil(Integer,2.5*λ₀*n)
    dN2 = ceil(Integer,0.25*λ₀*n)
    N_ext = N + 2(dN1+dN2)
    ℓ_ext = dx*(N_ext-1)

#    x_ext = linspace(-(dN1+dN2)*dx+∂[1], (dN1+dN2)*dx+∂[end], N_ext)
    x_ext = vcat(-[(dN1+dN2):-1:1;]*dx+∂[1],  linspace(∂[1],∂[end],N), [1:(dN1+dN2);]*dx+∂[end])
    x_inds = dN1+dN2+collect(1:N)
    x = x_ext[x_inds]

    ∂_ext = [x_ext[1]-dx/2 x_ext[dN1+1] ∂ x_ext[dN1+dN2+N+dN2+1] x_ext[end]+dx/2]

    F_min = 1e-15
    F[F .== zero(Float64)] = F_min
    F_ext = [F_min F_min F F_min F_min]

    ɛ_ext = [1 1 ɛ 1 1]
    Γ_ext = [Inf Inf  Γ Inf Inf]

    inputs = Dict{Any,Any}(
        "λ" => λ,
        "λ₀" => λ₀,
        "ω" => ω,
        "ω₀" => ω₀,
        "k" => k,
        "k₀" => k₀,
        "N" => N,
        "ℓ" => ℓ,
        "dx" => dx,
        "x_ext" => x_ext,
        "x_inds" => x_inds,
        "∂" => ∂,
        "ɛ" => ɛ,
        "F" => F,
        "N_ext" => N_ext,
        "ℓ_ext" => ℓ_ext,
        "∂_ext" => ∂_ext,
        "ɛ_ext" => ɛ_ext,
        "F_ext" => F_ext,
        "x" => x,
        "xᵨ₋" => xᵨ₋,
        "xᵨ₊" => xᵨ₊,
        "γ⟂" => γ⟂,
        "D₀" => D₀,
        "a" => a,
        "Γ" => Γ,
        "Γ_ext" => Γ_ext,
        "dN1" => dN1,
        "dN2" => dN2)

    return(inputs)

end 
# end of function processInputs



function updateInputs(inputs::Dict)

    ∂ = inputs["∂"]
    N = inputs["N"]
    λ₀ = inputs["λ₀"]
    λ = inputs["λ"]
    F = inputs["F"]
    ɛ = inputs["ɛ"]
    Γ = inputs["Γ"]
    
    xᵨ₊ = (∂[1]+∂[2])/2
    xᵨ₋ = (∂[end-1]+∂[end])/2
    
    ω₀ = 2π./λ₀
    ω = 2π./λ
    k = ω
    k₀ = ω₀
    ℓ = ∂[end] - ∂[1]

    ##########################

    dx = ℓ/(N-1);
    n = 1/dx
    dN1 = ceil(Integer,2.5*λ₀*n)
    dN2 = ceil(Integer,0.25*λ₀*n)
    N_ext = N + 2(dN1+dN2)
    ℓ_ext = dx*(N_ext-1)

    x_ext = vcat(-[(dN1+dN2):-1:1;]*dx+∂[1],  linspace(∂[1],∂[end],N), [1:(dN1+dN2);]*dx+∂[end])
    x_inds = dN1+dN2+collect(1:N)
    x = x_ext[x_inds]

    ∂_ext = [x_ext[1]-dx/2 x_ext[dN1+1] ∂ x_ext[dN1+dN2+N+dN2+1] x_ext[end]+dx/2]

    F_min = 1e-15
    F[F .== zero(Float64)] = F_min
    F_ext = [F_min F_min F F_min F_min]

    ɛ_ext = [1 1 ɛ 1 1]
    Γ_ext = [Inf Inf  Γ Inf Inf]

    inputsNew = Dict{Any,Any}(
        "λ" => λ,
        "λ₀" => λ₀,
        "ω" => ω,
        "ω₀" => ω₀,
        "k" => k,
        "k₀" => k₀,
        "N" => N,
        "ℓ" => ℓ,
        "dx" => dx,
        "x_ext" => x_ext,
        "x_inds" => x_inds,
        "∂" => ∂,
        "ɛ" => ɛ,
        "F" => F,
        "N_ext" => N_ext,
        "ℓ_ext" => ℓ_ext,
        "∂_ext" => ∂_ext,
        "ɛ_ext" => ɛ_ext,
        "F_ext" => F_ext,
        "x" => x,
        "xᵨ₋" => xᵨ₋,
        "xᵨ₊" => xᵨ₊,
        "γ⟂" => inputs["γ⟂"],
        "D₀" => inputs["D₀"],
        "a" => inputs["a"],
        "Γ" => Γ,
        "Γ_ext" => Γ_ext,
        "dN1" => dN1,
        "dN2" => dN2)

    return(inputsNew)

end 
# end of function updateInputs




function computeCFs_Core(inputs::Dict, k::Number, nTCFs::Int; F=1., η_init = [], ψ_init = [])
# No Line Pulling, calculation independent of pump D

    ## definitions block
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
    ##
    
    k²= k^2

    r = whichRegion(x_ext,∂_ext)

    ∇² = laplacian(ℓ_ext, N_ext, 1+1im*σ(x_ext,∂_ext)/k)

    Γ = zeros(N_ext,1)
    for dd in 2:length(∂_ext)-1
        δ,dummy1 = dirac_δ(x_ext,∂_ext[dd])
        Γ = Γ[:] + full(δ)/Γ_ext[dd]
    end

    ɛk² = sparse(1:N_ext, 1:N_ext, ɛ_ext[r]*k²   ,N_ext, N_ext, +)
    Γk² = sparse(1:N_ext, 1:N_ext, Γ[:]*k²       , N_ext, N_ext, +)
    sF = sparse(1:N_ext, 1:N_ext, sign(F.*F_ext[r]), N_ext, N_ext, +)
    FF = sparse(1:N_ext, 1:N_ext, abs(F.*F_ext[r]), N_ext, N_ext, +)

    if isempty(η_init) & isempty(ψ_init)
        (η,ψ_ext,nconv,niter,nmult,resid) = eigs(-sF*(∇²+ɛk²+Γk²)/k²,FF, which = :LM, nev = nTCFs, sigma = 1e-8)
    elseif !isempty(η_init) & isempty(ψ_init)
        (η,ψ_ext,nconv,niter,nmult,resid) = eigs(-sF*(∇²+ɛk²+Γk²)/k²,FF, which = :LM, nev = nTCFs, sigma = η_init)
    elseif !isempty(η_init) & !isempty(ψ_init)
        (η,ψ_ext,nconv,niter,nmult,resid) = eigs(-sF*(∇²+ɛk²+Γk²)/k²,FF, which = :LM, nev = nTCFs, sigma = η_init, v0 = ψ_init)
    end

    ψ = zeros(Complex128,length(inputs["x_ext"]),nTCFs)

    for ii = 1:length(η)
        N = trapz(ψ_ext[:,ii].*F.*F_ext[r].*ψ_ext[:,ii],dx)
        ψ_ext[:,ii] = ψ_ext[:,ii]/sqrt(N)
        ψ[:,ii] = ψ_ext[:,ii]
    end

    return η,ψ

end
#end of function computeCFs_Core

#######################################################################

end
# end of Module Core