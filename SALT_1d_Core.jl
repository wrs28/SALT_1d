module Core

export laplacian, whichRegion, trapz, dirac_δ, heaviside_Θ, subpixelSmoothing, processInputs, updateInputs

#############################################################

PML_extinction = 1e6
PML_ρ = 1/6
PML_power_law = 3
F_min = 1e-16
α_imag = -.15
subPixelNum = Int(1e2)

###################################################


"""
∇ =  grad(N, dx)

Gradient with N points, lattice spacing dx.

sparse ∇[N,N+1]

"""
function grad(N::Int, dx::Float64)

    I₁ = collect(1:N)
    J₁ = collect(1:N)
    V₁ = fill(Complex(-1/dx), N)

    I₂ = collect(1:N)
    J₂ = collect(2:(N+1))
    V₂ = fill(Complex(+1/dx), N)

    ∇ = sparse( [I₁;I₂], [J₁;J₂], [V₁;V₂], N, N+1, +)

end # end of function grad



"""
∇² =  laplacian(k, inputs)

Laplacian evaluated at k∈ℂ. Lattice size & spacing, bc's determined in inputs.

sparse ∇²[# lattice pts, # lattice pts]

"""
function laplacian(k::Number, inputs::Dict)

    # definitions block#
    N = inputs["N_ext"]
    bc = inputs["bc"]
    a = inputs["a"]
    b = inputs["b"]
    dx = inputs["dx"]
    dx² = dx^2
    ##
    
    # generate robin parameter
    λ = zeros(Complex128,2)
    for i in 1:2
        λ[i] = -(-1)^i*1im*k*(b[i]-a[i])/(b[i]+a[i])
    end
    
    ∇ = grad(N-1,dx)

    # create PML layer
    Σ = 1.+1im*σ(inputs)./real(k)
    s₁ = sparse( 1:N-1, 1:N-1, 2./( Σ[1:end-1] .+ Σ[2:end] ), N-1, N-1)
    s₂ = sparse( 1:N, 1:N, 1./Σ, N, N)
    ∇² = -s₂*∇.'*s₁*∇

    # truncate
    ind = [1 N 1]
    for i in 1:2
        if bc[i] in ["pml_out" "pml_in"] 
            ∇²[ind[i],ind[i]]   += -2/dx²
        elseif bc[i] in ["d" "dirichlet" "Dirichlet" "hard"]
            ∇²[ind[i],ind[i]]   += -2/dx²
        elseif bc[i] in ["n" "neumann" "Neumann" "soft"]
            ∇²[ind[i],ind[i]]   += 0
        elseif bc[i] in ["p" "periodic"]
            ∇²[ind[i],ind[i]]   += -1/dx²
            ∇²[ind[i],ind[i+1]] += +1/dx²
        elseif bc[i] in ["o" "out" "outgoing"]
            ∇²[ind[i],ind[i]]   += +(1im*k/dx)/(1-1im*dx*k/2)
        elseif bc[i] in ["i" "in" "incoming" "incident"]
            ∇²[ind[i],ind[i]]   += -(1im*k/dx)/(1+1im*dx*k/2)
        elseif bc[i] in ["r" "robin"]
            ∇²[ind[i],ind[i]]   += -(-1)^i*(1/dx)*(λ[i]/(1+(-1)^i*λ[i]*dx/2))
        else
            println("error in bc specification, not valid")
            return
        end
    end
    
    return ∇²

end # end of function laplacian



"""
Σ =  σ(inputs)

Conductivity for PML layer.

"""
function σ(inputs::Dict)

    x = inputs["x_ext"]
    dx = inputs["dx"]
    ∂ = inputs["∂_ext"]

    α = zeros(Complex128,2)
    for i in 1:2
        if inputs["bc"][i] == "pml_out"
            α[i] = +(1+α_imag*1im)*( (PML_ρ/dx)^(PML_power_law+1) )/ ( (PML_power_law+1)*log(PML_extinction) )^PML_power_law
        elseif inputs["bc"][i] == "pml_in"
            α[i] = -(1+α_imag*1im)*( (PML_ρ/dx)^(PML_power_law+1) )/ ( (PML_power_law+1)*log(PML_extinction) )^PML_power_law
        else
            α[i] = 0
        end
    end

    s = zeros(Complex128,length(x))

    for i in 1:length(x)
        if x[i] ≤ ∂[2]
            s[i] = α[1]*abs(x[i]-∂[2])^PML_power_law
        elseif x[i] ≥ ∂[end-1]
            s[i] = α[2]*abs(x[i]-∂[end-1])^PML_power_law
        else
            s[i] = 0
        end
    end

    return s

end # end of function σ



"""
region = whichRegion(x, ∂)

Takes vector x, gives vector of regions as defined in input file.

"""
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

end # end of function whichRegion



"""
∫z_dx = trapz(z, dx)

Trapezoidal sum of z.

"""
function trapz(z,dx::Float64)

    ∫z_dx = dx.*sum(z)-dx*(z[1]+z[end])/2

    return ∫z_dx

end # end of function trapz



"""
δ, ∇δ = dirac_δ(x, x₀)

δ sparse, dirac distribution weighted to be centered at x₀.
∇δ sparse, derivative of δ.

"""
function dirac_δ(x,x₀::Float64)

    ind₁ = findmin(abs2.(x-x₀))[2]
    ind₂ = ind₁ + (2*mod(findmin([findmin(abs2.(x[ind₁+1]-x₀))[1] findmin(abs2.(x[ind₁-1]-x₀))[1]])[2],2)[1] -1)
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

    ∇δ = ∇*(δ1.+δ2)/2

    return δ,∇δ

end # end of function dirac_δ



"""
Θ,∇Θ,∇²Θ = heaviside_Θ(x, x₀)

Θ,∇Θ,∇²Θ not sparse, weighted to be centered at x₀.
"""
function heaviside_Θ(x,x₀::Float64)

    ind₁ = findmin(abs2.(x-x₀))[2]
    ind₂ = ind₁ + (2*mod(findmin([findmin(abs2.(x[ind₁+1]-x₀))[1] findmin(abs2.(x[ind₁-1]-x₀))[1]])[2],2)[1] -1)
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

end # end of function heaviside_Θ



"""
ɛ_smoothed, F_smoothed = subpixelSmoothing(inputs; truncate = false, r = [])

Sub-pixel smoothed ɛ & F.

if truncate=true, then output is truncated to bulk region (sans PML)

r is the output of whichRegion. If computing r elsewhere, can save time by using that output here.

"""
function subpixelSmoothing(inputs; truncate = false, r = [])
    # for now it's for ɛ and F, could feasibly be extended eventually...
    
    if truncate
        x = inputs["x"]
    else
        x = inputs["x_ext"]
    end
    
    if isempty(r)
        r = whichRegion(x, inputs["∂_ext"]) 
    end

    ɛ = inputs["ɛ_ext"]
    F = inputs["F_ext"]
    ɛ_smoothed = deepcopy(ɛ[r])
    F_smoothed = deepcopy(F[r])
    
    for i in 2:(length(x)-1)
        
        nearestNeighborFlag = (r[i]!==r[i+1]) | (r[i]!==r[i-1])
        
        if nearestNeighborFlag
            
            x_min = (x[i]+x[i-1])/2
            x_max = (x[i]+x[i+1])/2
            
            X = linspace(x_min,x_max,inputs["subPixelNum"])
            
            subRegion = whichRegion(X, inputs["∂_ext"])
            ɛ_smoothed[i] = mean(ɛ[subRegion])
            F_smoothed[i] = mean(F[subRegion])
            
        end
        
    end
    
    return ɛ_smoothed, F_smoothed
    
end # end of function subpixelSmoothing



"""
processInputs(fileName = "./SALT_1d_Inputs.jl")


"""
function processInputs(fileName = "./SALT_1d_Inputs.jl")

    (N, k₀, k, bc, ∂, Γ, F, ɛ, γ⟂, D₀, a, b, subPixelNum) = evalfile(fileName)

    ω₀ = k₀
    ω  = k
    ℓ = ∂[end] - ∂[1]

    xᵨ₊ = (∂[1]+∂[2])/2
    xᵨ₋ = (∂[end-1]+∂[end])/2
    
    ##########################

    dx = ℓ/(N-1);
    n = 1/dx

    dN = [0 0]
    for i in 1:2
        if bc[i] in ["o" "out" "outgoing"]
            dN[i] = 0
        elseif bc[i] in ["i" "in" "incoming" "incident"]
            dN[i] = 0
        elseif bc[i] in ["d" "dirichlet" "Dirichlet" "hard"]
            dN[i] = 0
        elseif bc[i] in ["n" "neumann"   "Neumann"   "soft"]
            dN[i] = 0
        elseif bc[i] in ["p" "periodic"]
            dN[i] = 0
        elseif bc[i] in ["pml_out" "pml_in"]
            dN[i] = ceil(Int,(PML_power_law+1)*log(PML_extinction)/PML_ρ)
        elseif bc[i] in ["r" "robin"]
            dN[i] = 0
        else
            println("error in bc specification, not valid")
            return
        end
    end

    N_ext = N + sum(dN)
    ℓ_ext = dx*(N_ext-1)

    x_ext = vcat(-[dN[1]:-1:1;]*dx+∂[1],  linspace(∂[1],∂[end],N), [1:dN[2];]*dx+∂[end])
    x_inds = dN[1]+collect(1:N)
    x = x_ext[x_inds]

    ∂_ext = [x_ext[1]-dx/2 ∂ x_ext[end]+dx/2]

    F[F .== zero(Float64)] = F_min
    F_ext = [F_min F F_min]

    ɛ_ext = [1 ɛ 1]
    Γ_ext = [Inf Γ Inf]

    inputs1 = Dict{String,Any}(
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
        "b" => b,
        "Γ" => Γ,
        "Γ_ext" => Γ_ext,
        "dN" => dN,
        "bc" => bc,
        "subPixelNum" => subPixelNum)

    r_ext = whichRegion(x_ext,∂_ext)
    ɛ_sm, F_sm = subpixelSmoothing(inputs1; truncate = false, r = [])
    
    inputs2 = Dict{String,Any}(
        "r_ext" => r_ext,
        "ɛ_sm" => ɛ_sm,
        "F_sm" => F_sm
        )
        
    return(merge(inputs1,inputs2))
    
end  # end of function processInputs





"""
inputs = updateInputs(inputs)

If changes were made to the ∂, N, k₀, k, F, ɛ, Γ, bc, a, b, run updateInputs to propagate these changes through the system.

"""
function updateInputs(inputs::Dict)

    ∂ = inputs["∂"]
    N = inputs["N"]
    k₀ = inputs["k₀"]
    k = inputs["k"]
    F = inputs["F"]
    ɛ = inputs["ɛ"]
    Γ = inputs["Γ"]
    bc = inputs["bc"]
    a = inputs["a"]
    b = inputs["b"]
    
    xᵨ₊ = (∂[1]+∂[2])/2
    xᵨ₋ = (∂[end-1]+∂[end])/2
    
    ω₀ = k₀
    ω = k

    ℓ = ∂[end] - ∂[1]

    ##########################

    dx = ℓ/(N-1);
    n = 1/dx

    dN = [0 0]
    for i in 1:2
        if bc[i] in ["o" "out" "outgoing"]
            dN[i] = 0
        elseif bc[i] in ["i" "in" "incoming" "incident"]
            dN[i] = 0
        elseif bc[i] in ["d" "dirichlet" "Dirichlet" "hard"]
            dN[i] = 0
        elseif bc[i] in ["n" "neumann"   "Neumann"   "soft"]
            dN[i] = 0
        elseif bc[i] in ["p" "periodic"]
            dN[i] = 0
        elseif bc[i] in ["pml_out" "pml_in"]
            dN[i] = ceil(Int,(PML_power_law+1)*log(PML_extinction)/PML_ρ)
        elseif bc[i] in ["r" "robin"]
            dN[i] = 0
        else
            println("error in bc specification, not valid")
            return
        end
    end

    N_ext = N + sum(dN)
    ℓ_ext = dx*(N_ext-1)

    x_ext = vcat(-[dN[1]:-1:1;]*dx+∂[1],  linspace(∂[1],∂[end],N), [1:dN[2];]*dx+∂[end])
    x_inds = dN[1]+collect(1:N)
    x = x_ext[x_inds]

    ∂_ext = [x_ext[1]-dx/2 ∂ x_ext[end]+dx/2]

    F[F .== zero(Float64)] = F_min
    F_ext = [F_min F F_min]

    ɛ_ext = [1 ɛ 1]
    Γ_ext = [Inf Γ Inf]

    inputsNew1 = Dict{String,Any}(
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
        "b" => inputs["b"],
        "Γ" => Γ,
        "Γ_ext" => Γ_ext,
        "dN" => dN,
        "bc" => bc,
        "subPixelNum" => inputs["subPixelNum"])

    r_ext = whichRegion(x_ext,∂_ext)
    ɛ_sm, F_sm = subpixelSmoothing(inputsNew1; truncate = false, r = [])
    
    inputsNew2 = Dict{String,Any}(
        "r_ext" => r_ext,
        "ɛ_sm" => ɛ_sm,
        "F_sm" => F_sm
        )
        
    return(merge(inputsNew1,inputsNew2))

end # end of function updateInputs

#######################################################################

end # end of Module Core