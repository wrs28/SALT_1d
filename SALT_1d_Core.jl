module Core

export InputStruct, laplacian, whichRegion, trapz, dirac_δ, heaviside_Θ, subpixelSmoothing, processInputs, updateInputs!

#############################################################


"""
inputs = Inputs(ω, ω₀, k, k₀, N, ℓ, dx, x_ext, x_inds, ∂, ɛ, F, N_ext, ℓ_ext, ∂_ext, ɛ_ext, F_ext, x, xᵨ₋, xᵨ₊, γ⟂, D₀, a, b, Γ, dN, bc, subPixelNum, r_ext, ɛ_sm, F_sm)

"""
mutable struct InputStruct
    ω::Array{Complex128,1}
    ω₀::Complex128
    k::Array{Complex128,1}
    k₀::Complex128
    N::Int
    ℓ::Float64
    dx::Float64
    x_ext::Array{Float64,1}
    x_inds::Array{Int,1}
    ∂::Array{Float64,1}
    ɛ::Array{Complex128,1}
    F::Array{Float64,1}
    N_ext::Int
    ℓ_ext::Float64
    ∂_ext::Array{Float64,1}
    ɛ_ext::Array{Complex128,1}
    F_ext::Array{Float64,1}
    x::Array{Float64,1}
    xᵨ₋::Float64
    xᵨ₊::Float64
    γ⟂::Float64
    D₀::Float64
    a::Array{Complex128,1}
    b::Array{Complex128,1}
    Γ::Array{Complex128,1}
    Γ_ext::Array{Complex128,1}
    dN::Array{Int,1}
    bc::Array{String,1}
    subPixelNum::Int
    r_ext::Array{Int,1}
    ɛ_sm::Array{Complex128,1}
    F_sm::Array{Float64,1}
end



"""
∇ =  grad(N, dx)

Gradient with N points, lattice spacing dx. It's the forward gradient (I think).

sparse ∇[N,N+1]

"""
function grad(N::Int, dx::Float64)

    I₁ = Array(1:N)
    J₁ = Array(1:N)
    V₁ = fill(Complex(-1/dx), N)

    I₂ = Array(1:N)
    J₂ = Array(2:(N+1))
    V₂ = fill(Complex(+1/dx), N)

    ∇ = sparse( vcat(I₁,I₂), vcat(J₁,J₂), vcat(V₁,V₂), N, N+1, +)

end # end of function grad




"""
∇² =  laplacian(k, inputs)

Laplacian evaluated at k∈ℂ. Lattice size & spacing, bc's determined in inputs.

sparse ∇²[# lattice pts, # lattice pts]

"""
function laplacian(k::Complex128, inputs::InputStruct)

    # definitions block#
    N = inputs.N_ext
    bc = inputs.bc
    a = inputs.a
    b = inputs.b
    dx = inputs.dx
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
    ∇² = -s₂*transpose(∇)*s₁*∇

    # truncate
    ind = [1, N, 1]
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
function σ(inputs::InputStruct)

    ####################
    PML_extinction = 1e6
    PML_ρ = 1/6
    PML_power_law = 3
    α_imag = -.15
    ####################

    x = inputs.x_ext
    dx = inputs.dx
    N = inputs.N_ext
    ∂ = inputs.∂_ext

    α = zeros(Complex128,2)
    for i in 1:2
        if inputs.bc[i] == "pml_out"
            α[i] = +(1+α_imag*1im)*( (PML_ρ/dx)^(PML_power_law+1) )/ ( (PML_power_law+1)*log(PML_extinction) )^PML_power_law
        elseif inputs.bc[i] == "pml_in"
            α[i] = -(1+α_imag*1im)*( (PML_ρ/dx)^(PML_power_law+1) )/ ( (PML_power_law+1)*log(PML_extinction) )^PML_power_law
        else
            α[i] = 0
        end
    end

    s = zeros(Complex128,N)

    for i in 1:N
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
function whichRegion(x::Array{Float64,1}, ∂::Array{Float64,1})

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
function trapz(z::Array{Complex128,1},dx::Float64)::Complex128

    ∫z_dx = dx*sum(z)-dx*(z[1]+z[end])/2

    return ∫z_dx

end # end of function trapz



"""
δ, ∇δ = dirac_δ(x, x₀)

δ sparse, dirac distribution weighted to be centered at x₀.
∇δ sparse, derivative of δ.

"""
function dirac_δ(x::Array{Float64,1},x₀::Float64)

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
function heaviside_Θ(x::Array{Float64,1},x₀::Float64)

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
function subpixelSmoothing(inputs::InputStruct; truncate::Bool = false, r::Array{Int,1} = Int[])
    # for now it's for ɛ and F, could feasibly be extended eventually...

    X = inputs.x
    X_ext = inputs.x_ext
    ∂ = inputs.∂_ext
    ɛ = inputs.ɛ_ext
    F = inputs.F_ext
    subPixelNum = inputs.subPixelNum

    ɛ_smoothed, F_smoothed = subpixelSmoothing_core(X, X_ext, ∂, ɛ, F, subPixelNum, truncate, r)

    return ɛ_smoothed, F_smoothed

end # end of function subpixelSmoothing




"""
ɛ_smoothed, F_smoothed = subpixelSmoothing_core(inputs; truncate = false, r = [])

Sub-pixel smoothed ɛ & F.

heart of subpixelSmoothing routine, makes it usable in processInputs
"""
function subpixelSmoothing_core(X::Array{Float64,1}, X_ext::Array{Float64,1}, ∂::Array{Float64,1}, ɛ::Array{Complex128,1}, F::Array{Float64,1}, subPixelNum::Int, truncate::Bool, r::Array{Int,1})
    # for now it's for ɛ and F, could feasibly be extended eventually...

    if truncate
        x = X
    else
        x = X_ext
    end

    if isempty(r)
        r = whichRegion(x, ∂)
    end

    ɛ_smoothed = deepcopy(ɛ[r])
    F_smoothed = deepcopy(F[r])

    for i in 2:(length(x)-1)

        nearestNeighborFlag = (r[i]!==r[i+1]) | (r[i]!==r[i-1])

        if nearestNeighborFlag

            x_min = (x[i]+x[i-1])/2
            x_max = (x[i]+x[i+1])/2

            X = Array(linspace(x_min, x_max, subPixelNum))

            subRegion = whichRegion(X, ∂)
            ɛ_smoothed[i] = mean(ɛ[subRegion])
            F_smoothed[i] = mean(F[subRegion])

        end

    end

    return ɛ_smoothed, F_smoothed

end # end of function subpixelSmoothing_core





"""
processInputs(fileName = "./SALT_1d_Inputs.jl")


"""
function processInputs(fileName::String = "./SALT_1d_Inputs.jl")::InputStruct

    #################### See also definition block in sigma function
    F_min = 1e-16
    PML_extinction = 1e6
    PML_ρ = 1/6
    PML_power_law = 3
    ####################

    (N::Int, k₀::Complex128, k::Array{Complex128,1}, bc::Array{String,1}, ∂::Array{Float64,1}, Γ::Array{Complex128,1}, F::Array{Float64,1}, ɛ::Array{Complex128,1}, γ⟂::Float64, D₀::Float64, a::Array{Complex128,1}, b::Array{Complex128,1}, subPixelNum::Int) = evalfile(fileName)

    ω₀ = k₀
    ω  = k
    ℓ = ∂[end] - ∂[1]

    xᵨ₊ = (∂[1]+∂[2])/2
    xᵨ₋ = (∂[end-1]+∂[end])/2

    ##########################

    dx = ℓ/(N-1);

    dN = Int[0, 0]
    for i in 1:2
        if bc[i] in ["o", "out", "outgoing"]
            dN[i] = 0
        elseif bc[i] in ["i", "in", "incoming", "incident"]
            dN[i] = 0
        elseif bc[i] in ["d", "dirichlet", "Dirichlet", "hard"]
            dN[i] = 0
        elseif bc[i] in ["n", "neumann",   "Neumann",   "soft"]
            dN[i] = 0
        elseif bc[i] in ["p", "periodic"]
            dN[i] = 0
        elseif bc[i] in ["pml_out", "pml_in"]
            dN[i] = ceil(Int,(PML_power_law+1)*log(PML_extinction)/PML_ρ)
        elseif bc[i] in ["r", "robin"]
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

    ∂_ext = vcat([x_ext[1]-dx/2], ∂, [x_ext[end]+dx/2])

    F[iszero.(F)] = F_min
    F_ext = vcat([F_min], F, [F_min])

    ɛ_ext = vcat([1], ɛ, [1])
    Γ_ext = vcat([Inf], Γ, [Inf])

    r_ext = whichRegion(x_ext, ∂_ext)
    ɛ_sm, F_sm = subpixelSmoothing_core(x, x_ext, ∂_ext, ɛ_ext, F_ext, subPixelNum, false, r_ext)

    inputs = InputStruct(ω, ω₀, k, k₀, N, ℓ, dx, x_ext, x_inds, ∂, ɛ, F, N_ext, ℓ_ext, ∂_ext, ɛ_ext, F_ext, x, xᵨ₋, xᵨ₊, γ⟂, D₀, a, b, Γ, Γ_ext, dN, bc, subPixelNum, r_ext, ɛ_sm, F_sm)


end  # end of function processInputs




"""
inputs = updateInputs(inputs)

If changes were made to the ∂, N, k₀, k, F, ɛ, Γ, bc, a, b, run updateInputs to propagate these changes through the system.

"""
function updateInputs!(inputs::InputStruct, field::Symbol, value::Any)

    #################### See also definition block in sigma function
    F_min = 1e-16
    PML_extinction = 1e6
    PML_ρ = 1/6
    PML_power_law = 3
    α_imag = -.15
    ####################

    fields = fieldnames(inputs)
    setfield!(inputs,field,value)

    if field == :ω₀
        inputs.k₀ = inputs.ω₀
    elseif field == :k₀
        inputs.ω₀ = inputs.k₀
    elseif field == :ω
        inputs.k = inputs.ω
    elseif field == :k
        inputs.ω = inputs.k
    end

    if field in [:∂, :N, :bc]
        ∂ = inputs.∂
        N = inputs.N
        bc = inputs.bc

        xᵨ₊ = (∂[1]+∂[2])/2
        xᵨ₋ = (∂[end-1]+∂[end])/2

        ℓ = ∂[end] - ∂[1]

        ##########################

        dx = ℓ/(N-1);

        dN = Int[0, 0]
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
            elseif bc[i] in ["CW" "cw" "clockwise" "Clockwise" "clock" "Clock"]
                dN[i] = 0
            elseif bc[i] in ["CCW" "ccw" "counterclockwise" "CounterClockwise" "counter" "Counter"]
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

        ∂_ext = vcat([x_ext[1]-dx/2], ∂, [x_ext[end]+dx/2])

        inputs.xᵨ₊ = xᵨ₊
        inputs.xᵨ₋ = xᵨ₋
        inputs.ℓ = ℓ
        inputs.dx = dx
        inputs.dN = dN
        inputs.N_ext = N_ext
        inputs.ℓ_ext = ℓ_ext
        inputs.x_ext = x_ext
        inputs.x_inds = x_inds
        inputs.x = x
        inputs.∂_ext = ∂_ext

    end

    if  field in [:∂, :N, :bc, :F, :ɛ, :Γ, :subPixelNum]

        F = inputs.F
        ɛ = inputs.ɛ
        Γ = inputs.Γ
        subPixelNum = inputs.subPixelNum

        F[iszero.(F)] = F_min
        F_ext = vcat([F_min], F, [F_min])

        ɛ_ext = vcat([1], ɛ, [1])
        Γ_ext = vcat([Inf], Γ, [Inf])

        r_ext = whichRegion(inputs.x_ext, inputs.∂_ext)
        ɛ_sm, F_sm = subpixelSmoothing_core(inputs.x, inputs.x_ext, inputs.∂_ext, ɛ_ext, F_ext, inputs.subPixelNum, false, r_ext)

        inputs.F_ext = F_ext
        inputs.Γ_ext = Γ_ext
        inputs.r_ext = r_ext
        inputs.ɛ_sm = ɛ_sm
        inputs.F_sm = F_sm

    end

    return inputs

end # end of function updateInputs

#######################################################################

end # end of Module Core
