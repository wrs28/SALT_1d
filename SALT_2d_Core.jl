module Core

### PARAMETER BLOCK ###
PML_extinction = 1e2
PML_ρ = 1/3
PML_power_law = 2
F_min = 1e-15
dNₓ2 = 5
dNᵤ2 = 5
#######################


export laplacian, whichRegion, trapz, processInputs, updateInputs

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

function grad_1d(N::Int,dx::Float64)

    I₁ = collect(1:N)
    J₁ = collect(1:N)
    V₁ = fill(Complex(-1/dx),N)

    I₂ = collect(1:N)
    J₂ = collect(2:(N+1))
    V₂ = fill(Complex(+1/dx),N)

    ∇ = sparse([I₁;I₂],[J₁;J₂],[V₁;V₂],N,N+1,+)

end #end of function grad_1d



function scrambleX(M::SparseMatrixCSC{Complex{Float64},Int},N::Int)

    m, n = size(M)

    row = zeros(Int,N*nnz(M))
    col = zeros(Int,N*nnz(M))
    val = zeros(Complex128,N*nnz(M))

    rows = rowvals(M)
    vals = nonzeros(M)

    count = Int(1)

    for k in 1:N
        for i in 1:n
            for j in nzrange(M, i)
                row[count] = m*(k-1)+rows[j]
                col[count] = n*(k-1)+i
                val[count] = vals[j]
                count += 1
            end
        end
    end

    Mx = sparse(row,col,val,m*N,n*N)

end #end of function scrambleX


function scrambleY(M::SparseMatrixCSC{Complex{Float64},Int},N::Int)

    m, n = size(M)

    row = zeros(Int,nnz(M)*N)
    col = zeros(Int,nnz(M)*N)
    val = zeros(Complex128,nnz(M)*N)

    rows = rowvals(M)
    vals = nonzeros(M)

    count = Int(1)

    for k in 1:N
        for i in 1:n
            for j in nzrange(M, i)
                row[count] = N*(rows[j]-1)+k
                col[count] = N*(i-1)+k
                val[count] = vals[j]
                count += 1
            end
        end
    end

    My = sparse(row,col,val,N*m,N*n)

end # end of function scrambleY


function grad(N::Array{Int},dr::Array{Float64}; isScrambled=false)

    Nₓ = N[1]
    dx = dr[1]

    Nᵤ = N[2]
    du = dr[2]

    ∇ₓ = grad_1d(Nₓ-1,dx)
    ∇ᵤ = grad_1d(Nᵤ-1,du)

    if isScrambled
        ∇ₓ = scrambleX(∇ₓ,Nᵤ)
        ∇ᵤ = scrambleY(∇ᵤ,Nₓ)
    end

    return ∇ₓ,∇ᵤ

end # end of function grad


function laplacian(k::Number,inputs::Dict)

    ∂ = inputs["∂_ext"]
    geometry = inputs["geometry"]
    x = inputs["x_ext"]
    y = inputs["u_ext"]

    ℓₓ = inputs["ℓ_ext"][1]
    ℓᵤ = inputs["ℓ_ext"][2]

    Nₓ = inputs["N_ext"][1]
    Nᵤ = inputs["N_ext"][2]

    dx = inputs["dr"][1]
    du = inputs["dr"][2]

    Σₓ,Σᵤ = σ((x,y),∂,geometry)

    ∇ₓ,∇ᵤ = grad([Nₓ,Nᵤ],[dx, du]; isScrambled=false)

    sₓ₁ = sparse(1:Nₓ-1,1:Nₓ-1,1./(1+.5im*(Σₓ[1:end-1] + Σₓ[2:end])/real(k)),Nₓ-1,Nₓ-1)
    sₓ₂ = sparse(1:Nₓ,1:Nₓ,1./(1+1im*(Σₓ)/real(k)),Nₓ,Nₓ)

    sᵤ₁ = sparse(1:Nᵤ-1,1:Nᵤ-1,1./(1+.5im*(Σᵤ[1:end-1] + Σᵤ[2:end])/real(k)),Nᵤ-1,Nᵤ-1)
    sᵤ₂ = sparse(1:Nᵤ,1:Nᵤ,1./(1+1im*(Σᵤ)/real(k)),Nᵤ,Nᵤ)

    ∇ₓ²= -(sₓ₂*∇ₓ.'*sₓ₁*∇ₓ)
    ∇ₓ²[1,1]   += -2/dx^2
    ∇ₓ²[Nₓ,Nₓ] += -2/dx^2
    
    ∇ᵤ²= -(sᵤ₂*∇ᵤ.'*sᵤ₁*∇ᵤ)
    ∇ᵤ²[1,1]   += -2/du^2
    ∇ᵤ²[Nᵤ,Nᵤ] += -2/du^2

    ∇² = scrambleX(∇ₓ²,Nᵤ) + scrambleY(∇ᵤ²,Nₓ)
    
    return ∇²

end # end of function laplacian


function σ(r,∂,geometry)

    x = copy(r[1])
    y = copy(r[2])

    dx = x[2]-x[1]
    dy = y[2]-y[1]

    αₓ = (1+.1im)*( (PML_ρ/dx)^(PML_power_law+1) )/ ( (PML_power_law+1)*log(PML_extinction) )^PML_power_law
    αᵤ = (1+.1im)*( (PML_ρ/dy)^(PML_power_law+1) )/ ( (PML_power_law+1)*log(PML_extinction) )^PML_power_law


    r = whichRegion(r,∂,geometry)

    sₓ = similar(x,Complex128)
    sᵤ = similar(y,Complex128)

    for i in 1:length(x)
        if r[i,1] in (1,8,7)
            sₓ[i] = αₓ*abs(x[i]-∂[5])^PML_power_law
        #extinction*(abs(x[i]-∂[5])/abs(∂[1]-∂[5])).^power
        elseif r[i,1] in (3,4,5)
            sₓ[i] = αₓ*abs(x[i]-∂[6])^PML_power_law
        #extinction*(abs(x[i]-∂[6])/abs(∂[2]-∂[6])).^power
        else
            sₓ[i] = 0
        end
    end

    for j in 1:length(y)
        if r[1,j] in (1,2,3)
            sᵤ[j] = αᵤ*abs(y[j]-∂[8])^PML_power_law
        #extinction*(abs(y[j]-∂[8])/abs(∂[4]-∂[8])).^power
        elseif r[1,j] in (7,6,5)
            sᵤ[j] = αᵤ*abs(y[j]-∂[7])^PML_power_law
        #extinction*(abs(y[j]-∂[7])/abs(∂[3]-∂[7])).^power
        else
            sᵤ[j] = 0
        end
    end

    return sₓ,sᵤ

end # end of function σ


function whichRegion(r,∂,geometry)

    x = r[1]
    y = r[2]

    region = zeros(Int,length(x),length(y));

    for i in 1:length(x), j in 1:length(y)

        if ∂[1] ≤ x[i] ≤ ∂[5]
            if ∂[8] ≤ y[j] ≤ ∂[4]
                region[i,j] = 1
            elseif ∂[3] ≤ y[j] ≤ ∂[7]
                region[i,j] = 7
            elseif ∂[7] ≤ y[j] ≤ ∂[8]
                region[i,j] = 8
            end                
        end

        if ∂[5] ≤ x[i] ≤ ∂[6]
            if ∂[8] ≤ y[j] ≤ ∂[4]
                region[i,j] = 2
            elseif ∂[3] ≤ y[j] ≤ ∂[7]
                region[i,j] = 6
            end                
        end

        if ∂[6] ≤ x[i] ≤ ∂[2]
            if ∂[8] ≤ y[j] ≤ ∂[4]
                region[i,j] = 3
            elseif ∂[3] ≤ y[j] ≤ ∂[7]
                region[i,j] = 5
            elseif ∂[7] ≤ y[j] ≤ ∂[8]
                region[i,j] = 4
            end                
        end

        if  (∂[5] < x[i] < ∂[6]) & (∂[7] < y[j] < ∂[8])
            region[i,j] = 8 + geometry(x[i], y[j], ∂[5:end])
        end

    end

    return region

end # end of function whichRegion


function trapz(z,dr)

    dx = dr[1]
    dy = dr[2]

    integral = dx*dy*sum(z) # may have to address boundary terms later

    return integral

end # end of function trapz

#end
function processInputs()

   
    (N, λ₀, λ, ∂, F, ɛ, γ⊥, D₀, a, geometry, incidentWave, extras) = evalfile("SALT_2d_Inputs.jl")

    ω₀ = 2π./λ₀
    ω  = 2π./λ
    k  = ω
    k₀ = ω₀

    ℓ = [∂[2] - ∂[1], ∂[4] - ∂[3]]
    ℓₓ = ℓ[1]
    ℓᵤ = ℓ[2]

    Nₓ = N[1]
    Nᵤ = N[2]

    ##########################

    dx = ℓₓ/(Nₓ-1)
    du = ℓᵤ/(Nᵤ-1)
    dr = [dx, du]

    nₓ = 1/dx
    nᵤ = 1/du

    dNₓ1 = ceil(Int,(PML_power_law+1)*log(PML_extinction)/PML_ρ)
#    dNₓ1 = ceil(Integer,1.0*λ₀*nₓ)
#    dNₓ2 = ceil(Integer,0.05*λ₀*nₓ)

    dNᵤ1 = ceil(Int,(PML_power_law+1)*log(PML_extinction)/PML_ρ)
#    dNᵤ1 = ceil(Integer,1.0*λ₀*nᵤ)
#    dNᵤ2 = ceil(Integer,0.05*λ₀*nᵤ)

    dN1 = [dNₓ1 dNᵤ1]
    dN2 = [dNₓ2 dNᵤ2]

    Nₓ_ext = Nₓ + 2(dNₓ1+dNₓ2)
    ℓₓ_ext = dx*(Nₓ_ext-1)

    Nᵤ_ext = Nᵤ + 2(dNᵤ1+dNᵤ2)
    ℓᵤ_ext = du*(Nᵤ_ext-1)

    N_ext = [Nₓ_ext Nᵤ_ext]
    ℓ_ext = [ℓₓ_ext ℓᵤ_ext]

    x_ext = vcat(-[(dNₓ1+dNₓ2):-1:1;]*dx+∂[1],  linspace(∂[1],∂[2],Nₓ), [1:(dNₓ1+dNₓ2);]*dx+∂[2])
    x_inds = dNₓ1+dNₓ2+collect(1:Nₓ)
    x = x_ext[x_inds]

    u_ext = vcat(-[(dNᵤ1+dNᵤ2):-1:1;]*du+∂[3],  linspace(∂[3],∂[4],Nᵤ), [1:(dNᵤ1+dNᵤ2);]*du+∂[4])
    u_inds = dNᵤ1+dNᵤ2+collect(1:Nᵤ)
    u = u_ext[u_inds]
    
    xu_inds = ( x_inds*ones(Int,size(u_inds')) + (ones(Int,size(x_inds))*(u_inds-1)')*Nₓ_ext )[:]

    ∂_ext = [x_ext[1]-dx/2 x_ext[end]+dx/2 u_ext[1]-du/2 u_ext[end]+du/2 ∂]

    F[F .== zero(Float64)] = F_min
    F_ext = [F_min F_min F_min F_min F_min F_min F_min F_min F]

    ɛ_ext = [1 1 1 1 1 1 1 1 ɛ]

    inputs = Dict{Any,Any}(
        "λ" => λ,
        "λ₀" => λ₀,
        "ω" => ω,
        "ω₀" => ω₀,
        "k" => k,
        "k₀" => k₀,
        "N" => N,
        "ℓ" => ℓ,
        "dr" => dr,
        "x_ext" => x_ext,
        "x_inds" => x_inds,
        "x" => x,
        "u_ext" => u_ext,
        "u_inds" => u_inds,
        "u" => u,
        "xu_inds" => xu_inds,
        "∂" => ∂,
        "ɛ" => ɛ,
        "F" => F,
        "N_ext" => N_ext,
        "ℓ_ext" => ℓ_ext,
        "∂_ext" => ∂_ext,
        "ɛ_ext" => ɛ_ext,
        "F_ext" => F_ext,
        "x" => x,
        "γ⊥" => γ⊥,
        "D₀" => D₀,
        "a" => a,
        "geometry" => geometry,
        "incidentWave" => incidentWave,
        "extras" => extras,
        "dN1" => dN1,
        "dN2" => dN2)

    return inputs

end # end of function processInputs



function updateInputs(inputs::Dict)

    ∂ = inputs["∂"]
    N = inputs["N"]
    λ₀ = inputs["λ₀"]
    λ = inputs["λ"]
    F = inputs["F"]
    ɛ = inputs["ɛ"]

    ######################
    
    ω₀ = 2π./λ₀
    ω  = 2π./λ
    k  = ω
    k₀ = ω₀

    ℓ = [∂[2] - ∂[1], ∂[4] - ∂[3]]
    ℓₓ = ℓ[1]
    ℓᵤ = ℓ[2]

    Nₓ = N[1]
    Nᵤ = N[2]

    ##########################

    dx = ℓₓ/(Nₓ-1)
    du = ℓᵤ/(Nᵤ-1)
    dr = [dx, du]

    nₓ = 1/dx
    nᵤ = 1/du

    dNₓ1 = ceil(Int,(PML_power_law+1)*log(PML_extinction)/PML_ρ)
#    dNₓ1 = ceil(Integer,1.0*λ₀*nₓ)
#    dNₓ2 = ceil(Integer,0.05*λ₀*nₓ)

    dNᵤ1 = ceil(Int,(PML_power_law+1)*log(PML_extinction)/PML_ρ)
#    dNᵤ1 = ceil(Integer,1.0*λ₀*nᵤ)
#    dNᵤ2 = ceil(Integer,0.05*λ₀*nᵤ)

    dN1 = [dNₓ1 dNᵤ1]
    dN2 = [dNₓ2 dNᵤ2]

    Nₓ_ext = Nₓ + 2(dNₓ1+dNₓ2)
    ℓₓ_ext = dx*(Nₓ_ext-1)

    Nᵤ_ext = Nᵤ + 2(dNᵤ1+dNᵤ2)
    ℓᵤ_ext = du*(Nᵤ_ext-1)

    N_ext = [Nₓ_ext Nᵤ_ext]
    ℓ_ext = [ℓₓ_ext ℓᵤ_ext]

    x_ext = vcat(-[(dNₓ1+dNₓ2):-1:1;]*dx+∂[1],  linspace(∂[1],∂[2],Nₓ), [1:(dNₓ1+dNₓ2);]*dx+∂[2])
    x_inds = dNₓ1+dNₓ2+collect(1:Nₓ)
    x = x_ext[x_inds]

    u_ext = vcat(-[(dNᵤ1+dNᵤ2):-1:1;]*du+∂[3],  linspace(∂[3],∂[4],Nᵤ), [1:(dNᵤ1+dNᵤ2);]*du+∂[4])
    u_inds = dNᵤ1+dNᵤ2+collect(1:Nᵤ)
    u = u_ext[u_inds]
    
    xu_inds = ( x_inds*ones(Int,size(u_inds')) + (ones(Int,size(x_inds))*(u_inds-1)')*Nₓ_ext )[:]

    ∂_ext = [x_ext[1]-dx/2 x_ext[end]+dx/2 u_ext[1]-du/2 u_ext[end]+du/2 ∂]

    F[F .== zero(Float64)] = F_min
    F_ext = [F_min F_min F_min F_min F_min F_min F_min F_min F]

    ɛ_ext = [1 1 1 1 1 1 1 1 ɛ]
   
    
    inputsNew = Dict{Any,Any}(
        "λ" => λ,
        "λ₀" => λ₀,
        "ω" => ω,
        "ω₀" => ω₀,
        "k" => k,
        "k₀" => k₀,
        "N" => N,
        "ℓ" => ℓ,
        "dr" => dr,
        "x_ext" => x_ext,
        "x_inds" => x_inds,
        "x" => x,
        "u_ext" => u_ext,
        "u_inds" => u_inds,
        "u" => u,
        "xu_inds" => xu_inds,
        "∂" => ∂,
        "ɛ" => ɛ,
        "F" => F,
        "N_ext" => N_ext,
        "ℓ_ext" => ℓ_ext,
        "∂_ext" => ∂_ext,
        "ɛ_ext" => ɛ_ext,
        "F_ext" => F_ext,
        "x" => x,
        "γ⊥" => inputs["γ⊥"],
        "D₀" => inputs["D₀"],
        "a" => inputs["a"],
        "geometry" => inputs["geometry"],
        "incidentWave" => inputs["incidentWave"],
        "extras" => inputs["extras"],
        "dN1" => dN1,
        "dN2" => dN2)

    return inputsNew
    
    
end # end of function updateInputs

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

end # end of module Core