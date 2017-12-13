"""
inc = ChannelStruct(wg, side, tqn)
"""
mutable struct ChannelStruct
    tqn::Int        # transverse quantum number
    wg::Int         # waveguide number
    side::String    # side
end


"""
inputs = Inputs(coord, N, N_ext, ℓ, ℓ_ext, x̄, x̄_ext, x̄_inds, dx̄, x₁, x₁_ext, x₁_inds,
    x₂, x₂_ext, x₂_inds, ∂R, ∂R_ext, r, r_ext, ɛ, ɛ_ext, ɛ_sm, F, F_ext, F_sm, ω,
    ω₀, k, k₀, γ⟂, D₀, a, bc, bc_sig, bk, input_modes, scatteringRegions,
    incidentWaveRegions, channels, geometry, geoParams, subPixelNum)
"""
mutable struct InputStruct
    coord::String       # coordinate frame (xy or polar)
    N::Array{Int,1}     # number of pixels in each dimension
    N_ext::Array{Int,1} # number of pixels of extended problem in each dimension
    ℓ::Array{Float64,1} # dimensions of domain
    ℓ_ext::Array{Float64,1} #dimensions of extended domain
    x̄::Tuple{Array{Float64,1},Array{Float64,1}}
    x̄_ext::Tuple{Array{Float64,1},Array{Float64,1}}
    x̄_inds::Array{Int,1}
    dx̄::Array{Float64,1}
    x₁::Array{Float64,1}
    x₁_ext::Array{Float64,1}
    x₁_inds::Array{Int,1}
    x₂::Array{Float64,1}
    x₂_ext::Array{Float64,1}
    x₂_inds::Array{Int,1}
    ∂R::Array{Float64,1}    # domain boundary
    ∂R_ext::Array{Float64,1}# extended domain boundary
    ∂S::Array{Float64,1}   # emitting equivalent source surface
    r::Array{Int,2}
    r_ext::Array{Int,2}
    n₁::Array{Float64,1}    # real part of index for each region
    n₂::Array{Float64,1}    # imag part of index for each region
    ɛ::Array{Complex128,1}  # dielectric in each region
    ɛ_ext::Array{Complex128,1}  # dielectric in extended regions
    ɛ_sm::Array{Complex128,2}   # smoothed dielectric in extended regions
    F::Array{Float64,1}     # pump profile in each region
    F_ext::Array{Float64,1} # pump profile in extended regions
    F_sm::Array{Float64,2}  # smoothed pump profile in extended regions
    ω₀::Complex128          # atomic gain center frequency
    k₀::Complex128          # atomic gain center wave number
    γ⟂::Float64             # atomic depolarization rate
    D₀::Float64
    a::Array{Complex128,1}
    bc::Array{String,1}
    bc_sig::String
    bk::Array{Complex128,1}
    input_modes::Array{Array{Int64,1},1}
    scatteringRegions::Array{Int,1}
    channels::Array{ChannelStruct,1}
    geometry::Function
    geoParams::Array{Float64,1}
    wgd::Array{String,1}
    wgp::Array{Float64,1}
    wgt::Array{Float64,1}
    wge::Array{Float64,1}
    subPixelNum::Int
end


"""
∇ =  grad(N, dx)

    1-dim Gradient with N points, lattice spacing dx. It's the forward gradient (I think).

    sparse ∇[N,N+1]
"""
function grad_1d(N::Int, dx::Float64)::SparseMatrixCSC{Complex128,Int}

    I₁ = Array(1:N)
    J₁ = Array(1:N)
    V₁ = fill(Complex(-1/dx), N)

    I₂ = Array(1:N)
    J₂ = Array(2:(N+1))
    V₂ = fill(Complex(+1/dx), N)

    ∇ = sparse(vcat(I₁,I₂), vcat(J₁,J₂), vcat(V₁,V₂), N, N+1, +)
end #end of function grad_1d


"""
∇₁,∇₂ =  grad(N, dx)

    2-dim gradients with N[1],N[2] points, lattice spacing dx̄[1], dx̄[2].
    It's the forward gradient (I think).

    sparse (∇₁[N,N],∇₂[N,N])
"""
function grad(N::Array{Int,1}, dx̄::Array{Float64,1})::
    Tuple{SparseMatrixCSC{Complex128,Int},SparseMatrixCSC{Complex128,Int}}

    N₁ = N[1]
    dx₁ = dx̄[1]

    N₂ = N[2]
    dx₂ = dx̄[2]

    ∇₁ = grad_1d(N₁-1,dx₁)
    ∇₂ = grad_1d(N₂-1,dx₂)

    ∇₁ = kron(speye(N₂,N₂),∇₁)
    ∇₂ = kron(∇₂,speye(N₁,N₁))

    return ∇₁,∇₂
end # end of function grad


"""
∇₁², ∇₂² = laplacians(k, inputs)

    Computes 2-dim laplacian based on parameters and boundary conditions given in
    INPUTS, evaluated at (complex) frequency K.
"""
function laplacians(k::Complex128, inputs::InputStruct)::
    Tuple{SparseMatrixCSC{Complex128,Int},SparseMatrixCSC{Complex128,Int}}

    # definitions block#
    ∂R = inputs.∂R_ext
    x₁ = inputs.x₁_ext
    x₂ = inputs.x₂_ext
    bc = inputs.bc

    k₁ = inputs.bk[1]
    k₂ = inputs.bk[2]

    ℓ₁ = inputs.ℓ_ext[1]
    ℓ₂ = inputs.ℓ_ext[2]

    N₁ = inputs.N_ext[1]
    N₂ = inputs.N_ext[2]

    dx₁ = inputs.dx̄[1]
    dx₂ = inputs.dx̄[2]

    Σ₁,Σ₂ = σ(inputs)

    ∇₁ = grad_1d(N₁-1,dx₁)
    ∇₂ = grad_1d(N₂-1,dx₂)

    s₁₁ = sparse(1:N₁-1,1:N₁-1,1./(1+.5im*(Σ₁[1:end-1] + Σ₁[2:end])/real(k)),N₁-1,N₁-1)
    s₁₂ = sparse(1:N₁,1:N₁,1./(1+1im*(Σ₁)/real(k)),N₁,N₁)

    s₂₁ = sparse(1:N₂-1,1:N₂-1,1./(1+.5im*(Σ₂[1:end-1] + Σ₂[2:end])/real(k)),N₂-1,N₂-1)
    s₂₂ = sparse(1:N₂,1:N₂,1./(1+1im*(Σ₂)/real(k)),N₂,N₂)

    ∇₁² = -(s₁₂*transpose(∇₁)*s₁₁*∇₁)
    ∇₂² = -(s₂₂*transpose(∇₂)*s₂₁*∇₂)
    ind = [1, N₁, 1, N₂, 1]

    for i in 1:4
        if i ≤ 2
            if bc[i] in ["O", "I"]
                ∇₁²[ind[i],ind[i]]   += -2/dx₁^2
            elseif bc[i] == "d"
                ∇₁²[ind[i],ind[i]]   += -2/dx₁^2
            elseif bc[i] == "n"
                ∇₁²[ind[i],ind[i]]   += 0
            elseif bc[i] == "p"
                ∇₁²[ind[i],ind[i]]   += -1/dx₁^2
                ∇₁²[ind[i],ind[i+1]] += +exp((-1)^(i+1)*1im*ℓ₁*k₁)/dx₁^2
            end
        else
            if bc[i] in ["O", "I"]
                ∇₂²[ind[i],ind[i]]   += -2/dx₂^2
            elseif bc[i] == "d"
                ∇₂²[ind[i],ind[i]]   += -2/dx₂^2
            elseif bc[i] == "n"
                ∇₂²[ind[i],ind[i]]   += 0
            elseif bc[i] == "p"
                ∇₂²[ind[i],ind[i]]   += -1/dx₂^2
                ∇₂²[ind[i],ind[i+1]] += +exp((-1)^(i+1)*1im*ℓ₂*k₂)/dx₂^2
            end
        end
    end

    return ∇₁², ∇₂²
end # end of function laplacians


"""
∇² = laplacian(k, inputs)

    Computes 2-dim laplacian based on parameters and boundary conditions given in
    INPUTS, evaluated at (complex) frequency K.
"""
function laplacian(k::Complex128, inputs::InputStruct)::SparseMatrixCSC{Complex128,Int}

    # definitions block#
    ∂R = inputs.∂R_ext
    x₁ = inputs.x₁_ext
    x₂ = inputs.x₂_ext
    bc = inputs.bc

    ℓ₁ = inputs.ℓ_ext[1]
    ℓ₂ = inputs.ℓ_ext[2]

    N₁ = inputs.N_ext[1]
    N₂ = inputs.N_ext[2]

    dx₁ = inputs.dx̄[1]
    dx₂ = inputs.dx̄[2]

    ∇₁², ∇₂² = laplacians(k,inputs)

    if any(inputs.bc .== "o")

        if any(inputs.bc[1:2] .== "o") && !(bc[3:4]⊆["d", "n"])
            error("Inconsistent boundary conditions. Cannot have an open side without Dirichlet/Neumann top and bottom.")
        elseif any(inputs.bc[3:4] .== "o") && !(bc[1:2]⊆["d", "n"])
            error("Inconsistent boundary conditions. Cannot have open top or bottom without Dirichlet sides.")
        end

        if any(inputs.bc[1:2] .== "o")
            N⟂ = N₂
            Φ = zeros(Complex128,N⟂,N⟂,2)
            x⟂ = x₂ - ∂R[3]
            ℓ⟂ = ℓ₂
        else
            N⟂ = N₁
            Φ = zeros(Complex128,N⟂,N⟂,2)
            x⟂ = x₁ - ∂R[1]
            ℓ⟂ = ℓ₁
        end

        m_cutoff = 2*floor(Int,ℓ⟂*sqrt(real(k^2))/π)

    end

    for i in 1:4
        if i ≤ 2
            if bc[i] == "o"
                for m in 1:m_cutoff
                    if m in inputs.input_modes[i]
                        M = +1
                    else
                        M = -1
                    end
                    k⟂ = M*sqrt(k^2 - (m*π/ℓ⟂)^2)
                    Φ[:,:,i] += (1im*dx₁*dx₂/ℓ⟂)*k⟂*sin.(m*π*x⟂/ℓ⟂)*sin.(m*π*transpose(x⟂)/ℓ⟂)
                end
                Φ[:,:,i] = -(eye(Complex128,N⟂,N⟂)+Φ[:,:,i])\(Φ[:,:,i]*2/dx₁^2)
            end
        else
            if bc[i] == "o"
                for m in 1:m_cutoff
                    if m in inputs.input_modes[i]
                        M = +1
                    else
                        M = -1
                    end
                    k⟂ = M*sqrt(k^2 - (m*π/ℓ⟂)^2)
                    Φ[:,:,i-2] += (1im*dx₁*dx₂/ℓ⟂)*k⟂*sin.(m*π*x⟂/ℓ⟂)*sin.(m*π*transpose(x⟂)/ℓ⟂)
                end
                Φ[:,:,i-2] = -(eye(Complex128,N⟂,N⟂)+Φ[:,:,i-2])\(Φ[:,:,i-2]*2/dx₂^2)
            end
        end
    end

    if !any(inputs.bc .== "o")
        ∇² = kron(speye(N₂,N₂),∇₁²) + kron(∇₂²,speye(N₁,N₁))
    elseif any(inputs.bc[1:2] .== "o")
        ( ∇² = kron(speye(N₂,N₂),∇₁²) + kron(∇₂²,speye(N₁,N₁)) + kron(Φ[:,:,1],
            sparse([1],[1],[1],N₁,N₁)) + kron(Φ[:,:,2],sparse([N₁],[N₁],[1],N₁,N₁)) )
    elseif any(inputs.bc[3:4] .== "o")
        ( ∇² = kron(speye(N₂,N₂),∇₁²) + kron(∇₂²,speye(N₁,N₁)) +
            kron(sparse([1],[1],[1],N₂,N₂),Φ[:,:,1]) +
            kron(sparse([N₂],[N₂],[1],N₂,N₂),Φ[:,:,2]) )
    end

    return ∇²
end # end of function laplacian


"""
s₁, s₂ = σ(inputs)

    Computes conductivity for PML layer in dimensions 1 and 2.
"""
function σ(inputs::InputStruct)::Tuple{Array{Complex128,1},Array{Complex128,1}}

    F_min, PML_extinction, PML_ρ, PML_power_law, α_imag = PML_params()

    x₁ = inputs.x₁_ext
    x₂ = inputs.x₂_ext
    ∂R = inputs.∂R_ext

    dx₁ = inputs.dx̄[1]
    dx₂ = inputs.dx̄[2]

    α = zeros(Complex128,4)
    for i in 1:4
        if inputs.bc[i] == "O"
            ( α[i] = +(1+α_imag*1im)*( (PML_ρ/dx₁)^(PML_power_law+1) )/
                ( (PML_power_law+1)*log(PML_extinction) )^PML_power_law )
        elseif inputs.bc[i] == "I"
            ( α[i] = -(1+α_imag*1im)*( (PML_ρ/dx₂)^(PML_power_law+1) )/
                ( (PML_power_law+1)*log(PML_extinction) )^PML_power_law )
        else
            α[i] = 0
        end
    end

    s₁ = zeros(Complex128,length(x₁))
    s₂ = zeros(Complex128,length(x₂))

    for i in 1:length(x₁)
        if ∂R[1] ≤ x₁[i] ≤ ∂R[5]
            s₁[i] = α[1]*abs(x₁[i]-∂R[5])^PML_power_law
        elseif ∂R[6] ≤ x₁[i] ≤ ∂R[2]
            s₁[i] = α[2]*abs(x₁[i]-∂R[6])^PML_power_law
        end
    end

    for j in 1:length(x₂)
        if ∂R[3] ≤ x₂[j] ≤ ∂R[7]
            s₂[j] = α[3]*abs(x₂[j]-∂R[7])^PML_power_law
        elseif ∂R[8] ≤ x₂[j] ≤ ∂R[4]
            s₂[j] = α[4]*abs(x₂[j]-∂R[8])^PML_power_law
        end
    end

    return s₁,s₂
end # end of function σ


"""
r = whichRegion(xy, ∂R, geometry, geoParams)

    r is an array.

    xy = (x, y).
"""
function whichRegion(xy::Tuple{Array{Float64,1},Array{Float64,1}}, ∂R::Array{Float64,1},
    geometry::Function, geoParams::Array{Float64,1}, wgd::Array{String,1},
    wgp::Array{Float64,1}, wgt::Array{Float64,1})::Array{Int,2}

    x = xy[1]
    y = xy[2]

    region = zeros(Int,length(x),length(y))

    for i in 1:length(x), j in 1:length(y)

        region[i,j] = 8 + length(wgd) + geometry(x[i], y[j], geoParams)

        for w in 1:length(wgd)
            if wgd[w] in ["x", "X"]
                p = y[j]
            elseif wgd[w] in ["y", "Y"]
                p = x[i]
            else
                error("Invalid waveguide direction.")
            end
            if wgp[w]-wgt[w]/2 < p < wgp[w]+wgt[w]/2
                region[i,j] = 8 + w
            end
        end

        if region[i,j] == 8 + length(wgd) + 1
            if ∂R[1] ≤ x[i] ≤ ∂R[5]
                if ∂R[8] ≤ y[j] ≤ ∂R[4]
                    region[i,j] = 1
                elseif ∂R[3] ≤ y[j] ≤ ∂R[7]
                    region[i,j] = 7
                elseif ∂R[7] ≤ y[j] ≤ ∂R[8]
                    region[i,j] = 8
                end
            elseif ∂R[5] ≤ x[i] ≤ ∂R[6]
                if ∂R[8] ≤ y[j] ≤ ∂R[4]
                    region[i,j] = 2
                elseif ∂R[3] ≤ y[j] ≤ ∂R[7]
                    region[i,j] = 6
                end
            elseif ∂R[6] ≤ x[i] ≤ ∂R[2]
                if ∂R[8] ≤ y[j] ≤ ∂R[4]
                    region[i,j] = 3
                elseif ∂R[3] ≤ y[j] ≤ ∂R[7]
                    region[i,j] = 5
                elseif ∂R[7] ≤ y[j] ≤ ∂R[8]
                    region[i,j] = 4
                end
            end
        end

    end

    return region
end # end of function whichRegion


"""
ε_sm, F_sm = subPixelSmoothing_core(XY, XY_ext, ∂R, ε, F, subPixelNum, truncate,
    r, geometry, geoParams)

    XY = (x,y)
    XY_ext = (x_ext, y_ext)
"""
function subPixelSmoothing_core(XY::Tuple{Array{Float64,1}, Array{Float64,1}},
    XY_ext::Tuple{Array{Float64,1}, Array{Float64,1}}, ∂R::Array{Float64,1},
    ɛ::Array{Complex{Float64},1}, F::Array{Float64,1}, subPixelNum::Int,
    truncate::Bool, r::Array{Int,2}, geometry::Function, geoParams::Array{Float64,1},
    wgd::Array{String,1}, wgp::Array{Float64,1}, wgt::Array{Float64,1})::
    Tuple{Array{Complex128,2},Array{Float64,2},Array{Int,2}}

    if truncate
        xy = XY
    else
        xy = XY_ext
    end

    if isempty(r)
        r = whichRegion(xy, ∂R, geometry, geoParams, wgd, wgp, wgt)
    end

    ɛ_smoothed = deepcopy(ɛ[r])
    F_smoothed = deepcopy(F[r])

    for i in 2:(length(xy[1])-1), j in 2:(length(xy[2])-1)

        ( nearestNeighborFlag = (r[i,j]!==r[i,j+1]) | (r[i,j]!==r[i,j-1]) |
            (r[i,j]!==r[i+1,j]) | (r[i,j]!==r[i-1,j]) )

        ( nextNearestNeighborFlag = (r[i,j]!==r[i+1,j+1]) |
            (r[i,j]!==r[i-1,j-1]) | (r[i,j]!==r[i+1,j-1]) | (r[i,j]!==r[i-1,j+1]) )

        if nearestNeighborFlag | nextNearestNeighborFlag

            x_min = (xy[1][i]+xy[1][i-1])/2
            y_min = (xy[2][j]+xy[2][j-1])/2

            x_max = (xy[1][i]+xy[1][i+1])/2
            y_max = (xy[2][j]+xy[2][j+1])/2

            X = Array(linspace(x_min,x_max,subPixelNum))
            Y = Array(linspace(y_min,y_max,subPixelNum))

            subRegion = whichRegion((X,Y), ∂R, geometry, geoParams, wgd, wgp, wgt)
            ɛ_smoothed[i,j] = mean(ɛ[subRegion])
            F_smoothed[i,j] = mean(F[subRegion])

        end

    end

    return ɛ_smoothed, F_smoothed, r
end # end of function subpixelSmoothing_core


"""
ε_sm, F_sm = subPixelSmoothing(inputs; truncate = false)
"""
function subPixelSmoothing(inputs::InputStruct; truncate::Bool = false)::
    Tuple{Array{Complex{Float64},2}, Array{Float64,2},Array{Int,2}}

    XY = (inputs.x₁, inputs.x₂)
    XY_ext = (inputs.x₁_ext, inputs.x₂_ext)

    ɛ_smoothed, F_smoothed, r = subPixelSmoothing_core(XY, XY_ext, inputs.∂R_ext,
        inputs.ɛ_ext, inputs.F_ext, inputs.subPixelNum, truncate, inputs.r_ext,
        inputs.geometry, inputs.geoParams, inputs.wgd, inputs.wgp, inputs.wgt)

    return ɛ_smoothed, F_smoothed, r
end # end of function subpixelSmoothing


"""
∫z_dx = trapz(z, dx̄)
"""
function trapz(z::Array{Complex128,1}, dx̄::Array{Float64,1})::Complex128

    ∫z_dx = prod(dx̄)*sum(z) # may have to address boundary terms later

    return ∫z_dx
end # end of function trapz


"""
processInputs(k₀, k, bc, F, n₁, n₂, a, scatteringRegions, incidentWaveRegions,
    channels, geometry, geoParams;
    coord="xy", N=[300,300], ∂R=[-.5,.5,-.5,.5], D₀=0., γ⟂=1e8, bk=[0.,0.],
    input_modes=[[],[]], subPixelNum=10)
"""
function processInputs(
    geometry::Function,
    geoParams::Array{Float64,1},
    n₁::Array{Float64,1},
    n₂::Array{Float64,1},
    scatteringRegions::Array{Int,1},
    wgd::Array{String,1},
    wgp::Array{Float64,1},
    wgt::Array{Float64,1},
    wge::Array{Float64,1},

    F::Array{Float64,1},
    D₀::Float64,
    k₀::Complex128,
    γ⟂::Float64,

    ∂R::Array{Float64,1},
    bc::Array{String,1},
    bk::Array{Complex{Float64},1},
    input_modes::Array{Array{Int,1},1},

    ∂S::Array{Float64,1},
    channels::Array{ChannelStruct,1},
    a::Array{Complex128,1},

    coord::String,
    N::Array{Int,1},
    subPixelNum::Int
    )::InputStruct

    if !(length(wgd)==length(wgp)==length(wgt)==length(wge))
        error("Waveguide parameters have different lengths.")
    end

    F_min, PML_extinction, PML_ρ, PML_power_law, α_imag = PML_params()

    ω₀ = k₀

    ℓ = [∂R[2] - ∂R[1], ∂R[4] - ∂R[3]]

    dx₁ = ℓ[1]/N[1]
    dx₂ = ℓ[2]/N[2]
    dx̄ = [dx₁, dx₂]

    fix_bc!(bc)

    dN = Int[0, 0, 0, 0]
    for i in 1:4
        if bc[i] in ["O", "I"]
            dN[i] = ceil(Int,(PML_power_law+1)*log(PML_extinction)/PML_ρ)
        elseif bc[i] == "d"
            dN[i] = 0
        elseif bc[i] == "n"
            dN[i] = 0
        elseif bc[i] == "o"
            dN[i] = 0
        elseif bc[i] == "p"
            dN[i] = 0
        end
    end

    N₁_ext = N[1] + dN[1] + dN[2]
    ℓ₁_ext = dx₁*N₁_ext
    x₁_ext::Array{Float64,1} = ∂R[1] + dx₁*(1/2-dN[1]) + dx₁*(0:N₁_ext-1)
    x₁_inds = dN[1] + collect(1:N[1])
    x₁ = x₁_ext[x₁_inds]

    N₂_ext = N[2] + dN[3] + dN[4]
    ℓ₂_ext = dx₂*N₂_ext
    x₂_ext::Array{Float64,1} = ∂R[3] + dx₂*(1/2-dN[3]) + dx₂*(0:N₂_ext-1)
    x₂_inds = dN[3] + collect(1:N[2])
    x₂ = x₂_ext[x₂_inds]

    N_ext = [N₁_ext, N₂_ext]
    ℓ_ext = [ℓ₁_ext, ℓ₂_ext]
    x̄_ext = (reshape(repmat(x₁_ext,1,N_ext[2]),:), reshape(repmat(x₂_ext',N_ext[1],1),:))
    x̄_inds = reshape( repmat(x₁_inds,1,N[2]) + repmat(N₁_ext*(x₂_inds-1)',N[1],1) ,:)
    x̄ = (x̄_ext[1][x̄_inds],x̄_ext[2][x̄_inds])

    ∂R_ext = vcat([x₁_ext[1]-dx₁/2], [x₁_ext[end]+dx₁/2], [x₂_ext[1]-dx₂/2], [x₂_ext[end]+dx₂/2], ∂R)

    F[iszero.(F)] = F_min
    F_ext = vcat([F_min], [F_min], [F_min], [F_min], [F_min], [F_min], [F_min], [F_min], F_min*ones(size(wge)), F)
    ε = (n₁ + 1.0im*n₂).^2
    ɛ_ext = vcat([1], [1], [1], [1], [1], [1], [1], [1], wge, ɛ)

    ɛ_sm, F_sm, r_ext = subPixelSmoothing_core( (x₁, x₂), (x₁_ext, x₂_ext),
        ∂R_ext, ɛ_ext, F_ext, subPixelNum, false, [Int[] Int[]], geometry, geoParams,
        wgd, wgp, wgt)

    r = reshape(r_ext[x̄_inds],N[1],N[2])

    inputs = InputStruct(coord, N, N_ext, ℓ, ℓ_ext, x̄, x̄_ext, x̄_inds, dx̄, x₁, x₁_ext, x₁_inds,
        x₂, x₂_ext, x₂_inds, ∂R, ∂R_ext, ∂S, r, r_ext, n₁, n₂, ɛ, ɛ_ext, ɛ_sm, F, F_ext, F_sm,
        ω₀, k₀, γ⟂, D₀, a, bc, prod(bc), bk, input_modes, scatteringRegions,
        channels, geometry, geoParams, wgd, wgp, wgt, wge, subPixelNum)

    return inputs
end # end of function processInputs


"""
updateInputs!(inputs::InputStruct, fields, value)

    If changes were made to the ∂R, N, k₀, k, F, ɛ, Γ, bc, a, b, run updateInputs to
    propagate these changes through the system.
"""
function updateInputs!(inputs::InputStruct, field::Symbol, value::Any)::InputStruct

    #if !(length(inputs.wgd)==length(inputs.wgp)==length(inputs.wgt)==length(inputs.wge))
    #    error("Waveguide parameters have different lengths.")
    #end

    F_min, PML_extinction, PML_ρ, PML_power_law, α_imag = PML_params()

    setfield!(inputs,field,value)

    if field == :ω₀
        inputs.k₀ = inputs.ω₀
    elseif field == :k₀
        inputs.ω₀ = inputs.k₀
    end

    if field in [:∂R, :N, :bc]
        ∂R = inputs.∂R
        N = inputs.N
        fix_bc!(inputs.bc)
        bc = inputs.bc
        inputs.bc_sig = prod(bc)

        ℓ = [∂R[2] - ∂R[1], ∂R[4] - ∂R[3]]

        dx₁ = ℓ[1]/N[1]
        dx₂ = ℓ[2]/N[2]
        dx̄ = [dx₁, dx₂]

        dN = Int[0, 0, 0, 0]
        for i in 1:4
            if bc[i] in ["O", "I"]
                dN[i] = ceil(Int,(PML_power_law+1)*log(PML_extinction)/PML_ρ)
            elseif bc[i] == "d"
                dN[i] = 0
            elseif bc[i] == "n"
                dN[i] = 0
            elseif bc[i] == "o"
                dN[i] = 0
            elseif bc[i] == "p"
                dN[i] = 0
            end
        end

        N₁_ext = N[1] + dN[1] + dN[2]
        ℓ₁_ext = dx₁*N₁_ext
        x₁_ext::Array{Float64,1} = ∂R[1] + dx₁*(1/2-dN[1]) + dx₁*(0:N₁_ext-1)
        x₁_inds = dN[1] + collect(1:N[1])
        x₁ = x₁_ext[x₁_inds]

        N₂_ext = N[2] + dN[3] + dN[4]
        ℓ₂_ext = dx₂*N₂_ext
        x₂_ext::Array{Float64,1} = ∂R[3] + dx₂*(1/2-dN[3]) + dx₂*(0:N₂_ext-1)
        x₂_inds = dN[3] + collect(1:N[2])
        x₂ = x₂_ext[x₂_inds]

        N_ext = [N₁_ext, N₂_ext]
        ℓ_ext = [ℓ₁_ext, ℓ₂_ext]
        x̄_ext = (reshape(repmat(x₁_ext,1,N_ext[2]),:), reshape(repmat(x₂_ext',N_ext[1],1),:))
        x̄_inds = reshape( repmat(x₁_inds,1,N[2]) + repmat(N₁_ext*(x₂_inds-1)',N[1],1) ,:)
        x̄ = (x̄_ext[1][x̄_inds],x̄_ext[2][x̄_inds])

        ∂R_ext = vcat([x₁_ext[1]-dx₁/2], [x₁_ext[end]+dx₁/2],
            [x₂_ext[1]-dx₂/2], [x₂_ext[end]+dx₂/2], ∂R)

        inputs.N_ext = N_ext
        inputs.ℓ = ℓ
        inputs.ℓ_ext = ℓ_ext
        inputs.dx̄ = dx̄
        inputs.x̄ = x̄
        inputs.x̄_ext = x̄_ext
        inputs.x̄_inds = x̄_inds
        inputs.x₁ = x₁
        inputs.x₁_ext = x₁_ext
        inputs.x₁_inds = x₁_inds
        inputs.x₂ = x₂
        inputs.x₂_ext = x₂_ext
        inputs.x₂_inds = x₂_inds
        inputs.∂R = ∂R
        inputs.∂R_ext = ∂R_ext

    end

    if  field in [:∂R, :N, :bc, :F, :n₁, :n₂, :ε, :wge, :wgt, :wgp, :wgd, :subPixelNum, :geoParams]

        F = inputs.F
        if field == :ε
            n₁ = real(sqrt.(inputs.ε))
            n₂ = imag(sqrt.(inputs.ε))
        else
            n₁ = inputs.n₁
            n₂ = inputs.n₂
        end
        wge = inputs.wge

        F[iszero.(F)] = F_min
        F_ext = vcat([F_min], [F_min], [F_min], [F_min], [F_min], [F_min], [F_min], [F_min], F_min*ones(size(wge)), F)

        ε = (n₁ + 1.0im*n₂).^2
        ɛ_ext = vcat([1], [1], [1], [1], [1], [1], [1], [1], wge, ɛ)

        ɛ_sm, F_sm, r_ext = subPixelSmoothing_core( (inputs.x₁, inputs.x₂),
            (inputs.x₁_ext, inputs.x₂_ext), inputs.∂R_ext, ɛ_ext, F_ext,
            inputs.subPixelNum, false, [Int[] Int[]], inputs.geometry,
            inputs.geoParams, inputs.wgd, inputs.wgp, inputs.wgt)

        r = reshape(r_ext[inputs.x̄_inds],inputs.N[1],inputs.N[2])

        inputs.F_ext = F_ext
        inputs.r = r
        inputs.r_ext = r_ext
        inputs.ɛ_sm = ɛ_sm
        inputs.F_sm = F_sm

    end

    return inputs
end # end of function updateInputs
function updateInputs!(inputs::InputStruct, fields::Array{Symbol,1}, value::Array{Any,1})::InputStruct

    #if !(length(inputs.wgd)==length(inputs.wgp)==length(inputs.wgt)==length(inputs.wge))
    #    error("Waveguide parameters have different lengths.")
    #end

    F_min, PML_extinction, PML_ρ, PML_power_law, α_imag = PML_params()

    for f in 1:length(fields)
        setfield!(inputs,fields[f],value[f])
    end

    if any(fields .== :ω₀)
        inputs.k₀ = inputs.ω₀
    elseif any(fields .== :k₀)
        inputs.ω₀ = inputs.k₀
    end

    if !isempty(fields ∩ [:∂R, :N, :bc])
        ∂R = inputs.∂R
        N = inputs.N
        fix_bc!(inputs.bc)
        bc = inputs.bc
        inputs.bc_sig = prod(bc)

        ℓ = [∂R[2] - ∂R[1], ∂R[4] - ∂R[3]]

        dx₁ = ℓ[1]/N[1]
        dx₂ = ℓ[2]/N[2]
        dx̄ = [dx₁, dx₂]

        dN = Int[0, 0, 0, 0]
        for i in 1:4
            if bc[i] in ["O", "I"]
                dN[i] = ceil(Int,(PML_power_law+1)*log(PML_extinction)/PML_ρ)
            elseif bc[i] == "d"
                dN[i] = 0
            elseif bc[i] == "n"
                dN[i] = 0
            elseif bc[i] == "o"
                dN[i] = 0
            elseif bc[i] == "p"
                dN[i] = 0
            end
        end

        N₁_ext = N[1] + dN[1] + dN[2]
        ℓ₁_ext = dx₁*N₁_ext
        x₁_ext::Array{Float64,1} = ∂R[1] + dx₁*(1/2-dN[1]) + dx₁*(0:N₁_ext-1)
        x₁_inds = dN[1] + collect(1:N[1])
        x₁ = x₁_ext[x₁_inds]

        N₂_ext = N[2] + dN[3] + dN[4]
        ℓ₂_ext = dx₂*N₂_ext
        x₂_ext::Array{Float64,1} = ∂R[3] + dx₂*(1/2-dN[3]) + dx₂*(0:N₂_ext-1)
        x₂_inds = dN[3] + collect(1:N[2])
        x₂ = x₂_ext[x₂_inds]

        N_ext = [N₁_ext, N₂_ext]
        ℓ_ext = [ℓ₁_ext, ℓ₂_ext]
        x̄_ext = (reshape(repmat(x₁_ext,1,N_ext[2]),:), reshape(repmat(x₂_ext',N_ext[1],1),:))
        x̄_inds = reshape( repmat(x₁_inds,1,N[2]) + repmat(N₁_ext*(x₂_inds-1)',N[1],1) ,:)
        x̄ = (x̄_ext[1][x̄_inds],x̄_ext[2][x̄_inds])

        ∂R_ext = vcat([x₁_ext[1]-dx₁/2], [x₁_ext[end]+dx₁/2],
            [x₂_ext[1]-dx₂/2], [x₂_ext[end]+dx₂/2], ∂R)

        inputs.N_ext = N_ext
        inputs.ℓ = ℓ
        inputs.ℓ_ext = ℓ_ext
        inputs.dx̄ = dx̄
        inputs.x̄ = x̄
        inputs.x̄_ext = x̄_ext
        inputs.x̄_inds = x̄_inds
        inputs.x₁ = x₁
        inputs.x₁_ext = x₁_ext
        inputs.x₁_inds = x₁_inds
        inputs.x₂ = x₂
        inputs.x₂_ext = x₂_ext
        inputs.x₂_inds = x₂_inds
        inputs.∂R = ∂R
        inputs.∂R_ext = ∂R_ext

    end

    if  !isempty(fields ∩ [:∂R, :N, :bc, :F, :n₁, :n₂, :ε, :wge, :wgt, :wgp, :wgd, :subPixelNum, :geoParams])

        F = inputs.F
        if any(fields .== :ε)
            n₁ = real(sqrt.(inputs.ε))
            n₂ = imag(sqrt.(inputs.ε))
        else
            n₁ = inputs.n₁
            n₂ = inputs.n₂
        end
        wge = inputs.wge

        F[iszero.(F)] = F_min
        F_ext = vcat([F_min], [F_min], [F_min], [F_min], [F_min], [F_min], [F_min], [F_min], F_min*ones(size(wge)), F)

        ε = (n₁ + 1.0im*n₂).^2
        ɛ_ext = vcat([1], [1], [1], [1], [1], [1], [1], [1], wge, ɛ)

        ɛ_sm, F_sm, r_ext = subPixelSmoothing_core( (inputs.x₁, inputs.x₂),
            (inputs.x₁_ext, inputs.x₂_ext), inputs.∂R_ext, ɛ_ext, F_ext,
            inputs.subPixelNum, false, [Int[] Int[]], inputs.geometry,
            inputs.geoParams, inputs.wgd, inputs.wgp, inputs.wgt)

        r = reshape(r_ext[inputs.x̄_inds],inputs.N[1],inputs.N[2])

        inputs.F_ext = F_ext
        inputs.r = r
        inputs.r_ext = r_ext
        inputs.ɛ_sm = ɛ_sm
        inputs.F_sm = F_sm

    end

    return inputs
end # end of function updateInputs


"""
γ(k,inputs) is the lorentzian gain curve
"""
function γ(k::Complex128, inputs::InputStruct)::Complex128
    return inputs.γ⟂./(k-inputs.k₀+1im*inputs.γ⟂)
end


"""
fix_bc!(bc)
"""
function fix_bc!(bc::Array{String,1})::Array{String,1}
    for i in 1:4
        if bc[i] in ["pml_out", "PML_out"]
            bc[i] = "O"
        elseif bc[i] in ["pml_in", "PML_in"]
            bc[i] = "I"
        elseif bc[i] in ["d", "dirichlet", "Dirichlet", "hard", "h"]
            bc[i] = "d"
        elseif bc[i] in ["n", "neumann",   "Neumann",   "soft", "s"]
            bc[i] = "n"
        elseif bc[i] in ["o", "open", "Open"]
            bc[i] = "o"
        elseif bc[i] in ["p", "periodic", "Periodic"]
            bc[i] = "p"
        end
    end
    return bc
end


"""
inputs = open_to_pml_out(inputs)

    converts "open" or "pml_in" to "pml_out", and creates a copy of inputs if it does so.
"""
function open_to_pml_out(inputs1::InputStruct)::InputStruct
    if any(inputs1.bc .== "o") || any(inputs1.bc .== "I")
        inputs = deepcopy(inputs1)
        for i in 1:4
            if inputs.bc[i] in ["o", "I"]
                inputs.bc[i] = "pml_out"
            end
        updateInputs!(inputs, :bc, inputs.bc)
        end
    else
        inputs = inputs1
    end
    return inputs
end
function open_to_pml_out(inputs1::InputStruct, flag::Bool)::InputStruct
    inputs = deepcopy(inputs1)
    for i in 1:4
        if inputs.bc[i] in ["o", "I"]
            inputs.bc[i] = "pml_out"
        end
        updateInputs!(inputs, :bc, inputs.bc)
    end
    return inputs
end


"""
inputs = open_to_pml_in(inputs)

    converts "open" or "pml_out" to "pml_in", and creates a copy of inputs if it does so.
"""
function open_to_pml_in(inputs1::InputStruct)::InputStruct
    if any(inputs1.bc .== "o") || any(inputs1.bc .== "O")
        inputs = deepcopy(inputs1)
        for i in 1:4
            if inputs.bc[i] in ["o", "O"]
                inputs.bc[i] = "pml_in"
                updateInputs!(inputs, :bc, inputs.bc)
            end
        end
    else
        inputs = inputs1
    end
    return inputs
end


"""
PML_params()
"""
function PML_params()::Tuple{Float64,Float64,Float64,Int,Float64}

    F_min = 1e-15
    PML_extinction = 1e3
    PML_ρ = 1/3
    PML_power_law = 2
    α_imag = -.25

    return F_min, PML_extinction, PML_ρ, PML_power_law, α_imag
end
