
"""
j, ∇² = synthesize_source(inputs, k)
"""
function synthesize_source(inputs::InputStruct, k::Complex128)::
    Tuple{Array{Complex128,1},SparseMatrixCSC{Complex128,Int64}}

    N = prod(inputs.N_ext)
    φ₊ = zeros(Complex128,N)
    φ₋ = zeros(Complex128,N)
    φt₊ = zeros(Complex128,N)
    φt₋ = zeros(Complex128,N)

    M₊, M₋ = source_mask(inputs)

    for m in 1:length(inputs.channels)
        φt₊, φt₋ = incident_modes(inputs, k, m)
        φ₊ += inputs.a[m]*φt₊
        φ₋ += inputs.a[m]*φt₋
    end

    ∇² = laplacian(k,inputs)

    inputs1 = deepcopy(inputs)
    inputs1.n₁_vals[inputs1.n₁_inds[inputs1.scatteringRegions]] = 1.
    inputs1.n₂_vals[inputs1.n₂_inds[inputs1.scatteringRegions]] = 0.
    updateInputs!(inputs1, [:n₁_vals, :n₂_vals], Any[inputs1.n₁_vals, inputs1.n₂_vals]);

    k² = k^2
    ɛk² = sparse(1:N, 1:N, inputs1.ɛ_sm[:]*k², N, N, +)

    j = (∇²+ɛk²)*(M₊.*φ₊ + M₋.*φ₋)

    return j, ∇²
end


"""
M₊, M₋ = source_mask(inputs)

    M₊ is the mask for the incident field, corresponds to a circle or radius ∂S
        if length(∂S) = 1, and a rectangle with boundaries at the elements of ∂S
        otherwise
    M₋ is the mask for the outgoing field, corresponds to a rectangle at PML
        boundary
"""
function source_mask(inputs::InputStruct)::Tuple{Array{Bool,1},Array{Bool,1}}

    ∂S = inputs.∂S
    ∂R = inputs.∂R

    if length(∂S)>1
        M₊ = (∂S[1] .≤ inputs.x̄_ext[1] .≤ ∂S[2]) .& (∂S[3] .≤ inputs.x̄_ext[2] .≤ ∂S[4])
    elseif length(∂S)==1
        r = sqrt.(inputs.x̄_ext[1].^2 + inputs.x̄_ext[2].^2)
        M₊ = r .≤ ∂S[1]
    end
    M₋ = (∂R[1] .≤ inputs.x̄_ext[1] .≤ ∂R[2]) .& (∂R[3] .≤ inputs.x̄_ext[2] .≤ ∂R[4])
    return M₊, M₋
end


"""
incidentModes(inputs, k, m)
"""
function incident_modes(inputs::InputStruct, k::Complex128, m::Int)::
    Tuple{Array{Complex128,1},Array{Complex128,1}}

    φ₊ = zeros(Complex128, prod(inputs.N_ext))
    φ₋ = zeros(Complex128, prod(inputs.N_ext))

    bc_sig = inputs.bc_sig
    if bc_sig in ["Oddd", "Odnn", "Oddn", "Odnd"]
        x = inputs.x̄_ext[1] - inputs.∂R[2]
        y = inputs.x̄_ext[2]
        φy = quasi_1d_transverse_y.(inputs,m,y)
        kᵤ = quasi_1d_transverse_y(inputs,m)
        kₓ = sqrt(k^2 - kᵤ^2)
        φ₊ = +sqrt(1/real(kₓ))*exp(+1im*kₓ*x).*φy
        φ₋ = -sqrt(1/real(kₓ))*exp(-1im*kₓ*x).*φy
    elseif bc_sig in ["dOdd", "dOnn", "dOdn", "dOnd"]
        x = inputs.x̄_ext[1] - inputs.∂R[1]
        y = inputs.x̄_ext[2]
        φy = quasi_1d_transverse_y.(inputs,m,y)
        kᵤ = quasi_1d_transverse_y(inputs,m)
        kₓ = sqrt(k^2 - kᵤ^2)
        φ₊ = +sqrt(1/real(kₓ))*exp(-1im*kₓ*x).*φy
        φ₋ = -sqrt(1/real(kₓ))*exp(+1im*kₓ*x).*φy
    elseif (bc_sig in ["OOOO", "IIII"]) && (!isempty(inputs.wgd))
        kₓ, φy = wg_transverse_y(inputs, k, m)
        if inputs.channels[m].side in ["l", "L", "left", "Left"]
            x = inputs.x̄_ext[1] - inputs.∂R[1]
            φ₊ = +sqrt(1/real(kₓ))*exp.(+1im*kₓ*x).*φy
        elseif inputs.channels[m].side in ["r", "R", "right", "Right"]
            x = inputs.x̄_ext[1] - inputs.∂R[2]
            φ₊ = +sqrt(1/real(kₓ))*exp.(-1im*kₓ*x).*φy
        end
    elseif (bc_sig in ["OOOO", "IIII"])
        x = inputs.x̄_ext[1]
        y = inputs.x̄_ext[2]
        r = sqrt.(x.^2 + y.^2)
        θ = atan2.(y,x)
        q = inputs.channels[m].tqn
        φ₊ = exp.(1im*q*θ).*besselj.(q,k*r)
        M₊, M₋ = source_mask(inputs)
        φ₋[M₋ .& .!M₊] = exp.(1im*q*θ[M₋ .& .!M₊]).*hankelh1.(q,k*r[M₋ .& .!M₊])/2
    end

    return φ₊, φ₋
end


"""
kₓ = quasi_1d_transverse_x(inputs, m)
    OR
φ = quasi_1d_transverse_x(inputs, m, x)
"""
function quasi_1d_transverse_x(inputs::InputStruct, m::Int)::Float64

    ℓ = inputs.ℓ[1]
    bc = inputs.bc
    q = inputs.channels[m].tqn

    if bc[1:2] == ["n", "n"]
        kₓ = (q-1)*π/ℓ
    elseif bc[1:2] == ["n", "d"]
        kₓ = (q-1/2)*π/ℓ
    elseif bc[1:2] == ["d", "n"]
        kₓ = (q-1/2)*π/ℓ
    elseif bc[1:2] == ["d", "d"]
        kₓ = q*π/ℓ
    end

    return kₓ
end
function quasi_1d_transverse_x(inputs::InputStruct, m::Int, x::Float64)::Float64

    ℓ = inputs.ℓ[1]
    bc = inputs.bc
    kₓ = quasi_1d_transverse_y(inputs,m)

    if bc[1:2] == ["n", "n"]
        φ = sqrt(2/ℓ)*cos.(kₓ*(x-inputs.∂R[1]))
    elseif bc[1:2] == ["n", "d"]
        φ = sqrt(2/ℓ)*cos.(kₓ*(x-inputs.∂R[1]))
    elseif bc[1:2] == ["d", "n"]
        φ = sqrt(2/ℓ)*sin.(kₓ*(x-inputs.∂R[1]))
    elseif bc[1:2] == ["d", "d"]
        φ = sqrt(2/ℓ)*sin.(kₓ*(x-inputs.∂R[1]))
    end

    return φ
end


"""
kᵤ = quasi_1d_transverse_y(inputs, m)
    OR
φ = quasi_1d_transverse_y(inputs, m, y)
"""
function quasi_1d_transverse_y(inputs::InputStruct, m::Int)::Float64

    ℓ = inputs.ℓ[2]
    bc = inputs.bc
    q = inputs.channels[m].tqn

    if bc[3:4] == ["n", "n"]
        kᵤ = (q-1)*π/ℓ
    elseif bc[3:4] == ["n", "d"]
        kᵤ = (q-1/2)*π/ℓ
    elseif bc[3:4] == ["d", "n"]
        kᵤ = (q-1/2)*π/ℓ
    elseif bc[3:4] == ["d", "d"]
        kᵤ = q*π/ℓ
    end

    return kᵤ
end
function quasi_1d_transverse_y(inputs::InputStruct, m::Int, y::Float64)::Float64

    ℓ = inputs.ℓ[2]
    bc = inputs.bc
    kᵤ = quasi_1d_transverse_y(inputs,m)

    if bc[3:4] == ["n", "n"]
        φ = sqrt(2/ℓ)*cos.(kᵤ*(y-inputs.∂R[3]))
    elseif bc[3:4] == ["n", "d"]
        φ = sqrt(2/ℓ)*cos.(kᵤ*(y-inputs.∂R[3]))
    elseif bc[3:4] == ["d", "n"]
        φ = sqrt(2/ℓ)*sin.(kᵤ*(y-inputs.∂R[3]))
    elseif bc[3:4] == ["d", "d"]
        φ = sqrt(2/ℓ)*sin.(kᵤ*(y-inputs.∂R[3]))
    end

    return φ
end


"""
kᵤ, φᵤ = wg_transverse_y(inputs, m, y)
"""
function wg_transverse_y(inputs1::InputStruct, k::Complex128, m::Int)::
    Tuple{Complex128, Array{Complex128,1}}

    inputs = open_to_pml_out(inputs1, true)

    wg_pos_ind = 3
    ind = find( (inputs.r_ext.==(8+inputs.channels[m].wg) )[1,:])[wg_pos_ind]

    fields = [:wgd,:wgn,:wgt,:wgp]
    vals = [ String[inputs.wgd[inputs.channels[m].wg]],
             Float64[inputs.wgn[inputs.channels[m].wg]],
             Float64[inputs.wgt[inputs.channels[m].wg]],
             Float64[inputs.wgp[inputs.channels[m].wg]],0]
    updateInputs!(inputs,fields,vals)

    N = inputs.N_ext[2]
    k² = k^2
    ε_sm = real(inputs.ε_sm[inputs.x₁_inds[1],:])
    εk² = sparse(1:N, 1:N, ε_sm[:]*k², N, N, +)
    ∇₁², ∇₂² = laplacians(k,inputs)

    nev = 4 + 2*inputs.channels[m].tqn
    kₓ²,φ = eigs(∇₂²+εk², nev=nev, sigma=3*k², which = :LM)
    perm = sortperm(kₓ²; by = x -> real(sqrt.(x)), rev=true)
    φ_temp = φ[:,perm[inputs.channels[m].tqn]]
    φ_temp = φ_temp*( conj(φ_temp[ind])/abs(φ_temp[ind]) ) #makes field positive at wg_pos_ind
    φy = repmat(transpose(φ_temp),inputs.N_ext[1],1)[:]
    φy = φy/sqrt(sum(φ_temp.*ε_sm.*φ_temp)*inputs.dx̄[2])
    return (sqrt.(kₓ²[perm[inputs.channels[m].tqn]]), φy)
end


"""
s = analyze_output(inputs, k, ψ, m)
    s is the output coefficient in the mth channel

    S is constructed from s for unit inputs on each channel
"""
function analyze_output(inputs::InputStruct, k::Complex128,
    ψ::Array{Complex{Float64},1}, m::Int)::Complex128

    bc_sig = inputs.bc_sig

    if bc_sig in ["Oddd", "Odnn", "Oddn", "Odnd"]
        x = inputs.x₁[1] - inputs.∂R[2]
        y = inputs.x₂_ext
        φy = quasi_1d_transverse_y.(inputs,m,y)
        kᵤ = quasi_1d_transverse_y(inputs,m)
        kₓ = sqrt(k^2 - kᵤ^2)
        φ = +sqrt(1/kₓ)*exp(+1im*kₓ*x)*φy
        P = reshape(ψ[inputs.x̄_inds],inputs.N[1],:)[1,:]
        cm = sum(φ.*P)*inputs.dx̄[2]
        bm = -inputs.a[m]
    elseif bc_sig in ["dOdd", "dOnn", "dOdn", "dOnd"]
        x = inputs.x₁[end] - inputs.∂R[1]
        y = inputs.x₂_ext
        φy = quasi_1d_transverse_y.(inputs,m,y)
        kᵤ = quasi_1d_transverse_y(inputs,m)
        kₓ = sqrt(k^2 - kᵤ^2)
        φ = +sqrt(1/kₓ)*exp(-1im*kₓ*x)*φy
        P = reshape(ψ[inputs.x̄_inds],inputs.N[1],:)[end,:]
        cm = sum(φ.*P)*inputs.dx̄[2]
        bm = -inputs.a[m]
    elseif (bc_sig in ["OOOO", "IIII"]) && (!isempty(inputs.wgd))
        if (inputs.wgd[inputs.channels[m].wg] in ["x", "X"])
            kₓ, φy = wg_transverse_y(inputs, k, m)
            if inputs.channels[m].side in ["l", "L", "left", "Left"]
                x = inputs.x₁[1] - inputs.∂R[1]
                phs = exp.(+1im*kₓ*x)
                xb = inputs.x₁[1] - inputs.∂R[2] # ballistic
                phsb = exp(-1im*kₓ*xb)
                P = reshape(ψ[inputs.x̄_inds],inputs.N[1],:)[1,:]
                ε = inputs.ε_sm[1,inputs.x₂_inds]
            elseif inputs.channels[m].side in ["r", "R", "right", "Right"]
                x = inputs.x₁[end] - inputs.∂R[2]
                phs = exp.(-1im*kₓ*x)
                xb = inputs.x₁[end] - inputs.∂R[1] # ballistic
                phsb = exp(+1im*kₓ*xb)
                P = reshape(ψ[inputs.x̄_inds],inputs.N[1],:)[end,:]
                ε = inputs.ε_sm[end,inputs.x₂_inds]
            end
            φ = reshape(φy[inputs.x̄_inds],inputs.N[1],:)[1,:]
        elseif inputs.channels[m].wgd in ["y", "Y"]
            error("Haven't written vertical waveguide code yet.")
        end

        wg_bool = [inputs.channels[q].wg for q in 1:length(inputs.channels)] .== inputs.channels[m].wg
        tqn_bool = [inputs.channels[q].tqn for q in 1:length(inputs.channels)] .== inputs.channels[m].tqn
        side_bool = [inputs.channels[q].side for q in 1:length(inputs.channels)] .== inputs.channels[m].side
        wg_ind = find(wg_bool .& tqn_bool .& side_bool)
        wg_bal_ind = find(wg_bool .& tqn_bool .& .!side_bool)
        if (length(wg_ind)>1) | (length(wg_bal_ind)>1)
            error("Channels not uniquely defined.")
        end

        cm = sqrt(kₓ)*phs*sum(φ.*ε.*P)*inputs.dx̄[2]
        bm = inputs.a[wg_bal_ind[1]]*phsb
    elseif (bc_sig in ["OOOO", "IIII"])
        cm = analyze_into_angular_momentum(inputs, k, ψ, m, "out")
    end

    return cm
end


"""
cm = analyze_input(inputs, k, ψ, m)

    cm is the input power for a CPA input, that is, one where there is no output
"""
function analyze_input(inputs::InputStruct, k::Complex128,
    ψ::Array{Complex{Float64},1}, m::Int)::Complex128

    bc_sig = inputs.bc_sig

    if bc_sig in ["Oddd", "Odnn", "Oddn", "Odnd"]
        error("Haven't written input analyzer for one-sided input in a waveguide")
    elseif bc_sig in ["dOdd", "dOnn", "dOdn", "dOnd"]
        error("Haven't written input analyzer for one-sided input in a waveguide")
    elseif (bc_sig in ["OOOO", "IIII"]) && (!isempty(inputs.wgd))
        if (inputs.wgd[inputs.channels[m].wg] in ["x", "X"])
            kₓ, φy = wg_transverse_y(inputs, k, m)
            if inputs.channels[m].side in ["l", "L", "left", "Left"]
                x = inputs.x₁[1] - inputs.∂R[1]
                phs = exp.(-1im*kₓ*x)
                P = reshape(ψ[inputs.x̄_inds],inputs.N[1],:)[1,:]
                ε = inputs.ε_sm[1,inputs.x₂_inds]
            elseif inputs.channels[m].side in ["r", "R", "right", "Right"]
                x = inputs.x₁[end] - inputs.∂R[2]
                phs = exp.(+1im*kₓ*x)
                P = reshape(ψ[inputs.x̄_inds],inputs.N[1],:)[end,:]
                ε = inputs.ε_sm[end,inputs.x₂_inds]
            end
            φ = reshape(φy[inputs.x̄_inds],inputs.N[1],:)[1,:]
        elseif inputs.channels[m].wgd in ["y", "Y"]
            error("Haven't written vertical waveguide code yet.")
        end

        wg_bool = [inputs.channels[q].wg for q in 1:length(inputs.channels)] .== inputs.channels[m].wg
        tqn_bool = [inputs.channels[q].tqn for q in 1:length(inputs.channels)] .== inputs.channels[m].tqn
        side_bool = [inputs.channels[q].side for q in 1:length(inputs.channels)] .!== inputs.channels[m].side
        wg_ind = find(wg_bool .& tqn_bool .& side_bool)
        if length(wg_ind) > 1
            error("Channels not uniquely defined.")
        end

        cm = sqrt(kₓ)*phs*sum(φ.*ε.*P)*inputs.dx̄[2]

    elseif (bc_sig in ["OOOO", "IIII"])
        cm = analyze_into_angular_momentum(inputs, k, ψ, m, "in")
    end

    return cm
end


################################################################################
### Analyzer Subroutines
################################################################################

"""
analyze_into_angular_momentum(inputs, k, ψ, m, direction)
"""
function analyze_into_angular_momentum(inputs::InputStruct, k::Complex128,
    ψ::Array{Complex{Float64},1}, m::Int, direction::String)::Complex128

    nθ = Int(5e3)+1
    θ = linspace(0,2π,nθ)
    dθ = θ[2]-θ[1]

    # R is radius at which to interpolate
    R = (findmin(abs.(inputs.∂R))[1] + findmin(abs.(inputs.∂S))[1])/2 # FIX THIS

    # interpolate wavefunction at r=R, result is P(θ)
    p = interpolate(reshape(ψ,inputs.N_ext[1],:), BSpline(Linear()), OnGrid() )
    X = R*cos.(θ[1:end-1])
    Y = R*sin.(θ[1:end-1])
    X_int = inputs.N_ext[1]*(X-inputs.∂R_ext[1])/(inputs.∂R_ext[2]-inputs.∂R_ext[1])
    Y_int = inputs.N_ext[2]*(Y-inputs.∂R_ext[3])/(inputs.∂R_ext[4]-inputs.∂R_ext[3])
    P = [p[X_int[ii],Y_int[ii]] for ii in 1:(nθ-1)]

    q = inputs.channels[m].tqn

    if direction == "in"
        cm = sum(exp.(-1im*q*θ[1:end-1]).*P)*dθ./(π*hankelh2(q,k*R))
    elseif direction == "out"
        cm = sum(exp.(-1im*q*θ[1:end-1]).*P)*dθ./(π*hankelh1(q,k*R))
    else
        error("Invalid direction.")
    end

    return cm
end
