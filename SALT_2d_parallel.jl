################################################################################
### NL2 Eigenvalue routines
################################################################################


"""
K,r = computeK_L_core(inputs, k, fields, field_inds, field_vals, nk, F, truncate, ψ_init)

K   = computeK_L_core(inputs, k, fields, field_inds, field_vals, F, truncate, ψ_init)
"""
function computeK_L_core(inputs::InputStruct, k::Complex128, fields::Array{Symbol,1},
    field_inds::Array{Int,1}, field_vals::Array{Array{Float64,1},1}, nk::Int, F::Array{Float64,1},
    truncate::Bool, ψ_init::Array{Complex128,1})::Tuple{SharedArray,Channel}

    dims = tuple(nk, length.(field_vals)...)
    K = SharedArray{Complex128}(dims)

    r = Channel(length(procs(K)))
    for p in procs(K)
        @async put!(r, remotecall_fetch(computeK_L_core!, p, K, inputs, k, fields, field_inds,
                                field_vals, nk, F, truncate, ψ_init))
    end

    return K,r
end
function computeK_L_core(inputs::InputStruct, k::Array{Complex128,1}, fields::Array{Symbol,1},
    field_inds::Array{Int,1}, field_vals::Array{Array{Float64,1},1}, F::Array{Float64,1},
    truncate::Bool, ψ_init::Array{Complex128,1}, dispOpt::Bool)::Tuple{SharedArray,Channel}

    nk = length(k)
    dims = tuple(nk, length.(field_vals)...)
    K = SharedArray{Complex128}(dims)
    ψ = Complex128[]

    inputs1 = deepcopy(inputs)
    for i in 1:nk
        # for j in [1]#:length(field_vals[1])
            if !isempty(size(getfield(inputs1,fields[1])))
                vals_temp = getfield(inputs1,fields[1])
                vals_temp[field_inds[1]] = field_vals[1][1]
                updateInputs!(inputs1,fields[1],vals_temp)
            else
                updateInputs!(inputs1,fields[1],field_vals[1][1])
            end
            # if j == 1
                k_temp, ψ_temp = computeK_L_core(inputs1, k[i]; nk=1, F=F, truncate=truncate, ψ_init=ψ_init)
                K[i,1,ones(Int64,ndims(K)-2)...] = k_temp[1]
                ψ = ψ_temp[:,1]
            # else
            #     k_temp, ψ_temp = computeK_L_core(inputs1, K[i,j-1,ones(Int64,ndims(K)-2)...]; nk=1, F=F, truncate=truncate, ψ_init=ψ)
            #     K[i,j,ones(Int64,ndims(K)-2)...] = k_temp[1]
            #     ψ = ψ_temp[:,1]
            # end
        # end
    end

    # for d in 3:ndims(K)
    r = Channel(length(procs(K)))
    for d in 2:ndims(K)
        if dispOpt
            println("Computing dimension $(d-1)")
        end
        if d < ndims(K)
            @sync for p in procs(K)
                @async remotecall_fetch(computeK_L_core!, p, K, inputs, fields, field_inds,
                                        field_vals, d, F, truncate, ψ_init)
            end
        else

            for p in procs(K)
                @async put!(r,remotecall_fetch(computeK_L_core!, p, K, inputs, fields, field_inds,
                                        field_vals, d, F, truncate, ψ_init))
            end
        end
    end

    return K, r
end


"""
computeK_L_core!(K, inputs, k, fields, field_inds, field_vals, nk, F, truncate, ψ_init)

computeK_L_core!(K, inputs, fields, field_inds, field_vals, dim, F, truncate, ψ_init)
"""
function computeK_L_core!(K::SharedArray, inputs::InputStruct, k::Complex128,
    fields::Array{Symbol,1}, field_inds::Array{Int,1}, field_vals::Array{Array{Float64,1},1},
    nk::Int, F::Array{Float64,1}, truncate::Bool, ψ_init::Array{Complex128,1})

    inds = p_range(K)
    subs = ind2sub(size(K)[2:end],inds)

    for i in 1:length(inds)
        for f in 1:length(fields)
            if !isempty(size(getfield(inputs,fields[f])))
                vals_temp = getfield(inputs,fields[f])
                vals_temp[field_inds[f]] = field_vals[f][subs[f][i]]
                updateInputs!(inputs,fields[f],vals_temp)
            else
                updateInputs!(inputs,fields[f],field_vals[f][subs[f][i]])
            end
        end

        K[:,[subs[j][i] for j in 1:length(subs)]...], ψ = computeK_L_core(inputs, k; nk=nk, F=F, truncate=truncate, ψ_init=ψ_init)
    end

    return
end
function computeK_L_core!(K::SharedArray, inputs::InputStruct, fields::Array{Symbol,1},
    field_inds::Array{Int,1}, field_vals::Array{Array{Float64,1},1}, dim::Int64,
    F::Array{Float64,1}, truncate::Bool, ψ_init::Array{Complex128,1})

    inds = p_range(K,dim)
    subs = ind2sub(size(K)[1:dim-1],inds)
    for d in 2:size(K,dim)
        for i in 1:length(inds)
            for f in 1:length(fields)
                if f < dim-1
                    val_ind = subs[1+f][i]
                elseif f == dim-1
                    val_ind = d
                else
                    val_ind = 1
                end
                if !isempty(size(getfield(inputs,fields[f])))
                    vals_temp = getfield(inputs,fields[f])
                    vals_temp[field_inds[f]] = field_vals[f][val_ind]
                    updateInputs!(inputs,fields[f],vals_temp)
                else
                    updateInputs!(inputs,fields[f],field_vals[f][val_ind])
                end
            end
            k_temp, ψ = computeK_L_core(inputs, K[[subs[j][i] for j in 1:length(subs)]..., d-1, ones(Int64,ndims(K)-dim)...]; nk=1, F=F, truncate=truncate, ψ_init=ψ_init)
            K[[subs[j][i] for j in 1:length(subs)]..., d, ones(Int64,ndims(K)-dim)...] = k_temp[1]
        end
    end

    return
end


"""
K = computeZero_L(inputs, k, fields, field_inds, params; nz=1, F=[1.], truncate=false, ψ_init=[])
    does parallel computation of computeZero_L over fields[field_inds]=params

K = computeZero_L(inputs, k, fields, field_inds, field_vals; F=[1.], truncate=false, ψ_init=[])
"""
function computeZero_L(inputs1::InputStruct, k::Union{Complex128,Float64,Int},
    fields::Array{Symbol,1}, field_inds::Array{Int,1}, params::Array{Array{Float64,1},1};
    nz::Int=1, F::Array{Float64,1}=[1.], truncate::Bool=false,
    ψ_init::Array{Complex128,1}=Complex128[])::Tuple{SharedArray,Channel}

    inputs = open_to_pml_in(inputs1)

    K,r = computeK_L_core(inputs, complex(1.0*k), fields, field_inds, params, nz, F, truncate, ψ_init)
end # end of function computeZero_L
function computeZero_L(inputs1::InputStruct, k::Union{Array{Complex128,1},Array{Float64,1},Array{Int,1}},
    fields::Array{Symbol,1}, field_inds::Array{Int,1}, field_vals::Array{Array{Float64,1},1};
    F::Array{Float64,1}=[1.], truncate::Bool=false,
    ψ_init::Array{Complex128,1}=Complex128[], dispOpt::Bool=false)::Tuple{SharedArray,Channel}

    inputs = open_to_pml_in(inputs1)

    K = computeK_L_core(inputs, complex(1.0.*k), fields, field_inds, field_vals, F, truncate, ψ_init, dispOpt)
end # end of function computeZero_L


"""
inds = p_range(q)
inds = p_range(q, dim)

    q is a shared array.
    returns the indices to be computed on this process.
"""
function p_range(q::SharedArray)::Array{Int,1}
    idx = indexpids(q)
    if idx == 0 # This worker is not assigned a piece
        return 1:0
    end
    nchunks = length(procs(q))
    splits = [round(Int, s) for s in linspace(0, prod(size(q)[2:end]) , nchunks+1)]
    splits[idx]+1:splits[idx+1]
end
function p_range(q::SharedArray, dim::Int64)::Array{Int,1}
    idx = indexpids(q)
    if idx == 0 # This worker is not assigned a piece
        return 1:0
    end
    nchunks = length(procs(q))
    splits = [round(Int, s) for s in linspace(0, prod(size(q)[1:dim-1]) , nchunks+1)]
    splits[idx]+1:splits[idx+1]
end


################################################################################
### NL2 Eigenvalue routines
################################################################################


"""
k = computeK_NL2_parallel(inputs, kc, Radii; nk=3, Nq=100, F=[1.], R_min=.01, rank_tol=1e-8)

    Compute eigenfrequency with dispersion, using contour integration. BC's set
    by inputs.bc

    Contour is centered on kc, Radii = (x-diameter, y-diameter).

    nk is an upper bound on the number of eigenfrequencies contained in the contour.

    Nq is the number of contour quadrature points.

    Parallelizes quadrature.
"""
function computeK_NL2_parallel(inputs::InputStruct, kc::Union{Complex128,Float64,Int},
    Radii::Tuple{Float64,Float64}; nk::Int=3, Nq::Int=100, F::Array{Float64,1}=[1.],
    R_min::Float64=.01, rank_tol::Float64=1e-8)::Array{Complex128,1}

    N_ext = prod(inputs.N_ext); ε_sm = inputs.ε_sm; F_sm = inputs.F_sm
    D₀ = inputs.D₀; γ⟂ = inputs.γ⟂; k₀ = inputs.k₀

    k = Complex128(kc)

    ∇² = laplacian(k,inputs)

    M = rand(N_ext,nk)
    ϕ = 2π*(0:1/Nq:(1-1/Nq))
    Ω = k + Radii[1]*cos.(ϕ) + 1im*Radii[2]*sin.(ϕ)

    θ =  angle(k₀-1im*γ⟂-k)
    flag = abs(k₀-1im*γ⟂-k) < rad(Radii[1],Radii[2],θ)
    if flag
        RR = 2*R_min
        ΩΩ = k₀-1im*γ⟂ + (RR/2)*cos.(ϕ) + 1im*(RR/2)*sin.(ϕ)
    end

    AA = @parallel (+) for i in 1:Nq

        k′ = Ω[i]
        k′² = k′^2

        if (i > 1) & (i < Nq)
            dk′ = (Ω[i+1]-Ω[i-1]  )/2
        elseif i == Nq
            dk′ = (Ω[1]  -Ω[end-1])/2
        elseif i == 1
            dk′ = (Ω[2]  -Ω[end]  )/2
        end

        ɛk′² = sparse(1:N_ext, 1:N_ext, ɛ_sm[:]*k′²            , N_ext, N_ext, +)
        χk′² = sparse(1:N_ext, 1:N_ext, D₀*(γ⟂/(k′-k₀+1im*γ⟂))*F.*F_sm[:]*k′², N_ext, N_ext, +)

        A  = (∇²+ɛk′²+χk′²)\M
        A₀ = A*dk′/(2π*1im)
        A₁ = A*k′*dk′/(2π*1im)

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
            ɛkk′² = sparse(1:N_ext, 1:N_ext, ɛ_sm[:]*kk′², N_ext, N_ext, +)
            χkk′² = sparse(1:N_ext, 1:N_ext, D₀*(γ⟂/(kk′-k₀+1im*γ⟂))*F.*F_sm[:]*kk′², N_ext, N_ext, +)

            AA  = (∇²+ɛkk′²+χkk′²)\M
            AA₀ = AA*dkk′/(2π*1im)
            AA₁ = AA*kk′*dkk′/(2π*1im)

            A₀ = A₀-AA₀
            A₁ = A₁-AA₁
        end

        [A₀ A₁]

    end

    A₀ = AA[:,1:nk]
    A₁ = AA[:,nk + (1:nk)]

    P = svdfact(A₀,thin = true)
    temp = find(P[:S] .< rank_tol)
    if isempty(temp)
        println("Error. Need more nevals")
        return [NaN]
    else
        k = temp[1]-1
    end

    B = (P[:U][:,1:k])'*A₁*(P[:Vt][1:k,:])'*diagm(1./P[:S][1:k])

    D,V = eig(B)

    return D
end # end of function computeK_NL2_parallel


"""
computePole_NL2_parallel(inputs, kc, Radii; np=3, Nq=100, F=[1.], R_min=.01, rank_tol=1e-8)

    Compute poles with dispersion, using contour integration and outgoing PML's.

    Contour is centered on kc, Radii = (x-diameter, y-diameter).

    np is an upper bound on the number of eigenfrequencies contained in the contour.

    Nq is the number of contour quadrature points.

    Parallelizes quadrature.
"""
function computePole_NL2_parallel(inputs1::InputStruct, k::Union{Complex128,Float64,Int},
    Radii::Tuple{Float64,Float64}; np::Int=3, Nq::Int=100, F::Array{Float64,1}=[1.],
    R_min::Float64=.01, rank_tol::Float64=1e-8)::Array{Complex{Float64},1}

    inputs = open_to_pml_out(inputs1)

    k = computeK_NL2_parallel(inputs, kc, Radii; nk=np, Nq=Nq, F=F, R_min=R_min, rank_tol=rank_tol)
end # end of function computePole_NL2_parallel


"""
computeZero_NL2_parallel(inputs, kc, Radii; nz=3, Nq=100, F=[1.], R_min=.01, rank_tol=1e-8)

    Compute zeros with dispersion, using contour integration and incoming PML's.

    Contour is centered on kc, Radii = (x-diameter, y-diameter).

    nz is an upper bound on the number of eigenfrequencies contained in the contour.

    Nq is the number of contour quadrature points.

    Parallelizes quadrature.
"""
function computeZero_NL2_parallel(inputs1::InputStruct, kc::Union{Complex128,Float64,Int},
    Radii::Tuple{Float64,Float64}; nz::Int=3, Nq::Int=100, F::Array{Float64,1}=[1.],
    R_min::Float64=.01, rank_tol::Float64=1e-8)::Array{Complex128,1}

    inputs = open_to_pml_in(inputs1)

    k = computeK_NL2_parallel(inputs, kc, Radii; nk=nz, Nq=Nq, F=F, R_min=R_min, rank_tol=rank_tol)
end # end of function computeZero_NL2_parallel


"""
computeUZR_NL2_parallel(inputs, kc, Radii; nz=3, Nq=100, F=[1.], R_min=.01, rank_tol=1e-8, direction = [1,0])

    Compute UZR's with dispersion, using contour integration and outoing/incoming PML's.

    Contour is centered on kc, Radii = (x-diameter, y-diameter).

    nz is an upper bound on the number of eigenfrequencies contained in the contour.

    Nq is the number of contour quadrature points.

    If direction[1] = +1, have right-going. If -1, left-going. If 0, leave bc's
    as indicated in inputs.

    Parallelizes quadrature.
"""
function computeUZR_NL2_parallel(inputs1::InputStruct, kc::Union{Complex128,Float64,Int},
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

    k = computeK_NL2_parallel(inputs, kc, Radii; nk=nu, Nq=Nq, F=F, R_min=R_min, rank_tol=rank_tol)
end # end of function computeUZR_NL2_parallel


################################################################################
### S-matrix routines
################################################################################


"""
defaults to running S only on workers, not on head node. Use computeS_parallel! for a little more control
"""
function computeS_parallel(inputs::InputStruct, k::Array{Complex128,1}; isNonLinear::Bool=false,
    channels::Array{Int,1}=Array(1:length(inputs.channels)), F::Array{Float64,1}=[1.], dispOpt::Bool = true, fileName::String = "",
    N::Int=1, N_Type::String="D")

    M = length(inputs.channels)

    if isempty(fileName)
        S = SharedArray{Complex128}((length(k),M,M,N), pids=workers())
    else
        S = SharedArray{Complex128}(abspath(fileName),(length(k),M,M,N), pids=workers(), mode="w+")
    end

    for i in 1:length(S)
        S[i]=1im*NaN
    end

    P = procs(S)
    r = Channel(length(P))
    for pp in 1:length(P)
        p = P[pp]
        @async put!(r, remotecall_fetch(computeS_parallel_core!, p, S, deepcopy(inputs), k;
                    channels=channels, isNonLinear=isNonLinear, F=F, dispOpt=dispOpt, N=N, N_Type=N_Type) )
    end

    return S,r
end # end of function computeS_parallel


"""
computeS_parallel!
"""
function computeS_parallel!(S::SharedArray, inputs::InputStruct, k::Array{Complex128,1}; channels::Array{Int64}=Array(1:length(inputs.channels)),
    isNonLinear::Bool=false, F::Array{Float64,1}=[1.], dispOpt::Bool=true, N::Int=1, N_Type::String="D")

    P = procs(S)
    r = Channel(length(P))
    for pp in 1:length(P)
       p = P[pp]
       @async remotecall_fetch(computeS_parallel_core!, p, S, deepcopy(inputs); channels=channels,
                   isNonLinear=isNonLinear, F=F, dispOpt=dispOpt, N=N, N_Type=N_Type)
    end
end # end of function computeS_parallel!


"""
computeS_parallel_core!
"""
function computeS_parallel_core!(S::SharedArray, inputs::InputStruct, k::Array{Complex128,1}; channels::Array{Int64}=Array(1:length(inputs.channels)),
    isNonLinear::Bool=false, F::Array{Float64,1}=[1.], dispOpt::Bool=true, N::Int=1, N_Type::String="D")::SharedArray

    nc = length(inputs.channels)
    nk = length(k)
    M = length(procs(S))

    idx = indexpids(S)

    nchunks = ceil(Int,sqrt(M))
    mchunks = floor(Int,M/nchunks)

    if idx == 0 || idx > nchunks*mchunks # This worker is not assigned a piece
        a_inds = 1:0
        k_inds = 10:0
    else
        ncsplits = [round(Int, s) for s in linspace(0, nc, nchunks+1)]
        nksplits = [round(Int, s) for s in linspace(0, nk, mchunks+1)]
        a_idx, k_idx = ind2sub(zeros(Int,nchunks,mchunks),idx)

        a_inds = Array(Int,ncsplits[a_idx]+1:ncsplits[a_idx+1])
        k_inds = Array(Int,nksplits[k_idx]+1:nksplits[k_idx+1])
    end

     S[k_inds,a_inds,:,:] = idx # = computeS(deepcopy(inputs), k[k_inds]; channels=a_inds,
            # isNonLinear=isNonLinear, F=F, dispOpt=dispOpt, N=N, N_Type=N_Type)

    return S
end # end of function computeS_parallel_core!


"""
P_wait(r)
"""
function P_wait(r::Channel)

   c = 0
   while c < r.sz_max
       take!(r)
       c += 1
   end

   return
end # end of function S_wait

################################################################################
