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
    truncate::Bool, ψ_init::Array{Complex128,1}, dispOpt::Bool)::SharedArray

    nk = length(k)
    dims = tuple(nk, length.(field_vals)...)
    K = SharedArray{Complex128}(dims)
    ψ = Complex128[]

    inputs1 = deepcopy(inputs)
    if dispOpt
        println("Computing dimension 1")
    end
    for i in 1:nk
        # for j in [1]#:length(field_vals[1])
            if !isempty(size(getfield(inputs1,fields[1])))
                vals_temp = getfield(inputs1,fields[1])
                vals_temp[field_inds[1]] = field_vals[1][j]
                updateInputs!(inputs1,fields[1],vals_temp)
            else
                updateInputs!(inputs1,fields[1],field_vals[1][j])
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
    for d in 2:ndims(K)
        if dispOpt
            println("Computing dimension $(d-1)")
        end
        @sync for p in procs(K)
            @async remotecall_fetch(computeK_L_core!, p, K, inputs, fields, field_inds,
                                    field_vals, d, F, truncate, ψ_init)
        end
    end

    return K
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
    ψ_init::Array{Complex128,1}=Complex128[], dispOpt::Bool=false)::SharedArray

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
function computeS_parallel(inputs::InputStruct; isNonLinear::Bool=false,
    F::Array{Float64,1}=[1.], dispOpt::Bool = true, fileName::String = "",
    N::Int=1, N_Type::String="D")

    if isempty(fileName)
        S = SharedArray(Complex128,(length(inputs.k),2,2,N), pids=workers())
    else
        S = SharedArray(abspath(fileName),Complex128,(length(inputs.k),2,2,N), pids=workers(), mode="w+")
    end

    for i in 1:length(S)
        S[i]=1im*NaN
    end

    P = procs(S)
    r = Channel(length(P))
    for pp in 1:length(P)
        p = P[pp]
        @async put!(r, remotecall_fetch(computeS_parallel_core!, p, S, deepcopy(inputs);
                    isNonLinear=isNonLinear, F=F, dispOpt=dispOpt, N=N, N_Type=N_Type) )
    end

    return S,r
end # end of function computeS_parallel


"""
computeS_parallel!
"""
function computeS_parallel!(S::SharedArray,inputs::InputStruct; isNonLinear::Bool=false,
    F::Array{Float64,1}=[1.], dispOpt::Bool=true, N::Int=1, N_Type::String="D")

    P = procs(S)
    r = Channel(length(P))
    for pp in 1:length(P)
       p = P[pp]
       @async put!(r, remotecall_fetch(computeS_parallel_core!, p, S, deepcopy(inputs);
                   isNonLinear=isNonLinear, F=F, dispOpt=dispOpt, N=N, N_Type=N_Type))
    end
end # end of function computeS_parallel!


"""
computeS_parallel_core!
"""
function computeS_parallel_core!(S::SharedArray, inputs::InputStruct; isNonLinear::Bool=false,
    F::Array{Float64,1}=[1.], dispOpt::Bool=true, N::Int=1, N_Type::String="D")::SharedArray

    idx = indexpids(S)
    nchunks = length(procs(S))
    splits = [round(Int, s) for s in linspace(0,size(S,1),nchunks+1)] #define boudnaries between ranges
    k_inds = splits[idx]+1:splits[idx+1] #return ranges

    inputs1 = deepcopy(inputs)
    updateInputs(inputs1,:k,[inputs.k[k_inds]])

    S[k_inds,:,:,:] = computeS(inputs1; isNonLinear=isNonLinear, F=F, dispOpt=dispOpt,
                                    N=N, N_Type=N_Type)
    return S
end # end of function computeS_parallel_core


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













#function computePolesL!(inputs::Dict, k::Number, nPoles::Int; F=1., eval::SharedArray)
#     #No Line Pulling
#
#    ## DEFINITIONS BLOCK ##
#    dx = inputs["dx"]
#    x_ext = inputs["x_ext"]
#    ∂_ext = inputs["∂_ext"]
#    ℓ_ext = inputs["ℓ_ext"]
#    N_ext = inputs["N_ext"]
#    λ = inputs["λ"]
#    x_inds = inputs["x_inds"]
#    Γ_ext = inputs["Γ_ext"]
#    ɛ_ext = inputs["ɛ_ext"]
#    D₀ = inputs["D₀"]
#    F_ext = inputs["F_ext"]
#    ## END OF DEFINITIONS BLOCK ##
#
#
#    r = whichRegion(x_ext, ∂_ext)
#
#    ∇² = laplacian(ℓ_ext, N_ext, 1+1im*σ(x_ext,∂_ext)/real(k))
#
#    Γ = zeros(N_ext,1)
#    for dd in 2:length(∂_ext)-1
#        δ,dummy1 = dirac_δ(x_ext,∂_ext[dd])
#        Γ = Γ[:] + full(δ)/Γ_ext[dd]
#    end
#
#    ɛΓ⁻¹ = sparse(1:N_ext, 1:N_ext, 1./(ɛ_ext[r]+Γ[:]-1im*D₀.*F.*F_ext[r]), N_ext, N_ext, +)
#
#    (k²,ψ_ext,nconv,niter,nmult,resid) = eigs(-ɛΓ⁻¹*∇²,which = :LM, nev = nPoles, sigma = k^2+1.e-5)
#
#    ψ = zeros(Complex128,length(inputs["x_ext"]),nPoles)
#
#    inds1 = inputs["x_inds"][1]:inputs["x_inds"][end]
#    r1 = whichRegion(inputs["x"],inputs["∂"])
#    for ii = 1:length(k²)
#        N = trapz(ψ_ext[inds1,ii].*(inputs["ɛ"][r1]+Γ[inds1]-1im*D₀.*F.*inputs["F"][r1]).*ψ_ext[inds1,ii],dx)
#        ψ_ext[:,ii] = ψ_ext[:,ii]/sqrt(N)
#        ψ[:,ii] = ψ_ext[:,ii]
#    end
#
#    for idx in 1:nPoles
#        eval[idx] = sqrt(k²)[idx]
#    end
#
#    return eval
#
#end # end of function computePolesL!
#
#
#
#
#
#
#function computeZerosL_parallel_core!(Ω::SharedArray,Ω₊::SharedArray,Ω₋::SharedArray,inputs::Dict,ω,N::Int,M::Int,displayOpt)
#
#    a_min = .001
#
#    function ωa_range(q::SharedArray) #cannibalized from Julia docs page
#        idx = indexpids(q)
#        if idx == 0 # This worker is not assigned a piece
#            return 1:0, 1:0
#        end
#        nchunks = length(procs(q))
#        splits = [round(Int, s) for s in linspace(0,size(q,1)*size(q,3),nchunks+1)]
#        ind2sub((size(q,1),size(q,3)),splits[idx]+1:splits[idx+1])
#    end
#
#    ω_inds, a_inds = ωa_range(Ω)
#
#    D = linspace(0,inputs["D₀"],N)
#    a = linspace(a_min,inputs["a"],M)
#
#    ψ₊ = zeros(Complex128,inputs["N_ext"])
#    ψ₋ = copy(ψ₊)
#
#    if !isempty(ω_inds)
#        for k in 1:length(a_inds)
#            inputs["a"] = a[a_inds[k]]
#            if (k/25 == round(k/25)) & displayOpt
#                printfmtln("Zeros {1:5.1f}% done. Worker {2:2d} of {3:d}.",100*(k-1)/length(ω_inds),indexpids(Ω),length(procs(Ω)))
#            end
#            for i in 1:N
#
#                inputs["D₀"] = D[i]
#
#                if i == 1
#                    ψ₊ = solve_scattered(inputs,inputs["ω₀"],inputs["xᵨ₊"],+; isNonLinear = false, F = 0.)
#                    ψ₋ = solve_scattered(inputs,inputs["ω₀"],inputs["xᵨ₋"],-; isNonLinear = false, F = 0.)
#
#                    Ω[ω_inds[k],i,a_inds[k]],dummy1,dummy2,conv = computeZerosL(inputs,conj(ω[ω_inds[k]]),F=0.)
#                    Ω₊[ω_inds[k],i,a_inds[k]] = Ω[ω_inds[k],1,a_inds[k]]
#                    Ω₋[ω_inds[k],i,a_inds[k]] = Ω[ω_inds[k],1,a_inds[k]]
#                else
#                    ψ₊ = solve_scattered(inputs,inputs["ω₀"],inputs["xᵨ₊"],+; isNonLinear = true, ψ_init = ψ₊)
#                    ψ₋ = solve_scattered(inputs,inputs["ω₀"],inputs["xᵨ₋"],-; isNonLinear = true, ψ_init = ψ₋)
#
#                    Ω[ω_inds[k],i,a_inds[k]] ,dummy1,dummy2,conv = computeZerosL(inputs,Ω[ω_inds[k],i-1,a_inds[k]] ,F=1.)
#                    Ω₊[ω_inds[k],i,a_inds[k]],dummy1,dummy2,conv = computeZerosL(inputs,Ω₊[ω_inds[k],i-1,a_inds[k]],F=1./(1+abs2(ψ₊[:])))
#                    Ω₋[ω_inds[k],i,a_inds[k]],dummy1,dummy2,conv = computeZerosL(inputs,Ω₋[ω_inds[k],i-1,a_inds[k]],F=1./(1+abs2(ψ₋[:])))
#                end
#            end
#        end
#    end
#
#end # end of function computeZerosL_parallel_core!
#
#
#function computePolesL_parallel_core!(Ω::SharedArray,Ω₊::SharedArray,Ω₋::SharedArray,inputs::Dict,ω,N::Int,M::Int,displayOpt)
#
#    a_min = .001
#
#    function ωa_range(q::SharedArray) #cannibalized from Julia docs page
#        idx = indexpids(q)
#        if idx == 0 # This worker is not assigned a piece
#            return 1:0, 1:0
#        end
#        nchunks = length(procs(q))
#        splits = [round(Int, s) for s in linspace(0,size(q,1)*size(q,3),nchunks+1)]
#        ind2sub((size(q,1),size(q,3)),splits[idx]+1:splits[idx+1])
#    end
#
#    ω_inds, a_inds = ωa_range(Ω)
#
#    D = linspace(0,inputs["D₀"],N)
#    a = linspace(a_min,inputs["a"],M)
#
#    ψ₊ = zeros(Complex128,inputs["N_ext"])
#    ψ₋ = copy(ψ₊)
#
#    if !isempty(ω_inds)
#
#        for k in 1:length(a_inds)
#            inputs["a"] = a[a_inds[k]]
#            for i in 1:N
#                if (k/25 == round(k/25)) & displayOpt
#                    printfmtln("Zeros {1:5.1f}% done. Worker {2:2d} of {3:d}.",100*(k-1)/length(ω_inds),indexpids(Ω),length(procs(Ω)))
#                end
#                inputs["D₀"] = D[i]
#
#                if i == 1
#                    ψ₊ = solve_scattered(inputs,inputs["ω₀"],inputs["xᵨ₊"],+; isNonLinear = false, F = 0.)
#                    ψ₋ = solve_scattered(inputs,inputs["ω₀"],inputs["xᵨ₋"],-; isNonLinear = false, F = 0.)
#
#                    Ω[ω_inds[k],i,a_inds[k]] = ω[ω_inds[k]]
#                    Ω₊[ω_inds[k],i,a_inds[k]] = ω[ω_inds[k]]
#                    Ω₋[ω_inds[k],i,a_inds[k]] = ω[ω_inds[k]]
#                else
#                    ψ₊ = solve_scattered(inputs,inputs["ω₀"],inputs["xᵨ₊"],+; isNonLinear = true, ψ_init = ψ₊)
#                    ψ₋ = solve_scattered(inputs,inputs["ω₀"],inputs["xᵨ₋"],-; isNonLinear = true, ψ_init = ψ₋)
#
#                    Ω[ω_inds[k],i,a_inds[k]]  = computePolesL(inputs,Ω[ω_inds[k],i-1,a_inds[k]] ,1, F = 1.)[1][1]
#                    Ω₊[ω_inds[k],i,a_inds[k]] = computePolesL(inputs,Ω₊[ω_inds[k],i-1,a_inds[k]],1, F = 1./(1+abs2(ψ₊[:])))[1][1]
#                    Ω₋[ω_inds[k],i,a_inds[k]] = computePolesL(inputs,Ω₋[ω_inds[k],i-1,a_inds[k]],1, F = 1./(1+abs2(ψ₋[:])))[1][1]
#
#                end
#            end
#        end
#
#    end
#
#end # end of function computePolesL_parallel_core!
#function computePolesL_parallel(inputs::Dict, nPoles::Int, N::Int, M::Int)
#
#    a_min = .001
#
#    ω,dummy = computePolesL(inputs, inputs["k₀"], nPoles, F=0.)
#    Ω = SharedArray(Complex128,(nCF,N,M))
#    Ω₊ = SharedArray(Complex128,(nCF,N,M))
#    Ω₋ = SharedArray(Complex128,(nCF,N,M))
#
#    @sync begin
#        for p in procs(Ω)
#            @async remotecall_wait(p, computePolesL_parallel_core!, Ω, Ω₊, Ω₋, copy(inputs), ω, N, M)
#        end
#    end
#
#    return sdata(Ω), sdata(Ω₊), sdata(Ω₋)
#
#end
#
#
#
#function computeZerosL_parallel(inputs::Dict, nPoles::Int, N::Int, M::Int; displayOpt = true)
#
#    ω,dummy = computePolesL(inputs,inputs["ω₀"],nCF, F=0.)
#    Ω = SharedArray(Complex128,(nCF,N,M))
#    Ω₊ = SharedArray(Complex128,(nCF,N,M))
#    Ω₋ = SharedArray(Complex128,(nCF,N,M))
#
#    @sync begin
#        for p in procs(Ω)
#            @async remotecall_wait(p, computeZerosL_parallel_core!, Ω, Ω₊, Ω₋, copy(inputs), ω, N, M, displayOpt)
#        end
#    end
#
#    return sdata(Ω), sdata(Ω₊), sdata(Ω₋)
#
#end
