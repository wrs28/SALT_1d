module SALT_1d_Parallel

export computeS_parallel, computeS_parallel! #, computePolesL_parallel, computeZerosL_parallel

using SALT_1d



function computeS_parallel(inputs::Dict; N=10, N_Type="D", isNonLinear=false, F=1., dispOpt = true, fileName = "")
    # N is the number of steps to go from D0 = 0 to given D0 or a=0 to a, whichever is specified in N_Type
    # defaults to running S only on workers, not on head node. Use computeS_parallel! for a little more control

    if isempty(fileName)
        S = SharedArray(Complex128,(2,2,length(inputs["k"]),N), pids=workers())
    else
        S = SharedArray(abspath(fileName),Complex128,(2,2,length(inputs["k"]),N), pids=workers(), mode="w+")
    end
    
    for i in 1:length(S)
        S[i]=1im*NaN
    end

    P = procs(S)
    r = Channel(length(P))
    for pp in 1:length(P)
        p = P[pp]
        @async put!(r, remotecall_fetch(computeS_parallel_core!, p, S, deepcopy(inputs); N=N, N_Type=N_Type, isNonLinear=isNonLinear, F=F, dispOpt=dispOpt))
    end

    return S,r

end # end of function computeS_parallel



function computeS_parallel!(S::SharedArray,inputs::Dict; N=10, N_Type="D", isNonLinear=false, F=1., dispOpt = true)
# N is the number of steps to go from D0 = 0 to given D0 or a=0 to a

    P = procs(S)
    r = Channel(length(P))
    for pp in 1:length(P)
        p = P[pp]
        @async put!(r, remotecall_fetch(computeS_parallel_core!, p, S, deepcopy(inputs); N=N, N_Type=N_Type, isNonLinear=isNonLinear, F=F, dispOpt=dispOpt))
    end

end # end of function computeS_parallel!



function computeS_parallel_core!(S::SharedArray, inputs::Dict; N=10, N_Type="D", isNonLinear=false, F=1., dispOpt=true)

    idx = indexpids(S)
    nchunks = length(procs(S))
    splits = [round(Int, s) for s in linspace(0,size(S,3),nchunks+1)] #define boudnaries between ranges
    k_inds = splits[idx]+1:splits[idx+1] #return ranges

    inputs1 = deepcopy(inputs)
    inputs1["k"] = inputs["k"][k_inds]
    
    S[:,:,k_inds,:] = computeS(inputs1; N=N, N_Type=N_Type, isNonLinear=isNonLinear, F=F, dispOpt=dispOpt)
    
end # end of function computeS_parallel_core













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
#    ∇² = laplacian(ℓ_ext, N_ext, 1+1im*σ(x_ext,∂_ext,λ)/k)
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

end # end of Module SALT_1D.Parallel