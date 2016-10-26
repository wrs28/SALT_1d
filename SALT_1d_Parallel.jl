function computePolesL!(inputs::Dict, ω, nTCFs::Int, F, eval::SharedArray) #No Line Pulling

    dx = inputs["dx"]
    x_ext = inputs["x_ext"]
    ∂_ext = inputs["∂_ext"]
    ℓ_ext = inputs["ℓ_ext"]
    N_ext = inputs["N_ext"]
    λ = inputs["λ"]
    x_inds = inputs["x_inds"]
    Γ_ext = inputs["Γ_ext"]
    ɛ_ext = inputs["ɛ_ext"]
    D₀ = inputs["D₀"]
    F_ext = inputs["F_ext"]

    r = whichRegion(x_ext, ∂_ext)

    ∇² = laplacian(ℓ_ext, N_ext, 1+1im*σ(x_ext,∂_ext,λ)/ω)

    Γ = zeros(N_ext,1)
    for dd in 2:length(∂_ext)-1
        δ,dummy1 = dirac_δ(x_ext,∂_ext[dd])
        Γ = Γ[:] + full(δ)/Γ_ext[dd]
    end

    ɛΓ⁻¹ = sparse(1:N_ext, 1:N_ext, 1./(ɛ_ext[r]+Γ[:]-1im*D₀.*F.*F_ext[r]), N_ext, N_ext, +)

    (ω²,ψ_ext,nconv,niter,nmult,resid) = eigs(-ɛΓ⁻¹*∇²,which = :LM, nev = nTCFs, sigma = ω^2+1.e-5)

    for idx in 1:nTCFs
        eval[idx] = sqrt(ω²)[idx]
    end

    return eval

end # end of function computePolesL!



function computeS_parallel_core!(S::SharedArray, inputs::Dict, N::Int, M::Int, isNonLinear::Bool, F, displayOpt)

    a_min = .001
    D₀ = copy(inputs["D₀"])
    D = linspace(0,D₀,N)
    a₀ = inputs["a"]
    a = linspace(a_min,a₀,M)

    x_inds = inputs["x_inds"]

    ψ₊ = Array(Complex128,length(inputs["x_ext"]))
    ψ₋ = Array(Complex128,length(inputs["x_ext"]))

    function ωa_range(q::SharedArray) #cannibalized from Julia docs page
        idx = indexpids(q)
        if idx == 0 # This worker is not assigned a piece
            return 1:0, 1:0
        end
        nchunks = length(procs(q))
        splits = [round(Int, s) for s in linspace(0,size(q,3)*size(q,5),nchunks+1)]
        ind2sub((size(q,3),size(q,5)),splits[idx]+1:splits[idx+1])
    end

    ω_inds, a_inds = ωa_range(S)

    for k in 1:length(ω_inds)

        ω = inputs["ω"][ω_inds[k]]
        inputs["a"] = a[a_inds[k]]

        inputs["D₀"] = 0.0

        ψ₊ = solve_scattered(inputs,ω,inputs["xᵨ₊"],+,isNonLinear = false)
        ψ₋ = solve_scattered(inputs,ω,inputs["xᵨ₋"],-,isNonLinear = false)

        if (k/25 == round(k/25)) & displayOpt
            printfmtln("S-matrix {1:5.1f}% done. Worker {2:2d} of {3:d}.",100*(k-1)/length(ω_inds),indexpids(S),length(procs(S)))
        end

        for j in 1:N

            inputs["D₀"] = D[j]

            if isnan(ψ₊[1])
                ψ₊ = solve_scattered(inputs,ω,inputs["xᵨ₊"],+,isNonLinear = isNonLinear, F = F)
            else
                ψ₊ = solve_scattered(inputs,ω,inputs["xᵨ₊"],+,isNonLinear = isNonLinear, ψ_init = ψ₊, F = F)
            end

            if isnan(ψ₋[1])
                ψ₋ = solve_scattered(inputs,ω,inputs["xᵨ₋"],-,isNonLinear = isNonLinear, F = F)
            else
                ψ₋ = solve_scattered(inputs,ω,inputs["xᵨ₋"],-,isNonLinear = isNonLinear, ψ_init = ψ₋, F = F)
            end

            S[1,1,ω_inds[k],j,a_inds[k]] = ψ₊[x_inds[1]]/inputs["a"]
            S[2,1,ω_inds[k],j,a_inds[k]] = ψ₊[x_inds[end]]/inputs["a"]
            S[1,2,ω_inds[k],j,a_inds[k]] = ψ₋[x_inds[1]]/inputs["a"]
            S[2,2,ω_inds[k],j,a_inds[k]] = ψ₋[x_inds[end]]/inputs["a"]

        end
    end

end # end of function omputeS_parallel_core


function computeZerosL_parallel_core!(Ω::SharedArray,Ω₊::SharedArray,Ω₋::SharedArray,inputs::Dict,ω,N::Int,M::Int,displayOpt)

    a_min = .001

    function ωa_range(q::SharedArray) #cannibalized from Julia docs page
        idx = indexpids(q)
        if idx == 0 # This worker is not assigned a piece
            return 1:0, 1:0
        end
        nchunks = length(procs(q))
        splits = [round(Int, s) for s in linspace(0,size(q,1)*size(q,3),nchunks+1)]
        ind2sub((size(q,1),size(q,3)),splits[idx]+1:splits[idx+1])
    end

    ω_inds, a_inds = ωa_range(Ω)

    D = linspace(0,inputs["D₀"],N)
    a = linspace(a_min,inputs["a"],M)

    ψ₊ = zeros(Complex128,inputs["N_ext"])
    ψ₋ = copy(ψ₊)

    if !isempty(ω_inds)
        for k in 1:length(a_inds)
            inputs["a"] = a[a_inds[k]]
            if (k/25 == round(k/25)) & displayOpt
                printfmtln("Zeros {1:5.1f}% done. Worker {2:2d} of {3:d}.",100*(k-1)/length(ω_inds),indexpids(Ω),length(procs(Ω)))
            end
            for i in 1:N

                inputs["D₀"] = D[i]

                if i == 1
                    ψ₊ = solve_scattered(inputs,inputs["ω₀"],inputs["xᵨ₊"],+; isNonLinear = false, F = 0.)
                    ψ₋ = solve_scattered(inputs,inputs["ω₀"],inputs["xᵨ₋"],-; isNonLinear = false, F = 0.) 

                    Ω[ω_inds[k],i,a_inds[k]],dummy1,dummy2,conv = computeZerosL(inputs,conj(ω[ω_inds[k]]),F=0.)
                    Ω₊[ω_inds[k],i,a_inds[k]] = Ω[ω_inds[k],1,a_inds[k]]
                    Ω₋[ω_inds[k],i,a_inds[k]] = Ω[ω_inds[k],1,a_inds[k]]
                else
                    ψ₊ = solve_scattered(inputs,inputs["ω₀"],inputs["xᵨ₊"],+; isNonLinear = true, ψ_init = ψ₊)
                    ψ₋ = solve_scattered(inputs,inputs["ω₀"],inputs["xᵨ₋"],-; isNonLinear = true, ψ_init = ψ₋)

                    Ω[ω_inds[k],i,a_inds[k]] ,dummy1,dummy2,conv = computeZerosL(inputs,Ω[ω_inds[k],i-1,a_inds[k]] ,F=1.)  
                    Ω₊[ω_inds[k],i,a_inds[k]],dummy1,dummy2,conv = computeZerosL(inputs,Ω₊[ω_inds[k],i-1,a_inds[k]],F=1./(1+abs2(ψ₊[:])))
                    Ω₋[ω_inds[k],i,a_inds[k]],dummy1,dummy2,conv = computeZerosL(inputs,Ω₋[ω_inds[k],i-1,a_inds[k]],F=1./(1+abs2(ψ₋[:])))
                end
            end
        end
    end

end # end of function computeZerosL_parallel_core!


function computePolesL_parallel_core!(Ω::SharedArray,Ω₊::SharedArray,Ω₋::SharedArray,inputs::Dict,ω,N::Int,M::Int,displayOpt)

    a_min = .001

    function ωa_range(q::SharedArray) #cannibalized from Julia docs page
        idx = indexpids(q)
        if idx == 0 # This worker is not assigned a piece
            return 1:0, 1:0
        end
        nchunks = length(procs(q))
        splits = [round(Int, s) for s in linspace(0,size(q,1)*size(q,3),nchunks+1)]
        ind2sub((size(q,1),size(q,3)),splits[idx]+1:splits[idx+1])
    end

    ω_inds, a_inds = ωa_range(Ω)

    D = linspace(0,inputs["D₀"],N)
    a = linspace(a_min,inputs["a"],M)

    ψ₊ = zeros(Complex128,inputs["N_ext"])
    ψ₋ = copy(ψ₊)

    if !isempty(ω_inds)

        for k in 1:length(a_inds)
            inputs["a"] = a[a_inds[k]]
            for i in 1:N
                if (k/25 == round(k/25)) & displayOpt
                    printfmtln("Zeros {1:5.1f}% done. Worker {2:2d} of {3:d}.",100*(k-1)/length(ω_inds),indexpids(Ω),length(procs(Ω)))
                end
                inputs["D₀"] = D[i]

                if i == 1
                    ψ₊ = solve_scattered(inputs,inputs["ω₀"],inputs["xᵨ₊"],+; isNonLinear = false, F = 0.)
                    ψ₋ = solve_scattered(inputs,inputs["ω₀"],inputs["xᵨ₋"],-; isNonLinear = false, F = 0.) 

                    Ω[ω_inds[k],i,a_inds[k]] = ω[ω_inds[k]]
                    Ω₊[ω_inds[k],i,a_inds[k]] = ω[ω_inds[k]]
                    Ω₋[ω_inds[k],i,a_inds[k]] = ω[ω_inds[k]]
                else
                    ψ₊ = solve_scattered(inputs,inputs["ω₀"],inputs["xᵨ₊"],+; isNonLinear = true, ψ_init = ψ₊)
                    ψ₋ = solve_scattered(inputs,inputs["ω₀"],inputs["xᵨ₋"],-; isNonLinear = true, ψ_init = ψ₋)

                    Ω[ω_inds[k],i,a_inds[k]]  = computePolesL(inputs,Ω[ω_inds[k],i-1,a_inds[k]] ,1, F = 1.)[1][1] 
                    Ω₊[ω_inds[k],i,a_inds[k]] = computePolesL(inputs,Ω₊[ω_inds[k],i-1,a_inds[k]],1, F = 1./(1+abs2(ψ₊[:])))[1][1]
                    Ω₋[ω_inds[k],i,a_inds[k]] = computePolesL(inputs,Ω₋[ω_inds[k],i-1,a_inds[k]],1, F = 1./(1+abs2(ψ₋[:])))[1][1]

                end
            end
        end

    end

end # end of function computePolesL_parallel_core!

##########################################################################################

module Parallel

export computeS_parallel, computePolesL_parallel, computeZerosL_parallel

function computeS_parallel(inputs::Dict,N::Int,M::Int; isNonLinear=false, F=1.)
# N is the number of steps to go from D0 = 0 to given D0
# M is the number of steps to go from a=a_min to given a

    S = SharedArray(Complex128,(2,2,length(inputs["ω"]),N,M))

    @sync begin
        for p in procs(S)
            @async remotecall_wait(p, computeS_parallel_core!, S, copy(inputs), N, M, isNonLinear, F, true)
        end
    end  

    return sdata(S)

end



function computePolesL_parallel(inputs::Dict,nCF::Int,N::Int,M::Int)

    a_min = .001

    ω,dummy = computePolesL(inputs,inputs["ω₀"],nCF, F=0.)
    Ω = SharedArray(Complex128,(nCF,N,M))
    Ω₊ = SharedArray(Complex128,(nCF,N,M))
    Ω₋ = SharedArray(Complex128,(nCF,N,M))

    @sync begin
        for p in procs(Ω)
            @async remotecall_wait(p, computePolesL_parallel_core!, Ω, Ω₊, Ω₋, copy(inputs), ω, N, M)
        end
    end  

    return sdata(Ω), sdata(Ω₊), sdata(Ω₋)

end



function computeZerosL_parallel(inputs::Dict,nCF::Int,N::Int,M::Int;displayOpt = true)

    ω,dummy = computePolesL(inputs,inputs["ω₀"],nCF, F=0.)
    Ω = SharedArray(Complex128,(nCF,N,M))
    Ω₊ = SharedArray(Complex128,(nCF,N,M))
    Ω₋ = SharedArray(Complex128,(nCF,N,M))

    @sync begin
        for p in procs(Ω)
            @async remotecall_wait(p, computeZerosL_parallel_core!, Ω, Ω₊, Ω₋, copy(inputs), ω, N, M, displayOpt)
        end
    end  

    return sdata(Ω), sdata(Ω₊), sdata(Ω₋)

end

end # end of Module SALT_1D.Parallel