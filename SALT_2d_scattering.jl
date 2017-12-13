"""
ψ, ϕ, A, inputs = compute_scatter(inputs, k; isNonLinear=false, dispOpt=false, ψ_init=[],
    F=[1.], truncate=false, A=[], fileName="", ftol=2e-8, iter=750)

    Solves inhomogeneous problem with source defined in incident wave file.

    k is the injection frequency.

    A is a factorized wave operator.
"""
function compute_scatter(inputs::InputStruct, K::Union{Complex128,Float64,Int};
    isNonLinear::Bool=false, dispOpt::Bool = false, ψ_init::Array{Complex128,1}=Complex128[],
    F::Array{Float64,1}=[1.], truncate::Bool=false,
    A::Base.SparseArrays.UMFPACK.UmfpackLU=lufact(speye(1,1)), fileName::String = "",
    ftol::Float64=2e-8, iter::Int=150)::
    Tuple{Array{Complex128,1},Array{Complex128,1},Base.SparseArrays.UMFPACK.UmfpackLU,InputStruct}

    k = complex(1.0*K)

    if !isNonLinear
        ψ, φ, A, inputs = compute_scatter_linear(inputs, k; A=A, F=F)
    elseif isNonLinear
        ψ, φ, A, inputs = compute_scatter_nonlinear(inputs, k; A=A, F=F, dispOpt=dispOpt, ψ_init=ψ_init, ftol=ftol, iter=iter)
    end

    if !isempty(fileName)
        if truncate
            fid = open(fileName,"w")
            serialize(fid, (ψ[inputs.x̄_inds], φ[inputs.x̄_inds], inputs, k) )
            close(fid)
        else
            fid = open(fileName,"w")
            serialize(fid, (ψ, φ, inputs, k) )
            close(fid)
        end
    end

    if truncate
        return ψ[inputs.x̄_inds], φ[inputs.x̄_inds], A, inputs
    else
        return ψ, φ, A, inputs
    end
end # end of function computePsi


"""
ψ, ϕ, A, inputs = compute_scatter_linear(inputs, k; A=[], F=[1.])

    Solves linear inhomogeneous problem with sources determined by choice
    of boundary conditions.

    k is the injection frequency.

    A is a factorized wave operator.
"""
function compute_scatter_linear(inputs1::InputStruct, k::Complex128;
    A::Base.SparseArrays.UMFPACK.UmfpackLU=lufact(speye(1,1)),
    F::Array{Float64,1}=[1.])#::Tuple{Array{Complex128,1},Array{Complex128,1},Base.SparseArrays.UMFPACK.UmfpackLU,InputStruct}

    inputs = open_to_pml_out(inputs1)

    k²= k^2

    j, ∇², φ₊, φ₋ = synthesize_source(inputs, k)

    if (A.m == A.n == 1)
        N = prod(inputs.N_ext); ε_sm = inputs.ε_sm; F_sm = inputs.F_sm
        ɛk² = sparse(1:N, 1:N, ɛ_sm[:]*k², N, N, +)
        χk² = sparse(1:N, 1:N, inputs.D₀*γ(k,inputs)*F.*F_sm[:]*k², N, N, +)
        A = factorize(∇²+ɛk²+χk²)
    end

    ψ = A\j

    return ψ-φ₊, φ₊, A, inputs
end # end of function computePsi_linear


"""
ψ, ϕ, A, inputs = computePsi(inputs, k; dispOpt=false, ψ_init=[],
    F=[1.], A=[], fileName="", ftol=2e-8, iter=750)

    Solves inhomogeneous problem with source defined in incident wave file.

    k is the injection frequency.

    A is a factorized wave operator.
"""
function compute_scatter_nonlinear(inputs1::InputStruct, k::Complex128;
    dispOpt::Bool = false, ψ_init::Array{Complex128,1}=Complex128[], F::Array{Float64,1}=[1.],
    A::Base.SparseArrays.UMFPACK.UmfpackLU=lufact(speye(1,1)),
    ftol::Float64=2e-8, iter::Int=150)::Tuple{Array{Complex128,1},Array{Complex128,1},Base.SparseArrays.UMFPACK.UmfpackLU,InputStruct}

        inputs = open_to_pml(inputs1)

        j, ∇², φ₊, φ₋ = createJ(inputs, k, m)

        N = prod(inputs.N_ext); ε_sm = inputs.ε_sm; k²= k^2;
        D₀ = inputs.D₀; F_sm = inputs.F_sm
        ɛk² = sparse(1:N, 1:N, ɛ_sm[:]*k², N, N, +)
        χk² = sparse(1:N, 1:N, D₀*γ(k,inputs)*F.*F_sm[:]*k², N, N, +)

        f!(Ψ, fvec) = scattering_residual(Ψ, fvec, j, ∇², εk², k, inputs)
        jac!(Ψ, jacarray) = scattering_jacobian(Ψ, jacarray, j, ∇², εk², k, inputs)
        df = DifferentiableSparseMultivariateFunction(f!, jac!)

        Ψ_init = Array{Float64}(2*length(j))
        Ψ_init[1:length(j)]     = real(ψ_init)
        Ψ_init[length(j)+1:2*length(j)] = imag(ψ_init)

        z = nlsolve(df, Ψ_init, show_trace=dispOpt, ftol=ftol, iterations=iter)

        if converged(z)
            ψ = z.zero[1:length(j)] + 1im*z.zero[length(j)+1:2*length(j)]
        else
            ψ = NaN*ψ
            println("Warning, solve_scattered did not converge. Returning NaN.")
        end

        if !isempty(fileName)
            if truncate
                fid = open(fileName,"w")
                serialize(fid, ((ψ-φ₊)[inputs.x̄_inds], φ₊[inputs.x̄_inds], inputs, k) )
                close(fid)
            else
                fid = open(fileName,"w")
                serialize(fid, ((ψ-φ₊), φ₊, inputs, k) )
                close(fid)
            end
        end

        if truncate
            return (ψ-φ₊)[inputs.x̄_inds], φ₊[inputs.x̄_inds], A, inputs
        else
            return ψ-φ₊, φ₊, A, inputs
        end
end # end of function computePsi_nonlinear


"""
S =  computeS(inputs; isNonLinear=false, F=[1.], dispOpt = true,
                                    N=1, N_Type="D", ψ_init = [], fileName = "")

    N is the number of steps to go from D0 = 0 to given D0
"""
function computeS(inputs::InputStruct,
    k::Union{Array{Complex128,1},Array{Float64,1},Complex128,Float64,Int};
    isNonLinear::Bool=false, F::Array{Float64,1}=[1.], dispOpt::Bool=true,
    fileName::String = "", N::Int=1, N_Type::String="D",
    ψ_init::Array{Complex128,1}=Complex128[])::Array{Complex128,4}

    if isempty(k)
        K = Complex128[k]
    else
        K = complex(1.0*k)
    end

    if !isNonLinear
        S = computeS_linear(inputs, K; F=F, dispOpt=dispOpt, fileName=fileName)
    elseif isNonLinear
        S = computeS_nonlinear(inputs, K; N=N, N_Type=N_Type, isNonLinear=isNonLinear,
                            F=F, dispOpt=dispOpt, ψ_init=ψ_init, fileName=fileName)
    end

    return S
end # end of fuction computeS


"""
S =  computeS_linear(inputs; F=[1.], dispOpt=true, fileName = "")
"""
function computeS_linear(inputs::InputStruct, k::Array{Complex128,1};
    F::Array{Float64,1}=[1.], dispOpt::Bool=true,
    fileName::String = "")::Array{Complex128,4}

    M = length(inputs.channels)
    S = NaN*ones(Complex128,length(inputs.k),M,M,1)
    a_original = inputs.a
    a = zeros(Complex128,M)
    ψ₋ = Array{Complex128}(1)

    for ii in 1:length(inputs.k)
        k = inputs.k[ii]
        if (ii/1 == round(ii/1)) & dispOpt
            if typeof(k)<:Real
                printfmtln("Solving for frequency {1:d} of {2:d}, ω = {3:2.3f}.",ii,length(inputs.k),k)
            else
                printfmtln("Solving for frequency {1:d} of {2:d}, ω = {3:2.3f}{4:+2.3f}i.",ii,length(inputs.k),real(k),imag(k))
            end
        end

        ζ = lufact(speye(1,1))

        for m in 1:M
            a = 0*a
            a[m] = 1.
            updateInputs!(inputs, :a, a)
            ψ₋, ϕ, ζ, inputs_s = compute_scatter(inputs, k; A=ζ)
            for m′ in 1:M
                 bm, cm = analyze_output(inputs_s, k, ψ₋, m′)
                 S[ii,m,m′,1] = bm + cm
            end
        end

        updateInputs!(inputs, :a, a_original)

        if !isempty(fileName)
            fid = open(fileName,"w")
            serialize(fid,(S,inputs,ii))
            close(fid)
        end
    end

    return S
end # end of fuction computeS_linear


"""
S =  computeS(inputs::InputStruct; N=10, N_Type="D", isNonLinear=false, F=1.,
    dispOpt = true, ψ_init = [], fileName = "")

    N is the number of steps to go from D0 = 0 to given D0
"""
function computeS_nonlinear(inputs1::InputStruct, k::Array{Complex128,1};
    N::Int=1, N_Type::String="D", isNonLinear::Bool=false, F::Array{Float64,1}=[1.],
    dispOpt::Bool=true, ψ_init::Array{Complex128,1}=Complex128[],
    fileName::String = "")::Array{Complex128,4}

    if !isempty(inputs1.bc ∩ ["o", "open", "pml_in"])
        inputs = deepcopy(inputs1)
        for i in 1:4
            if inputs.bc[i] in ["o", "open", "pml_in"]
                inputs.bc[i] = "pml_out"
                updateInputs!(inputs, :bc, inputs.bc)
            end
        end
    else
        inputs = inputs1
    end

    M = length(inputs.channels)
    D₀ = inputs.D₀
    A = inputs.a

    ψ₋ = Array{Complex128}(prod(inputs.N_ext))
    ψ  = Array{Complex128}(prod(inputs.N_ext))

    # if isempty(ψ_init) && (N>1)
    S = NaN*ones(Complex128,length(inputs.k),N,M,M)
    # else
        # S = NaN*ones(Complex128,length(inputs.k),1,2,2)
    # end

    if N > 1
        D = linspace(0,D₀,N)
        A_vec = linspace(0.001,1,N)
    else
        D = [inputs.D₀]
        A_vec = [1]
    end

    for ii in 1:length(inputs.k)

        k = inputs.k[ii]

        if (ii/1 == round(ii/1)) & dispOpt
            if typeof(k)<:Real
                printfmtln("Solving for frequency {1:d} of {2:d}, ω = {3:2.3f}.",ii,length(inputs.k),k)
            else
                printfmtln("Solving for frequency {1:d} of {2:d}, ω = {3:2.3f}{4:+2.3f}i.",ii,length(inputs.k),real(k),imag(k))
            end
        end

        if isempty(ψ_init)

            for j in 1:N

                if N_Type == "D"
                    updateInputs!(inputs, :D₀, D[j])
                elseif N_Type == "A"
                    updateInputs!(inputs, :a, A_vec[j]*A)
                end

                if isNonLinear
                    if isnan(ψ[1]) | j==1
                        ψ, W = computePsi(inputs,k,isNonLinear = true, F = F)
                    else
                        ψ, W = computePsi(inputs,k,isNonLinear = true, F = F, ψ_init = ψ)
                    end
                else
                    ψ = 0.
                end

                # compute linearized S-matrix
                ζ = lufact(speye(1,1)) # initialize lufact variable (speeds up repeated inhomogeneous solves)
                for m1 in 1:M
                    # set flux-normalized input amplitudes to unity
                    at = zeros(Complex128,M)
                    at[m1] = 1.
                    updateInputs!(inputs,:a,at)
                    # solve for scattered field
                    ψ₋, dummy1, ζ, inputs_s = computePsi(inputs, k; isNonLinear=false, F=F./(1+abs2.(γ(k,inputs)*ψ)), A=ζ)
                    # analyze into channels
                    println("here 1")
                    for m2 in m1:M
                        dt=0
                        if inputs.channelBoundaries[m2] in [1,2]
                            t = inputs_s.x₂_ext
                            u = inputs_s.x₁_inds
                            dt = inputs.dx̄[2]
                        else
                            t = inputs_s.x₁_ext
                            u = inputs_s.x₂_inds
                            dt = inputs.dx̄[1]
                        end
                        println("here 2")
                        ϕ = zeros(Complex128,length(t))
                        for q in 1:length(t)
                             ϕ[q] = conj(inputs.incidentWave(k, m2, inputs.∂R[inputs.channelBoundaries[m2]], t[ii], inputs.∂R, inputs.geometry, inputs.geoParams)[1])
                        end
                        println("here 3")
                        P = 0
                        if inputs.channelBoundaries[m2] in [1,3]
                            println("here 4")
                            println(size(ψ₋))
                            println(size(t))
                            println(size(u))
                            P = reshape(ψ₋,:,length(t))[u[1],:]
                            println("here 5")
                        else
                            println("here 6")
                            P = reshape(ψ₋,:,length(t))[u[end],:]
                        end
                        println("here 7")
                        println(size(P))
                        println(size(S))
                        println(j)
                        println(ii)
                        println(size(ϕ))
                        S[ii,j,m1,m2] = sum(P.*ϕ)*dt
                        S[ii,j,m2,m1] = S[ii,j,m1,m2]
                    end
                end
    println("here 8")
                updateInputs!(inputs,:D₀, D₀)
                updateInputs!(inputs,:a , A )

                if !isempty(fileName)
                    fid = open(fileName,"w")
                    serialize(fid,(inputs,D,S,ii,j))
                    close(fid)
                end

            end

        else
            if isNonLinear
                ψ = computePsi(inputs,k,isNonLinear = true, F = F, ψ_init = ψ_init)
            else
                ψ = 0.
            end

            inputs.a = [1.0,0.0]
            ψ₊, W = computePsi(inputs,k,isNonLinear = false, F = F./(1+abs2.(γ(inputs,k)*ψ)))

            inputs.a = [0.0,1.0]
            ψ₋, W = computePsi(inputs,k,isNonLinear = false, F = F./(1+abs2.(γ(inputs,k)*ψ)), A = W)

            S[1,1,ii,1] = ψ₊[x_inds[1]]*exp(+1im*inputs.dx*k)
            S[2,1,ii,1] = ψ₊[x_inds[end]]
            S[1,2,ii,1] = ψ₋[x_inds[1]]
            S[2,2,ii,1] = ψ₋[x_inds[end]]*exp(-1im*inputs.dx*k)

            inputs.D₀ = D₀
            inputs.a = A

            if !isempty(fileName)
                foo2(fid) = serialize(fid,(inputs,D,S,ii,1))
                open(foo2,fileName,"w")
            end

        end

    end
    println("here 9")
    return S
end # end of fuction computeS


################################################################################
### scattering auxilliaries
################################################################################


"""
synthesize_source(inputs, k)
"""
function synthesize_source(inputs::InputStruct, k::Complex128)::
    Tuple{Array{Complex128,1},SparseMatrixCSC{Complex128,Int64},Array{Complex128,1},Array{Complex128,1}}

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
    inputs1.ε[inputs1.scatteringRegions] = 1
    updateInputs!(inputs1, :ε, inputs1.ε);

    k² = k^2
    ɛk² = sparse(1:N, 1:N, inputs1.ɛ_sm[:]*k², N, N, +)

    j = (∇²+ɛk²)*(M₊.*φ₊ + M₋.*φ₋)

    return j, ∇², M₊.*φ₊, M₋.*φ₋
end


"""
source_mask(inputs)
"""
function source_mask(inputs::InputStruct)::Tuple{Array{Bool,1},Array{Bool,1}}
    M₊ = (inputs.∂S₊[1] .≤ inputs.x̄_ext[1] .≤ inputs.∂S₊[2]) .& (inputs.∂S₊[3] .≤ inputs.x̄_ext[2] .≤ inputs.∂S₊[4]) .& (inputs.∂S₋ .≤ sqrt.(inputs.x̄_ext[1].^2 + inputs.x̄_ext[2].^2))
    M₋ = (inputs.∂R[1] .≤ inputs.x̄_ext[1] .≤ inputs.∂R[2]) .& (inputs.∂R[3] .≤ inputs.x̄_ext[2] .≤ inputs.∂R[4]) .& (inputs.∂S₋ .≤ sqrt.(inputs.x̄_ext[1].^2 + inputs.x̄_ext[2].^2))
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
        r_inds = r .≥ inputs.∂S₋
        θ = atan2.(y,x)
        q = inputs.channels[m].tqn
        φ₊[r_inds] = exp.(1im*q*θ[r_inds]).*hankelh2.(q,k*r[r_inds])/2
        φ₋[r_inds] = exp.(1im*q*θ[r_inds]).*hankelh1.(q,k*r[r_inds])/2
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
wg_transverse_y(inputs, m, y)
"""
function wg_transverse_y(inputs1::InputStruct, k::Complex128, m::Int)::
    Tuple{Complex128, Array{Complex128,1}}

    inputs = open_to_pml_out(inputs1, true)

    wg_pos_ind = 3
    ind = find( (inputs.r_ext.==(8+inputs.channels[m].wg) )[1,:])[wg_pos_ind]

    fields = [:wgd,:wge,:wgt,:wgp]
    vals = [ String[inputs.wgd[inputs.channels[m].wg]],
             Float64[inputs.wge[inputs.channels[m].wg]],
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
analyze_output(inputs, k, ψ, m)
"""
function analyze_output(inputs::InputStruct, k::Complex128,
    ψ::Array{Complex{Float64},1}, m::Int)::Tuple{Complex128,Complex128}

    bm = 0.0im # ballistic coefficient
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
    elseif bc_sig in ["dOdd", "dOnn", "dOdn", "dOnd"]
        x = inputs.x₁[end] - inputs.∂R[1]
        y = inputs.x₂_ext
        φy = quasi_1d_transverse_y.(inputs,m,y)
        kᵤ = quasi_1d_transverse_y(inputs,m)
        kₓ = sqrt(k^2 - kᵤ^2)
        φ = +sqrt(1/kₓ)*exp(-1im*kₓ*x)*φy
        P = reshape(ψ[inputs.x̄_inds],inputs.N[1],:)[end,:]
        cm = sum(φ.*P)*inputs.dx̄[2]
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
        side_bool = [inputs.channels[q].side for q in 1:length(inputs.channels)] .!== inputs.channels[m].side
        wg_ind = find(wg_bool .& tqn_bool .& side_bool)
        if length(wg_ind) > 1
            error("Channels not uniquely defined.")
        end
        bm = inputs.a[wg_ind[1]]*phsb
        cm = sqrt(kₓ)*phs*sum(φ.*ε.*P)*inputs.dx̄[2]
    elseif (bc_sig in ["OOOO", "IIII"])
        nθ = Int(5e2)
        θ = linspace(0,2π,nθ)
        dθ = θ[2]-θ[1]
        R = findmin(inputs.∂R)[1]
        p = interpolate(reshape(ψ,inputs.N_ext[1],:), BSpline(Linear()), OnGrid() )
        X = R*cos.(θ[1:end-1])
        Y = R*sin.(θ[1:end-1])
        X_int = inputs.N_ext[1]*(X-inputs.∂R_ext[1])/(inputs.∂R_ext[2]-inputs.∂R_ext[1])
        Y_int = inputs.N_ext[2]*(Y-inputs.∂R_ext[3])/(inputs.∂R_ext[4]-inputs.∂R_ext[3])
        P = [p[X_int[ii],Y_int[ii]] for ii in 1:(nθ-1)]
        q = inputs.channels[m].tqn
        cm = 2*sum(exp.(-1im*q*θ[1:end-1]).*P)*dθ./hankelh1(q,k*R)
        bm = inputs.a[m]
    end

    return bm, cm
end


################################################################################
### Nonlinear auxilliaries for jacobian etc
################################################################################


"""
scattering_residual(Ψ, fvec, j, ∇², εk², k, inputs)
"""
function scattering_residual(Ψ::Array{Float64,1}, fvec::Array{Float64,1}, j,
                                    ∇², εk², k::Complex128, inputs::InputStruct)

    ψ = similar(j,Complex128)
    ind_r = 1:length(j)
    ind_i = length(j)+1:2*length(j)
    ψ = Ψ[ind_r] + 1im*Ψ[ind_i]
    temp = (∇²+ɛk²+χ(ψ,k,inputs)*k^2)*ψ - j
    fvec[ind_r] = real(temp)
    fvec[ind_i] = imag(temp)
end


"""
scattering_jacobian(Ψ, jacarray, j, ∇², εk², k, inputs)
"""
function scattering_jacobian(Ψ::Array{Float64,1}, jacarray, j, ∇², εk², k::Complex128, inputs::InputStruct)
    ψ = similar(j,Complex128)
    ind_r = 1:length(j)
    ind_i = length(j)+1:2*length(j)
    ψ = Ψ[ind_r] + 1im*Ψ[ind_i]
    temp = ∇²+ɛk²+χ(ψ,k,inputs)*k^2
    tempr = similar(temp,Float64)
    tempi = similar(temp,Float64)
    tr = nonzeros(tempr)
    ti = nonzeros(tempi)
    tr[:] = real((nonzeros(temp)))
    ti[:] = imag((nonzeros(temp)))
    tempj = [tempr+χʳʳ′(ψ,k,inputs) -tempi+χʳⁱ′(ψ,k,inputs); tempi+χⁱʳ′(ψ,k,inputs) tempr+χⁱⁱ′(ψ,k,inputs)]
    jacarray[:,:] = tempj[:,:]
end


"""
χ(ψ, k, inputs)
"""
function χ(ψ::Array{Complex128,1}, k::Complex128, inputs::InputStruct)::SparseMatrixCSC{Complex{Float64},Int64}
    N = prod(inputs.N_ext)
    h = hb(ψ,k,inputs)
    V = inputs.F_sm[:].*γ(k,inputs)*inputs.D₀./(1+h)
    return sparse(1:N, 1:N, V, N, N, +)
end


"""
"""
function hb(ψ::Array{Complex128,1},k::Complex128,inputs::InputStruct)::Array{Float64,1}
    N = prod(inputs.N_ext)
    h = abs2.(γ.(k, inputs)*ψ)
    return h
end


"""
χʳʳ′(Ψ, k, inputs)
"""
function χʳʳ′(ψ::Array{Complex128,1}, k::Complex128, inputs::InputStruct)::SparseMatrixCSC{Float64,Int64}
    N = prod(inputs.N_ext); D₀ = inputs.D₀; F_sm = inputs.F_sm; γt = γ(k, inputs)
    h = hb(ψ,k,inputs)
    V = -2D₀.*F_sm[:].*abs2(γt).*real.(γt.*ψ).*real.(ψ)./(1+h).^2
    return sparse(1:N,1:N,V,N,N,+)
end


"""
χⁱʳ′(Ψ, k, inputs)
"""
function χⁱʳ′(ψ::Array{Complex128,1}, k::Complex128, inputs::InputStruct)::SparseMatrixCSC{Float64,Int64}
    N = prod(inputs.N_ext); D₀ = inputs.D₀; F_sm = inputs.F_sm; γt = γ(k, inputs)
    h = hb(ψ,k,inputs)
    V = -2D₀.*F_sm[:].*abs2(γt).*imag.(γt.*ψ).*real.(ψ)./(1+h).^2
    return sparse(1:N,1:N,V,N,N,+)
end


"""
χʳⁱ′(Ψ, k, inputs)
"""
function χʳⁱ′(ψ::Array{Complex128,1}, k::Complex128, inputs::InputStruct)::SparseMatrixCSC{Float64,Int64}
    N = prod(inputs.N_ext); D₀ = inputs.D₀; F_sm = inputs.F_sm; γt = γ(k, inputs)
    h = hb(ψ,k,inputs)
    V = -2D₀.*F_sm[:].*abs2(γt).*real.(γt.*ψ).*imag.(ψ)./(1+h).^2
    return sparse(1:N,1:N,V,N,N,+)
end


"""
χⁱⁱ′(Ψ, k, inputs)
"""
function χⁱⁱ′(ψ::Array{Complex128,1}, k::Complex128, inputs::InputStruct)::SparseMatrixCSC{Float64,Int64}
    N = prod(inputs.N_ext); D₀ = inputs.D₀; F_sm = inputs.F_sm; γt = γ(k, inputs)
    h = hb(ψ,k,inputs)
    V = -2D₀.*F_sm[:].*abs2(γt).*imag.(γt.*ψ).*imag.(ψ)./(1+h).^2
    return sparse(1:N,1:N,V,N,N,+)
end
