module SALT_2d_plus

( export InputStruct, ChannelStruct, processInputs, updateInputs!, computeCF,
computePole_L, computeZero_L, computeUZR_L, computeK_NL1, computePole_NL1,
computeZero_NL1, computeUZR_NL1, computeK_NL2, computePole_NL2, computeZero_NL2,
computeUZR_NL2, computeK_NL2_parallel, computePole_NL2_parallel, computeZero_NL2_parallel,
computeUZR_NL2_parallel, compute_scatter, computeS, analyze_output, P_wait, wavePlot )
#, updateInputs, computePolesL, computePolesNL1, computePolesNL2, computeZerosL, computeZerosNL1, computeZerosNL2, computeCFs, solve_SPA, solve_scattered, solve_single_mode_lasing, solve_CPA, computeS, bootstrap, CF_analysis, CF_synthesis, computeZerosL2

include("SALT_2d.jl")
using .SALT_2d
using PyPlot

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""
wavePlot(ψ, inputs)
"""
function wavePlot(ψ::Union{Array{Complex{Float64},1}, Array{Complex{Float64},2}},
    inputs::InputStruct; truncated::Bool = false, array::Bool = true,
    how::Function = abs2)::Tuple{Array{Float64,1},Array{Float64,1},Array{Complex128,3}}

    N = size(ψ,2)

    if truncated
        M = inputs.N
        x₁ = inputs.x₁
        x₂ = inputs.x₂
        ∂R = inputs.∂R
        r = reshape(inputs.r_ext[inputs.x̄_inds],inputs.N[1],:)
        ε_sm = reshape(inputs.ε_sm[inputs.x̄_inds],inputs.N[1],:)
        F_sm = reshape(inputs.F_sm[inputs.x̄_inds],inputs.N[1],:)
    else
        M = inputs.N_ext
        x₁ = inputs.x₁_ext
        x₂ = inputs.x₂_ext
        ∂R = inputs.∂R_ext
        r = inputs.r_ext
        ε_sm = inputs.ε_sm
        F_sm = inputs.F_sm
    end

    ψ_plot = NaN*zeros(Complex128, M[1], M[2], N)

    for i in 1:N
        ψ_plot[:,:,i] = reshape(ψ[:,i], (M[1],M[2]) )
    end

    if array

        figure(figsize=4.8*[3,(N+1)*(∂R[4]-∂R[3])/(∂R[2]-∂R[1])])
        subplots_adjust(hspace=0.0)
        subplots_adjust(wspace=0.0)

        subplot(N+1,3,1); axt = gca()
        pcolormesh(x₁, x₂, transpose(r))
        xlim( [ ∂R[1],∂R[2] ] )
        ylim( [ ∂R[3],∂R[4] ] )
        axis("tight")
        xlabel("Region/Real")
        setp(axt[:xaxis][:tick_top]())
        setp(axt[:xaxis][:set_label_position]("top"))

        subplot(N+1,3,2); axt = gca()
        pcolormesh(x₁, x₂, transpose(real(ɛ_sm)))
        xlim( [ ∂R[1],∂R[2] ] )
        ylim( [ ∂R[3],∂R[4] ] )
        axis("tight")
        xlabel("Index/Imag")
        setp(axt[:get_xticklabels](),visible=false)
        setp(axt[:get_yticklabels](),visible=false)
        setp(axt[:xaxis][:tick_top]())
        setp(axt[:xaxis][:set_label_position]("top"))

        subplot(N+1,3,3); axt = gca()
        pcolormesh(x₁, x₂, transpose(F_sm))
        xlim( [ ∂R[1],∂R[2] ] )
        ylim( [ ∂R[3],∂R[4] ] )
        axis("tight")
        xlabel("F/abs2")
        setp(axt[:get_xticklabels](),visible=false)
        setp(axt[:get_yticklabels](),visible=false)
        setp(axt[:xaxis][:tick_top]())
        setp(axt[:xaxis][:set_label_position]("top"))

        for i in 1:N

            subplot(N+1,3,3+3*(i-1)+1); axt = gca()
            pcolormesh(x₁, x₂,transpose(real(ψ_plot[:,:,i])),cmap = "bwr")
            xlim( [ ∂R[1],∂R[2] ] )
            ylim( [ ∂R[3],∂R[4] ] )
            axis("tight")
            clim([-1,1])
            setp(axt[:get_xticklabels](),visible=false)
            setp(axt[:get_yticklabels](),visible=false)

            subplot(N+1,3,3+3*(i-1)+2); axt = gca()
            pcolormesh(x₁, x₂,transpose(imag(ψ_plot[:,:,i])),cmap = "bwr")
            xlim( [ ∂R[1],∂R[2] ] )
            ylim( [ ∂R[3],∂R[4] ] )
            axis("tight")
            clim([-1,1])
            setp(axt[:get_xticklabels](),visible=false)
            setp(axt[:get_yticklabels](),visible=false)

            subplot(N+1,3,3+3*(i-1)+3); axt = gca()
            pcolormesh(x₁, x₂,transpose(abs2.(ψ_plot[:,:,i])),cmap = "gray_r")
            xlim( [ ∂R[1],∂R[2] ] )
            ylim( [ ∂R[3],∂R[4] ] )
            axis("tight")
            clim([0,1])
            setp(axt[:get_xticklabels](),visible=false)
            setp(axt[:get_yticklabels](),visible=false)

        end

    else

        for i in 1:N
            figure(i, figsize=8*[1,(∂R[4]-∂R[3])/(∂R[2]-∂R[1])])
            if how==abs2
                cmap = "gray_r"
            else
                cmap = "bwr"
            end
            pcolormesh(x₁, x₂,transpose(how.(ψ_plot[:,:,i])),cmap=cmap)
            xlim( [ ∂R[1],∂R[2] ] )
            ylim( [ ∂R[3],∂R[4] ] )
            axis("equal")
        end

    end

    return (x₁,x₂,ψ_plot)
end # end of function wavePlot

end #end of module SALT_2d_plus
