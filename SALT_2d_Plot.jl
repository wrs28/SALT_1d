module SALT_2d_Plot

export wavePlot

include("SALT_2d_Core.jl")

using .Core
using PyPlot


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

function wavePlot(ψ, inputs::Dict; truncate = false, array = true, how = abs2)
    # plot results of computePolesL wavefunctions

    N = size(ψ,2)

    if truncate
        M = inputs["N"]
        x = inputs["x"]
        y = inputs["u"]
        ∂ = inputs["∂"]
    else
        M = inputs["N_ext"]
        x = inputs["x_ext"]
        y = inputs["u_ext"]
        ∂ = inputs["∂_ext"]
    end
    
    ψ_plot = NaN*zeros(Complex128, N, M[1], M[2])

    for i in 1:N
        ψ_plot[i,:,:] = reshape(ψ[:,i], (M[1],M[2]) )
    end
    
    
    if array
    
        figure(figsize=4.8*[3,(N+1)*(∂[4]-∂[3])/(∂[2]-∂[1])])
        subplots_adjust(hspace=0.0)
        subplots_adjust(wspace=0.0)

        subplot(N+1,3,1); axt = gca()
        r = whichRegion( (x,y), inputs)
        pcolormesh(x, y, r.')
        xlim( [ ∂[1],∂[2] ] )
        ylim( [ ∂[3],∂[4] ] )
axis("tight")
        xlabel("Region/Real")
        setp(axt[:xaxis][:tick_top]())
        setp(axt[:xaxis][:set_label_position]("top")) 

        subplot(N+1,3,2); axt = gca()
        ɛ, F = subpixelSmoothing(inputs; truncate=truncate, r=r)
        pcolormesh(x,y,ɛ.')
        xlim( [ ∂[1],∂[2] ] )
        ylim( [ ∂[3],∂[4] ] )
axis("tight")
        xlabel("Index/Imag")
        setp(axt[:get_xticklabels](),visible=false)
        setp(axt[:get_yticklabels](),visible=false)
        setp(axt[:xaxis][:tick_top]())
        setp(axt[:xaxis][:set_label_position]("top")) 

        subplot(N+1,3,3); axt = gca()
        pcolormesh(x, y, F.')
        xlim( [ ∂[1],∂[2] ] )
        ylim( [ ∂[3],∂[4] ] )
axis("tight")
        xlabel("F/abs2")
        setp(axt[:get_xticklabels](),visible=false)
        setp(axt[:get_yticklabels](),visible=false)
        setp(axt[:xaxis][:tick_top]())
        setp(axt[:xaxis][:set_label_position]("top")) 

        for i in 1:N

            subplot(N+1,3,3+3*(i-1)+1); axt = gca()
            pcolormesh(x,y,real(ψ_plot[i,:,:]).',cmap = "bwr")
            xlim( [ ∂[1],∂[2] ] )
            ylim( [ ∂[3],∂[4] ] )
axis("tight")
            clim([-1,1])
            setp(axt[:get_xticklabels](),visible=false)
            setp(axt[:get_yticklabels](),visible=false)

            subplot(N+1,3,3+3*(i-1)+2); axt = gca()
            pcolormesh(x,y,imag(ψ_plot[i,:,:]).',cmap = "bwr")
            xlim( [ ∂[1],∂[2] ] )
            ylim( [ ∂[3],∂[4] ] )
axis("tight")
            clim([-1,1])
            setp(axt[:get_xticklabels](),visible=false)
            setp(axt[:get_yticklabels](),visible=false)

            subplot(N+1,3,3+3*(i-1)+3); axt = gca()
            pcolormesh(x,y,abs2.(ψ_plot[i,:,:]).',cmap = "gray_r")
            xlim( [ ∂[1],∂[2] ] )
            ylim( [ ∂[3],∂[4] ] )
axis("tight")
            clim([0,1])
            setp(axt[:get_xticklabels](),visible=false)
            setp(axt[:get_yticklabels](),visible=false)

        end
        
    else
       
        for i in 1:N
            figure(i, figsize=8*[1,(∂[4]-∂[3])/(∂[2]-∂[1])])
            if how==abs2
                cmap = "gray_r"
            else
                cmap = "bwr"
            end
            pcolormesh(x,y,how.(ψ_plot[i,:,:]).',cmap=cmap)
            xlim( [ ∂[1],∂[2] ] )
            ylim( [ ∂[3],∂[4] ] )
            axis("equal")
        end
        
    end
    
    return (x,y,ψ_plot)
    
end # end of function wavePlot

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

end