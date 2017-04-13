N = 451*[1,1]

println(pwd())

λ₀ = 2π./14.8
λ  = 2π./linspace(14,16,10)

origin = [0.,0.]
∂ = 0.5*[-1    1    -1    1] + [origin[1] origin[1] origin[2] origin[2]]
bc =    ["o"  "o"   "o"  "o"]

F = [0.0   0.0    1.0]
ɛ = [1.0   3.0    3.0].^2

γ⟂ = 1.
D₀ = 0.00

a = 1

extras = (π/3) #(θ) this is in general a tuplet

#############################################

R1 = .3 # radius of disk
R2 = .25 # radius of disk

function geometry(x,y,∂)

    local region::Int
        
    if ( (x-origin[1])^2 + (y-origin[2])^2 < R2^2)
        region = 3
    elseif ( (x-origin[1])^2 + (y-origin[2])^2 < R1^2)
        region = 2
    else
        region = 1
    end
    
    return region

end

#############################################

function incidentWave(inds,k::Number,inputs::Dict)
    
    x = inputs["x_ext"]
    y = inputs["u_ext"]
    Nₓ = inputs["N_ext"][1]
    Nᵤ = inputs["N_ext"][2]
    x_inds,y_inds = ind2sub((Nₓ,Nᵤ),inds)
    θ = inputs["extras"][1]
    
    kₓ = k*cos(θ)
    kᵤ = k*sin(θ)

    φ = zeros(Complex64,Nₓ*Nᵤ)
    for i in 1:length(inds)
        φ[inds[i]] = exp(1im*(kₓ*x[x_inds[i]] + kᵤ*y[y_inds[i]]))
    end
    
    return φ
    
end

###########################################

return (N, λ₀, λ, ∂, bc, F, ɛ, γ⟂, D₀, a, geometry, incidentWave, extras)