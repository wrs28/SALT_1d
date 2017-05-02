function incidentWave(inds,k::Number,inputs::Dict)
    
    x = inputs["x_ext"]
    y = inputs["u_ext"]
    Nₓ = inputs["N_ext"][1]
    Nᵤ = inputs["N_ext"][2]
    x_inds,y_inds = ind2sub((Nₓ,Nᵤ),inds)
    θ = inputs["extras"][2]
    
    kₓ = k*cos(θ)
    kᵤ = k*sin(θ)

    φ = zeros(Complex64,Nₓ*Nᵤ)
    for i in 1:length(inds)
        φ[inds[i]] = exp(1im*(kₓ*x[x_inds[i]] + kᵤ*y[y_inds[i]]))
    end
    
    return φ
    
end

return incidentWave