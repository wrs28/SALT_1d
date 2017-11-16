function geometry(x::Float64, y::Float64, geoParams::Array{Float64,1})::Int

    local region::Int

    g = geoParams[3]
    θ = geoParams[4]

    if (    (-geoParams[1] < (x-g)*cos(θ) - y*sin(θ) < geoParams[1]) &&
            (-geoParams[2] < (x-g)*sin(θ) + y*cos(θ) < geoParams[2])    )
        region = 3
    elseif x > -1
        region = 2
    else
        region = 1
    end

    return region

end

return geometry
