n = complex(2.)
F = [0.0, 0.0, 0.0, 1.0, 0.0]
ɛ = [1.0,   n,   n,   n,   n].^2

incidentWaveRegions = [2, 3]
scatteringRegions = [3]
geoParams = Float64[1,2]

wgd = ["y"]
wgp = [0.3]
wgt = [0.1]
wge = [2.0].^2

function geometry(x::Float64, y::Float64, geoParams::Array{Float64,1})::Int

    local region::Int

    if  -.2<x<.2 && -.2<y<.2
        region = 2
    else
        region = 1
    end

    return region
end
