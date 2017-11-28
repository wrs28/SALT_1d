@everywhere function geometry(x::Float64, y::Float64, geoParams::Array{Float64,1})::Int

    local region::Int

    R = geoParams[1]
    r = sqrt(x^2+y^2)
    if  r≤R
        region = 2
    else
        region = 1
    end

    return region
end

n = 2.
F = [0., 0.]
n₁ = [ 1.,  n].^2
n₂ = [ 0., 0.].^2

scatteringRegions = [2]
geoParams = Float64[.41]

wgd = ["x", "x"]
wgp = [+0.45,-0.45]
wgt = [ 0.025, 0.025]
wgn = [  2.0, 2.0]
wge = wgn.^2

∂R  =  [-.65,  .65,   -.65,   .65]
bc =  ["pml_in", "pml_in", "pml_in", "pml_in"]
bk = Complex128[.5,  0.] # Bloch wave vector [kₓ kᵤ]
input_modes = [[1],Int[]]

k₀ = complex(20.)
γ⟂ = 1e8
D₀ = 0.00

∂S  =  [-.55,  .55,   -.55,   .55]
a = Complex128[1.,0.,0.,0.]
channels = [ChannelStruct(1,"l",1),
            ChannelStruct(1,"r",1),
            ChannelStruct(2,"l",1),
            ChannelStruct(2,"r",1)]

subPixelNum = 10
coord = "xy" # XY, xy, Cart, cart, Cartesian, cartesian, rθ, rtheta, polar, Polar
N = [351,351]
