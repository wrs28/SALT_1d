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

n₁ = [ 1.,  2.]
n₂ = [ 0., 0.]
F = [0.0, 0.0, 0.0]

scatteringRegions = [2]
geoParams = Float64[.41]

wgd = String[] #["x", "x"]
wgp = Float64[]#[+0.45,-0.45]
wgt = Float64[]#[ 0.025, 0.025]
wgn = Float64[]#[  2.0, 2.0]
wge = wgn.^2

∂R  =  [-.65,  .65,   -.65,   .65]
bc =  ["pml_in", "pml_in", "pml_in", "pml_in"]
bk = Complex128[.5,  0.] # Bloch wave vector [kₓ kᵤ]
input_modes = [[1],Int[]]

k₀ = complex(20.)
γ⟂ = 1e8
D₀ = 0.00

∂S₊ =  [-.55,  .55,   -.55,   .55]
∂S₋ = .01
a = Complex128[1.,0.,0.,0.]
channels = [ChannelStruct( 0,0,""),
            ChannelStruct( 1,0,""),
            ChannelStruct(-1,0,""),
            ChannelStruct( 2,0,"")]

subPixelNum = 10
coord = "xy" # XY, xy, Cart, cart, Cartesian, cartesian, rθ, rtheta, polar, Polar
N = [351,351]
