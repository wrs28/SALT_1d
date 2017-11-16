coord = "xy" # XY, xy, Cart, cart, Cartesian, cartesian, rθ, rtheta, polar, Polar
N = [604,151]

∂  =  [-1.5,  .5,   -.2,   .2]
bc =  ["o", "d", "d", "d"]
bk = [.5,  0.] # Bloch wave vector [kₓ kᵤ]

n = 2.
F = [0.0, 0.0, 1.0]
ɛ = [1.0, 1.0, n  ].^2
incidentWaveRegions = [2, 3]
scatteringRegions = [3]

k₀ = 20
k  = 20.6 + linspace(-.5,+.5,75)
γ⊥ = 1e8
D₀ = 0.00

geoParams = [ .9, .9, 1.3, π/5] # in general an array

nChannels = 2
a = [1., 0.]
inputs_modes = [[1],[]]

subPixelNum = 15

return (coord, N, k₀, k, ∂, bc, bk, inputs_modes, F, ɛ, γ⊥, D₀, a, geoParams,
         incidentWaveRegions, scatteringRegions, nChannels, subPixelNum)
