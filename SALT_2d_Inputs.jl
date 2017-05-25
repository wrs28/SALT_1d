N = 251*[1,1]
n = 2.

R = .5 # radius of disk
β = 1.0842857142857143

λ₀ = 2π*R./10
λ  = 2π*R./linspace(9,11,10)

origin = (0.,0.)
∂  =  [-1   1    -1    1]*1.2*R + [origin[1] origin[1] origin[2] origin[2]]
bc =  ["o"  "o"  "o"  "o"]
bk = [1.       0.] # Bloch wave vector [kₓ kᵤ]

F = [0.0   1.0   0.0]
ɛ = [1.0   n     n  ].^2

γ⊥ = 1e8
D₀ = 0.00

a = 1

extras = (origin,π/3,R,β) #(θ,R) this is in general a tuplet

geometryFile = "SALT_2d_Geometry.jl"
incidentWaveFile = "SALT_2d_IncidentWave.jl"

return (N, λ₀, λ, ∂, bc, bk, F, ɛ, γ⊥, D₀, a, geometryFile, incidentWaveFile, extras)