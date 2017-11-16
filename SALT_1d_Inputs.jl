N = 1001
n = 3.0 + 0.0im

k₀ = 40
k  = linspace(18,22,10)

bc = ["out", "out"]

∂ = [ 0.0,    0.1,     0.9,      1.0]
Γ = [ Inf,     Inf,    Inf,      Inf]
F =     [0.0,       +1.0,     0.0]
ɛ =     [1.0,        n  ,     1.0].^2

γ⟂ = 10
D₀ = -0.01

a = [0.5, 0.2] # incoming: [L R]
b = [0.0, 0.0] # outgoing: [L R]

subPixelNum = 50

(N, k₀, k, bc, ∂, Γ, F, ɛ, γ⟂, D₀, a, b, subPixelNum)