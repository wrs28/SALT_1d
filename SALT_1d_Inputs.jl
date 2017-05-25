N = 3001
n = 3.0

k₀ = (39.7)
k  = linspace(18,22,100)

bc = ["pml_out" "pml_out"]

∂ = [-0.1     0.0     0.5      1.0      1.1]
Γ = [ Inf     10      Inf      1.0      Inf]
F =     [0.0      +1.0    -1.0     0.0]
ɛ =     [1.0       n       n       1.0].^2 + 0im

γ⟂ = 0.16
D₀ = 0.2

a = [1.0, 0.0] # incoming: [L R]
b = [0.0, 1.0] # outgoing: [L R]

(N, k₀, k, bc, ∂, Γ, F, ɛ, γ⟂, D₀, a, b)