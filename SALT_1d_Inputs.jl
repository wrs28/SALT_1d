N = 3001
n = 3.0

k₀ = (39.7)
k  = linspace(18,22,100)

bc = ["r" "r"]

∂ = [-0.1     0.0     0.5      1.0      1.1]
Γ = [ Inf     Inf     Inf      Inf      Inf]
F =     [0.0      +1.0    +1.0     0.0]
ɛ =     [1.0       n       n/2     1.0].^2 + 0im

γ⟂ = 2.0
D₀ = -0.1

a = [1.0, 1.0] # incoming: [L R]
b = [0.0, 1.0] # outgoing: [L R]

subPixelNum = 10

(N, k₀, k, bc, ∂, Γ, F, ɛ, γ⟂, D₀, a, b, subPixelNum)