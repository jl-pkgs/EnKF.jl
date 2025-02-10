using EnKF
using OrdinaryDiffEq
using Plots
gr(framestyle=:box)

include("main_duffing.jl")

u0 = [1.0; -1.0]
tspan = (0.0, 50.0)

Δt = 1e-2
T = tspan[1]:Δt:tspan[end]

prob = ODEProblem(duffing, u0, tspan)
sol = solve(prob, RK4(), adaptive=false, dt=Δt)

integrator = init(prob, RK4(), adaptive=false, dt=Δt, save_everystep=false)

plot(sol, idxs=(1))

begin
  fprop = PropagationFunction()
  m = MeasurementFunction()
  z = RealMeasurementFunction()
  A = IdentityInflation()
  ϵ = AdditiveInflation(MvNormal(zeros(1), 1.0 * I))

  N = 50
  NZ = 1

  x₀ = [0.5, -0.5]
  ens = initialize(N, MvNormal(x₀, 2.0 * I))
  ens.S[1]
  estimation_state = [deepcopy(ens.S)]

  true_state = [deepcopy(x₀)]
  covs = []
  g = FilteringFunction()

  enkf = ENKF(N, NZ, fprop, A, g, m, z, ϵ;
    isinflated=false, isfiltered=false, isaugmented=false)
end


Δt = 1e-2
Tsub = 0.0:Δt:50.0-Δt

for (n, t) in enumerate(Tsub)
  global ens
  t, ens, cov = enkf(t, Δt, ens)

  push!(estimation_state, deepcopy(ens.S))
  push!(covs, deepcopy(cov))
end

s = hcat(sol(T).u...)
ŝ = hcat(mean.(estimation_state)...)



plt = plot(layout=(2, 1), legend=:bottomright)
plot!(plt[1], T, s[1, 1:end], linewidth=3, label="truth")
plot!(plt[1], Tsub, ŝ[1, 1:end-1], linewidth=3, markersize=2, label="EnKF mean", xlabel="t", ylabel="x", linestyle=:dash)

plot!(plt[2], T, s[2, 1:end], linewidth=3, label="truth")
plot!(plt[2], Tsub, ŝ[2, 1:end-1], linewidth=3, markersize=2, label="EnKF mean", xlabel="t", ylabel="y", linestyle=:dash)
