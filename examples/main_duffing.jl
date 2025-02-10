using EnKF


function duffing(du, u, p, t)
  γ = 0.1
  d = 0.1
  ω = 1.4

  du[1] = u[2]
  du[2] = u[1] - u[1]^3 - γ * u[2] + d * cos(ω * t)
end



function (::PropagationFunction)(t::Float64, ENS::EnsembleState{N,TS}) where {N,TS}
  for (i, s) in enumerate(ENS.S)
    set_t!(integrator, deepcopy(t))
    set_u!(integrator, deepcopy(s))
    step!(integrator)
    ENS.S[i] = deepcopy(integrator.u)
  end
  return ENS
end

function (::MeasurementFunction)(t::Float64, s::TS) where TS
  return [s[2]]
end

function (::MeasurementFunction)(t::Float64) # error function
  return reshape([0.0, 1.0], (1, 2))
end

function (::RealMeasurementFunction)(t::Float64, ENS::EnsembleState{N,TZ}) where {N,TZ}
  let s = sol(t)
    fill!(ENS, [deepcopy(s[2])])
  end
  return ENS
end
