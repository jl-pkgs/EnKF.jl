module EnKF

using Distributions, Statistics, LinearAlgebra
import Statistics: mean, var, std, cov
import Base: size, length, show, hcat, +, -, fill!
import Distributions
import Random: AbstractRNG


export PropagationFunction,
  MeasurementFunction,
  FilteringFunction,
  RealMeasurementFunction
export ENKF
export initialize, initialize!

struct PropagationFunction end
struct MeasurementFunction end
struct FilteringFunction end
struct RealMeasurementFunction end


include("state.jl")
include("inflation.jl")
# include("update.jl")
# include("system.jl")
# include("stochasticEnKF.jl")
include("ETKF.jl")


""""
Define system ENKF which performs the data assimilation
using the stochastic ensemble Kalman filter

Fields:
 - 'f' : propagation function
 - 'A' : inflation
 - 'G' : filtering function acting on the state
 
 - 'm' : measurement function based on state
 - 'z' : real measurement function
 - 'ϵ' : measurement noise distribution

 - 'bounds' : bounds on certain states

 - 'isinflated' : Bool = true if state is inflated,
     = false otherwise

 - 'isfiltered' : Bool = true if state has to be filtered,
     = false otherwise

 - 'isaugmented' : Bool = true if measurement function is nonlinear,
     = false otherwise
"""
mutable struct ENKF{N,NZ}
  # "Ensemble of states"
  # ENS::EnsembleState{N, NS, TS}
  "Propagation function"
  f::PropagationFunction

  "Covariance Inflation"
  A::Union{InflationType,RecipeInflation}

  "Filter function"
  G::FilteringFunction

  "Measurement function based on state"
  m::MeasurementFunction

  "Real measurement function"
  z::RealMeasurementFunction

  "Measurement noise distribution"
  ϵ::AdditiveInflation{NZ}

  "Boolean: is data assimilation used"
  isenkf::Bool

  "Boolean: is state vector inflated"
  isinflated::Bool

  "Boolean: is state vector filtered"
  isfiltered::Bool

  "Boolean: is state vector augmented"
  isaugmented::Bool
end
# "Bounds on certain state"
# bounds


"""
    Define action of ENKF on EnsembleState
"""
function (enkf::ENKF{N,NZ})(t::Float64, Δt::Float64,
  ens::EnsembleState{N,TS}) where {N,NZ,TS}

  enkf.f(t, ens) # Propagate each ensemble member

  "Is data assimilation used"
  if enkf.isenkf == true

    enkf.isinflated && (enkf.A(ens)) # Covariance inflation
    enkf.isfiltered && (enkf.G(ens)) # State filtering

    ensfluc = EnsembleState(N, ens.S[1])
    deviation(ensfluc, ens) # Compute mean and deviation

    A′ = hcat(ensfluc)

    " Additional computing for RTPS inflation"
    if typeof(enkf.A) <: Union{RTPSInflation,RTPSAdditiveInflation,RTPSRecipeInflation}
      # correct scaling by 1/N-1 instead of 1/N for small ensembles
      σᵇ = std(ensfluc.S; corrected=false)
    end

    "Compute measurement"
    mens = EnsembleState(N, zeros(NZ))

    for (i, s) in enumerate(ens.S)
      mens.S[i] = enkf.m(t, deepcopy(s))
    end
    Â = hcat(deepcopy(mens))

    "Define measurement matrix H for linear observations, can be time-varying"
    if enkf.isaugmented == false
      H = deepcopy(enkf.m(t))
    end

    "Compute deviation from measurement of the mean"
    Ŝ = mean(deepcopy(ens))
    Â′ = Â .- enkf.m(t, Ŝ)

    "Get actual measurement"
    zens = EnsembleState(N, zeros(NZ))
    enkf.z(t + Δt, zens)
    # @show zens

    "Perturb actual measurement"
    enkf.ϵ(zens)
    D = hcat(zens)

    "Analysis step with representers, Evensen, Leeuwen et al. 1998"
    "Construct representers"
    if enkf.isaugmented == true
      b = ((Â′ * Â′') + (N - 1) * cov(enkf.ϵ) * I) \ (D - Â)
      Bᵀb = (A′ * Â′') * b
    else
      b = (H * (A′ * A′') * H' + (N - 1) * cov(enkf.ϵ) * I) \ (D - Â)
      Bᵀb = (A′ * A′') * H' * b
    end

    "Analysis step"
    ens += cut(Bᵀb)

    " Additional computing for RTPS inflation"
    if typeof(enkf.A) <: Union{RTPSInflation,RTPSAdditiveInflation,RTPSRecipeInflation}
      ensfluc = EnsembleState(N, ens.S[1])
      deviation(ensfluc, ens)
      σᵃ = std(ensfluc.S, corrected=false)
      enkf.A(ens, σᵇ, σᵃ)
    end

    "State filtering if 'isfiltered==true' "
    enkf.isfiltered && (enkf.G(ens))

    " Compute a posteriori covariance"
    deviation(ensfluc, ens)
    A′ = hcat(ensfluc)

    return t + Δt, ens, A′ * A′'
  else
    return t + Δt, ens
  end
end

# Create constructor for ENKF
function ENKF(N, NZ, f, A, G, m, z, ϵ;
  isenkf::Bool=true, isinflated::Bool=false, isfiltered::Bool=false, isaugmented::Bool=false)
  return ENKF{N,NZ}(f, A, G, m, z, ϵ, isenkf, isinflated, isfiltered, isaugmented)
end

# size(enkf::ENKF{N, TS, NZ, TZ}) where {N, TS, NZ, TZ} = (N, size, NZ)

# function Base.show(io::IO, sys::ENKF{N, TS, NZ, TZ}) where {N, TS, NZ, TZ}
#     NS = size()
#     print(io, "Ensemble Kalman filter with $N members of state of size $ and measurement vector of length $NZ")
# end

end # module
