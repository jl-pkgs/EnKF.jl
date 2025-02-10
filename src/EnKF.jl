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

struct PropagationFunction end
struct MeasurementFunction end
struct FilteringFunction end
struct RealMeasurementFunction end


include("state.jl")
include("initial.jl")
include("inflation.jl")

# include("update.jl")
# include("system.jl")
include("stochasticEnKF.jl")
include("ETKF.jl")

end # module
