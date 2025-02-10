using Distributions, LinearAlgebra, Statistics
using EnKF, Test

@testset "Initialize" begin
  A = MvNormal(zeros(4), I)
  ENS = initialize(10, A)
  @test size(ENS) == (10, (4,))

end
