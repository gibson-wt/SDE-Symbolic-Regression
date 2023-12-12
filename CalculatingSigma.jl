import SymbolicUtils
import SymbolicRegression: Options, SRRegressor
import MLJ: machine, fit!, predict, report
import DynamicExpressions: eval_tree_array
using Interpolations

using StochasticDiffEq
using DiffEqNoiseProcess
include("Loss_Functions.jl")
include("Interpolate.jl")
include("Analysis.jl")

N = 250
T=1.0
dt=0.1
sigmas = 1e-2:1e-2:1
num_sigmas = length(sigmas)

B = RealWienerProcess(0.0, 0.0)
prob = NoiseProblem(B, (0.0, T))
sol = solve(prob,dt=dt)
t=sol.t

num_timesteps = length(t)

function StochasticExponentiate(W,t,σ)
    return exp.(σ .* W .- (σ)^2/2 .* t)
end

function find_NonLinearSDE_sol(W,t,σ)
    StochasticExponential = StochasticExponentiate(W,t,σ)
    integral_term = integrate_increments(1 ./ sqrt.(StochasticExponential), t)
    return StochasticExponential .* (1 .+ 0.5* integral_term).^2
end

data = zeros(Float64,num_timesteps,N,num_sigmas)

for j in 1:num_sigmas
    for i in 1:N
        sigma = sigmas[j]
        sol = solve(prob; dt = dt)
        NonLinearSDE_sol = find_NonLinearSDE_sol(sol.u,t,sigma)
        data[:,i,j]=NonLinearSDE_sol
    end
end

ff(z)=sqrt(z)
sigma_pred = zeros(Float64, num_sigmas)
for j in 1:num_sigmas
    X=data[:,:,j]
    b = ff.(X)
    It = find_It(X,b,t,num_timesteps,N)
    sigma2 = find_sigma2(log.(X),It,t,num_timesteps)
    sigma_pred[j] = sqrt(abs(sigma2))
end

scatter(sigmas,sigma_pred,label="",xlabel="True",ylabel="Prediction",title="True Vs Predicted value of sigma")