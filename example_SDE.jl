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

# Global Constants
N = 10
σ = 0.1
dt=0.25
T=5.0
S=50

# Construct Paths
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

using Plots
plot([],[], label="")
sol = solve(prob; dt = dt)
NonLinearSDE_sol = find_NonLinearSDE_sol(sol.u,t,σ)
plot!(t,NonLinearSDE_sol,label="")
data=NonLinearSDE_sol
if N > 1
    for i in 1:N-1
        sol = solve(prob; dt = dt)
        NonLinearSDE_sol = find_NonLinearSDE_sol(sol.u,t,σ)
        data=hcat(data,NonLinearSDE_sol)
        plot!(t,NonLinearSDE_sol,label="")
    end
end
plot!(t, 0.25*(t.+2).^2) # Plot curve if no noise taken into consideration
display(Plots.current())
# Preprocess data
x = reshape(data, num_timesteps*N,1)
t_ = reshape(reduce(vcat,collect([t for i in 1:N])), num_timesteps*N,1)

# State Options
unary_operators = [sqrt,log]
binary_operators = [*,+] 
populations = 5
options = Options(
    binary_operators=binary_operators,
    unary_operators=unary_operators,
    populations=populations,
)

Loss(tree,dataset,options) = MGFloss(tree,dataset,options,S,num_timesteps,N,t)

model = SRRegressor(
    niterations=25, 
    binary_operators=binary_operators,
    unary_operators=unary_operators,
    loss_function=Loss,
    populations=populations,
    parallelism=:multithreading,
)
mach = machine(model,x,t_)

# Train Model
@time fit!(mach)

I2 = find_I2(x,S,N,num_timesteps,t)
DLoss(tree,dataset,options) = only_f_loss(tree,dataset,options, I2, S, N, num_timesteps, t)
DCODE = SRRegressor(
    niterations=25, 
    binary_operators=binary_operators,
    unary_operators=unary_operators,
    loss_function=DLoss,
    populations=populations,
    parallelism=:multithreading,
)
machD = machine(DCODE,x,t_)
@time fit!(machD)

# Determine best model 
r_f = report(mach)
f = linear_model(r_f.equations[4], options)

r_D = report(machD)
D=linear_model(r_D.equations[end], options)

X = data
b = f.(data)
It = find_It(X,b,t,num_timesteps,N)
sigma2 = find_sigma2(log.(X),It,t,num_timesteps)
sigma = sqrt(abs(sigma2))
g(z) = sigma*z

plot([],[],label="", xlabel='t',ylabel="X_t",title="Predicted Soltution Vs Data Given")
plot!(t,data,label="")
@time umm = SDE_Analysis_xonly(f,g,T,0.1,0.9)
plot!(umm)
sol = ODE_Analysis_xonly(D,T)
plot!(sol.t,sol.u)

# Find True Solution
ff(z) = sqrt(z)
bb = ff.(data)
IIt = find_It(X,bb,t,num_timesteps,N)
sigma2_true = find_sigma2(log.(X),IIt,t,num_timesteps)
sigma_true = sqrt(abs(sigma2_true))
gg(z) = sigma_true*z
summ_true = SDE_Analysis_xonly(ff,gg,T,0.1,0.9)
plot([],[],label="", xlabel='t',ylabel="X_t",title="True Soltution Vs Data Given")
plot!(t,data, label="")
plot!(t, 0.25*(t.+2).^2,)
plot!(summ_true, label="True", legend=true) # Plot curve if no noise taken into consideration
display(Plots.current())


# Repeat for Model involving Quadratic variation.
QV = find_QV(data, num_timesteps,N)

Loss_g(tree,dataset,options)= QVloss(tree,dataset,options,num_timesteps,N,t,QV)

model_g = SRRegressor(
    niterations=5, 
    binary_operators=binary_operators,
    unary_operators=unary_operators,
    loss_function=Loss_g,
    populations=populations,
    parallelism=:multithreading,
)

mach_g = machine(model_g,x,t_)
@time fit!(mach_g)

r  =report(mach_g) # 0.21 x
h = linear_model(r.equations[2],options)
pred_g=abs.(h.(data))
Ig=find_Ig(pred_g,data, num_timesteps,N)

Loss_f(tree,dataset,options) = MGFloss2(tree,dataset,options,S,num_timesteps,N,t,pred_g,Ig)

model_Q = SRRegressor(
    niterations=5, 
    binary_operators=binary_operators,
    unary_operators=unary_operators,
    loss_function=Loss_f,
    populations=populations,
    parallelism=:multithreading,
)

mach_Q = machine(model_Q,x,t_)
@time fit!(mach_f)

rQ = report(mach_Q)
fQ = linear_model(rQ.equations[2],options)

plot([],[],label="", xlabel='t',ylabel="X_t",title="Predicted Soltution Vs Data Given")
plot!(t,data,label="")
@time summ = SDE_Analysis_xonly(fQ,h,T,0.2,0.8)
plot!(summ)
