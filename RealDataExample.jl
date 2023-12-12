import SymbolicUtils
import SymbolicRegression: Options, SRRegressor
import MLJ: machine, fit!, predict, report
import DynamicExpressions: eval_tree_array
using Interpolations
using StochasticDiffEq

include("Loss_Functions.jl")
include("Interpolate.jl")
include("Analysis.jl")

## Import Data 

using Pickle
using PyCall

py"""
import pickle
 
def load_pickle(fpath):
    with open(fpath, "rb") as f:
        data = pickle.load(f)
    return data
"""

load_pickle = py"load_pickle"
data = load_pickle("real_data_c1.pkl")

using CSV
using DataFrames

# Assuming your CSV file is named "data.csv"
data = CSV.read("gemini_BTCUSD_2020_1min", DataFrame)


## Preprocess Data
S=50
num_timesteps = size(data)[1]
M = size(data)[2]
NT =num_timesteps # Chnage if you want less steps in data
x = reshape(data[:,1,1],NT,1)
t = reshape(data[:,1,2],NT,1)
a = reshape(data[:,1,2],NT,1)

# Fortunately t increments are the same which speeds up calcultions

N=1
for i in 2:M-1 # Final time increments of different size so ignore
    if !any(y->y<=0,data[:,i,1])
        x = vcat(x,data[:,i,1])
        t = vcat(t,data[:,i,2])
        N+=1
    end
end
N
inputs = hcat(x,t)
starts = collect(1:NT:N*NT)
inputs[starts,1]
# State Options
unary_operators = [sqrt,log]
binary_operators = [*,+] 
populations = 5
options = Options(
    binary_operators=binary_operators,
    unary_operators=unary_operators,
    populations=populations,
)

Loss(tree,dataset,options) = QVloss2(tree,dataset,options,NT,N,a)

model = SRRegressor(
    niterations=15, 
    binary_operators=binary_operators,
    unary_operators=unary_operators,
    loss_function=Loss,
    populations=populations,
    parallelism=:multithreading,
)
input = reshape(inputs[1:N*NT,:],N*NT,2)
input_t = reshape(t[1:N*NT],N*NT,1)
mach = machine(model,input,input_t)
input_t
# Train Model
@time fit!(mach)
rf=report(mach)
f = interp_2D(rf.equations[3], options,1.0)

f(1,1)
X = reshape(inputs[:,1],NT,N)
tt = reshape(inputs[:,2],NT,N)
b = f.(X,tt)
It = find_It(X,b,a,NT,N)
findall(isnan,It)

sigma2 = find_sigma2(log.(X),It,a,NT)
sigma = sqrt(abs(sigma2))
g(z) = sigma*z

# summ = SDE_Analysis_xonly(ff,g,1.0,0.45,0.55)
using Plots

summ = SDE_Analysis_xt(f,g,1.0,0.05,0.95)
plot(summ)
plot!(data[:,1,2],data[:,1,1])

plot([],[])
plot(data[:,1,2],data[:,1,1])
for i in 2:100
    if !any(y->y<=5e-2,data[:,i,1])
        plot!(data[:,i,2],data[:,i,1])
    end
end
display(Plots.current())


ff(z,s)=4.48*s^2*z+log(s+1e-12)
X
tt
b = ff.(X,tt)
QV_loss(X,b,a,num_timesteps,N)
MGF_loss(X,b,a,S,num_timesteps,N)
