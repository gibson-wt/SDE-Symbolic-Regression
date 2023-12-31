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

## Preprocess Data
S=50
num_timesteps = size(data)[1]
M = size(data)[2]
inds = [1+5*i for i in 0:72]
NT = size(inds)[1] # Chnage if you want less steps in data
x = reshape(data[inds,1,1],NT,1)
t = reshape(data[inds,1,2],NT,1)
a = reshape(data[inds,1,2],NT,1)

# Fortunately t increments are the same which speeds up calcultions

N=1
for i in 2:M-1 # Final time increments of different size so ignore
    if !any(y->y<=0,data[inds,i,1])
        x = vcat(x,data[inds,i,1])
        t = vcat(t,data[inds,i,2])
        N+=1
    end
end
N
# N = 30 to restrict number of data point if necessary
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
input = reshape(inputs[1:N*NT,:],N*NT,2)
input_t = reshape(t[1:N*NT],N*NT,1)

QV = find_QV(reshape(input[:,1],NT,N),NT,N)

Loss_g(tree,dataset,options) = QVloss(tree,dataset,options,NT,N,a,QV)

model-g = SRRegressor(
    niterations=5, 
    binary_operators=binary_operators,
    unary_operators=unary_operators,
    loss_function=Loss-g,
    populations=populations,
    parallelism=:multithreading,
)
mach_g = machine(model_g,input,input_t)

# Train Model
@time fit!(mach_g)
rg=report(mach_g)
g =linear_model(rg.equations[3],options) # May need to change to 2D interpolation if model requires so

pred_g=abs.(g.(reshape(input[:,1],NT,N)))
Ig = find_Ig(pred_g,reshape(input[:,1],NT,N),NT,N)
Loss(tree,dataset,options) = MGFloss2(tree,dataset,options,S,NT,N,a,pred_g,Ig)

model = SRRegressor(
    niterations=10, 
    binary_operators=binary_operators,
    unary_operators=unary_operators,
    loss_function=Loss,
    populations=populations,
    parallelism=:multithreading,
)

mach = machine(model,input,input_t)

@time fit!(mach)
rf=report(mach)
f = interp_2D(rf.equations[7], options,1.0)

using Plots

summ = SDE_Analysis_xt(f,g,1.0,0.2,0.8)
plot(summ)
plot!(data[:,1,2],data[:,1,1]) # plot first data path as well

# Plot all data if desired
# plot([],[])
# plot(data[:,1,2],data[:,1,1])
# for i in 2:100
#     if !any(y->y<=5e-2,data[:,i,1])
#         plot!(data[:,i,2],data[:,i,1])
#     end
# end
# display(Plots.current())


