using Statistics
import SymbolicUtils
import SymbolicRegression: Options, SRRegressor
import MLJ: machine, fit!, predict, report
import DynamicExpressions: eval_tree_array
using Interpolations

include("Loss_Functions.jl")

# Initialise global constants
N=3
mesh=100
T=1.0
sigma = 5e-2
# Construct times
aa = 0:1/mesh:T
num_timesteps = size(aa)[1]
a = collect(aa)
size(a)
A = [a  for i in 1:N]
t = reduce(hcat,A)

# Construct x data
theta1 = 2.0
theta2 = 2.0
x0=1.0
A = log(theta2 * x0)
x = 1/theta2 .* exp.(A*exp.(-theta1 .* t))

# with some noise
noise = [sigma * sqrt(aa) * randn(Float64, 1, N) for aa in a]
noise = reduce(vcat,noise)

x_n = x+noise
t_=reshape(t,N*(mesh+1),1)
x_=(reshape(x_n,N*(mesh+1),1))

using Plots
display(plot(t,x_n, labels="", xlabel="t", ylabel="x",title="Data to be learnt from"))

# Set up loss functions
I2 =  find_I2(x_,S,N,num_timesteps,a)# calculate integral not involving f so we don't need to this on each iteration
I2_per_batch = find_I2_perbatch(x_,S,N,num_timesteps,a)

loss(tree,dataset,options) = LossDCODE(tree,dataset,options,S,N,num_timesteps,I2_per_batch,a)
loss_modified(tree,dataset,options) = LossDCODE_modified(tree,dataset,options,S,N,num_timesteps,I2,a)

# Define Operatpors

unary_operators = [log]
binary_operators = [*,+,/] 
populations = 5
options = Options(
    binary_operators=binary_operators,
    unary_operators=unary_operators,
    populations=populations,
)

# Set up symbolic regression

DCODE = SRRegressor(
    niterations=25,
    binary_operators=binary_operators,
    unary_operators=unary_operators,
    loss_function=loss,
    populations=populations,
    parallelism=:multithreading,
)

DCODE_modified = SRRegressor(
    niterations=25,
    binary_operators=binary_operators,
    unary_operators=unary_operators,
    loss_function=loss_modified,
    populations=populations,
    parallelism=:multithreading,
)

# Train models
mach = machine(DCODE,x_,x_)
@time fit!(mach)
r = report(mach)

mach_modified = machine(DCODE_modified,x_,x_)
@time fit!(mach_modified)
r_modified = report(mach_modified)

# Results Analysis
y,_=eval_tree_array(r.equations[3],reshape(x,1,N*(mesh+1)), options)
y_modified,_ = eval_tree_array(r2.equations[4],reshape(x,1,N*(mesh+1)), options)
true_ = -theta1 .* x .* log.(theta2 .* x)

#
scatter(true_, y1, xlabel="Truth", ylabel="Prediction")
scatter(true_, y2, xlabel="Truth", ylabel="Prediction")

#
plot([],[],label="", xlabel="t",ylabel="t",title="Comparison")
plot!(a,true_[1:num_timesteps],label="Target")
plot!(a,y1[1:num_timesteps],label="D-CODE")
plot!(a,y2[1:num_timesteps],label="Mod. D-CODE")