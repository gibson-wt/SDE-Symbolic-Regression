import SymbolicUtils
import DynamicExpressions: eval_tree_array
using Interpolations

function linear_model(eqn,options)
    x_range = collect(-1000:0.1:1000)
    data,_ = eval_tree_array(eqn,reshape(x_range,1,length(x_range)),options)
    f(z) = linear_interpolation(x_range,data)(z)
    return f
end

function interp_2D(eqn,options,T)
    x_range = collect(-1000:0.1:1000)
    t_range = 0:0.1:T
    data_grid_f = [eval_tree_array(eqn,reshape([i,j],2,1),options)[1][1] for i in x_range, j in t_range]
    f(x,t) = interpolate((x_range,t_range),data_grid_f, Gridded(Linear()))(x,t)
    return f
end