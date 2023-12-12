using OrdinaryDiffEq
using StochasticDiffEq

function SDE_Analysis_xonly(f, g, T, low, high)
    h(u,p,t) = f(u)
    j(u,p,t) = g(u)
    u0 = 1.0
    tspan = (0.0,T)
    dt = 1//16
    prob = SDEProblem(h,j,u0,tspan)
    ensembleprob = EnsembleProblem(prob)
    sol = solve(ensembleprob,EM(),dt=dt, EnsembleThreads(), trajectories=1000)
    summ = EnsembleSummary(sol,0:0.01:T;quantiles= [low, high])
    return summ
end

function SDE_Analysis_xt(f, g, T, low, high)
    h(u,p,t) = f(u,t)
    j(u,p,t) = g(u)
    u0 = 1.0
    tspan = (0.0,T)
    dt = 1//16
    prob = SDEProblem(h,j,u0,tspan)
    ensembleprob = EnsembleProblem(prob)
    sol = solve(ensembleprob,EM(),dt=dt, EnsembleThreads(), trajectories=1000)
    summ = EnsembleSummary(sol,0:0.01:T;quantiles= [low, high])
    return summ
end

function ODE_Analysis_xonly(f,T)
    g(u,p,t) = f(u)
    u0 = 1.0
    tspan = (0.0,T)
    dt = 1//16
    prob = ODEProblem(g,u0,tspan)
    sol = solve(prob, Tsit5(), dt=dt) 
    return sol
end
