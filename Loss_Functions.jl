import SymbolicUtils
import DynamicExpressions: eval_tree_array
using Statistics

include("Integrate.jl")

phi(z,s) = sin(s*pi*z)
iphi(z,s) = -cos(s*pi*z)/(s * pi)
iiphi(z,s) = sin(s*pi*z)/((s*pi)^2)

function LossDCODE(tree,dataset,options,S,N,num_timesteps,I2_perbatch,t)
    prediction, flag = eval_tree_array(tree, dataset.X, options)
    if !flag
        return Inf
    end
    return DCODE_loss(t,prediction,S,N,num_timesteps,I2_perbatch)
end

function LossDCODE_modified(tree,dataset,options,S,N,num_timesteps,I2,t)
    prediction, flag = eval_tree_array(tree, dataset.X, options)
    if !flag
        return Inf
    end
    return DCODE_loss_modified(t,prediction,S,N,num_timesteps,I2)
end

function DCODE_loss(t,prediction,S,N,num_timesteps,I2_perbatch)
    total = 0
    for s in 1:S
        iphi_(z) = iphi(z,s)
        iiphi_(z) = iiphi(z,s)
        tot=0
        for d in 1:N
            ind = 1+(d-1)*num_timesteps:d*num_timesteps
            pred_f = prediction[ind]
            I1 = integrate_xf(iphi_,iiphi_, pred_f, t)
            tot+=1/N*(I1+I2_perbatch[s,d])^2
        end
        total += tot
    end
    return total
end

function DCODE_loss_modified(t,prediction,S,N,num_timesteps,I2) 
    I1 = find_I1(prediction, S, N, num_timesteps, t)
    return sum((I1 .+ I2).^2)
end
function find_I2_perbatch(y,S,N,num_timesteps,t)
    I2 = zeros(Float64,S,N)
    for s in 1:S
        neg_iphi(z) = -1*iphi(z,s)
        phi_(z) = phi(z,s)
        # Find E[X] - no need to do per batch to reduce number of integration steps
        xx = zeros(Float64,num_timesteps)
        for d in 1:N
            ind = 1+(d-1)*(num_timesteps):d*(num_timesteps)
            xx = y[ind]
            I2[s,d] = integrate_xf(phi_, neg_iphi, xx, t)
        end
    end
    return I2
end

function find_f_error(prediction_f, I2_perbatch, S, N, num_timesteps, t)
    err_f = zeros(Float64,S)
    for s in 1:S
        iphi_(z) = iphi(z,s)
        iiphi_(z) = iiphi(z,s)
        tot=0
        for d in 1:N
            ind = 1+(d-1)*num_timesteps:d*num_timesteps
            pred_f = prediction_f[ind]
            I1 = integrate_xf(iphi_,iiphi_, pred_f, t)
            tot+=1/N*(I1+I2_perbatch[s,d])^2
        end
        err_f[s] = tot
    end
    return err_f
end

function find_I1(prediction_f, S,N,num_timesteps, t)
    I1=zeros(Float64,S)
    for s in 1:S
        iphi_(z) = iphi(z,s)
        iiphi_(z) = iiphi(z,s)
        pred_f = zeros(Float64,num_timesteps)
        for d in 1:N
            ind = 1+(d-1)*num_timesteps:d*num_timesteps
            pred_f .+= prediction_f[ind]/N      
        end
        I1[s] = integrate_xf(iphi_,iiphi_, pred_f, t)
    end
    return I1
end

function find_I2(y,S,N,num_timesteps,t)
    I2 = zeros(Float64,S)
    for s in 1:S
        neg_iphi(z) = -1*iphi(z,s)
        phi_(z) = phi(z,s)

        # Find E[X] - no need to do per batch to reduce number of integration steps
        xx = zeros(Float64,num_timesteps)
        for d in 1:N
            ind = 1+(d-1)*(num_timesteps):d*(num_timesteps)
            xx .+= y[ind]/N
        end
        I2[s] = integrate_xf(phi_, neg_iphi, xx, t)
    end
    return I2
end


function only_f_loss(tree,dataset,options, I2, S, N, num_timesteps, t)
    prediction, flag = eval_tree_array(tree, dataset.X, options)
    if !flag
        return Inf
    end
    I1 = find_I1(prediction, S, N, num_timesteps, t)
    return sum((I1 .+ I2).^2)
end

function find_It(X,b,t,num_timesteps,N)
    I = zeros(Float64, num_timesteps,N)
    for j in 1:N 
        integrand = b[:,j] ./ X[:,j]
        I[:,j] = integrate_increments(integrand, t)
    end
    return I
end

function find_sigma2(lnX,It,t,num_timesteps)
    ElnXt = collect([mean(row) for row in eachrow(lnX[2:end,:])])
    ElnX0 = mean(lnX[1,:])
    EIt = collect([mean(row) for  row in eachrow(It[2:end,:])])
    sigma2 = 2/num_timesteps * sum( (ElnX0 .+EIt .- ElnXt) ./ t[2:end]) # no division by 0
    return sigma2
end

function find_Rt(X,It, s)
    X0 =transpose(reduce(hcat,collect([X[1,:] for row in eachrow(X)])))
    R = X.^(s/S) ./ (exp.(s*It/S) .* (X0).^(s/S))
    ER = collect([mean(row) for  row in eachrow(R)])
    return ER
end

function MGF_loss(X,pred,t,S,num_timesteps,N)
    It = find_It(X,pred,t,num_timesteps,N)
    sigma2 = find_sigma2(log.(X),It,t,num_timesteps)
    loss = 0
    for s in -S:1:S
        Rt = find_Rt(X,It,s)
        integrand = (exp.(sigma2/2*((s/S)^2-s/S) * t) .- Rt).^2
        loss += integrate(integrand,t)
    end
    return loss/(2S+1)
end

function MGFloss(tree,dataset,options,S,num_timesteps,N,t)
    prediction, flag = eval_tree_array(tree, dataset.X, options)
    # For 2D input
    X = reshape(dataset.X[1,:],num_timesteps,N)
    pred = reshape(prediction,num_timesteps,N)
    if !flag
        return Inf
    end
    return MGF_loss(X,pred,t,S,num_timesteps,N)
end

function find_QV(X, num_timesteps,N)
    h_u = X[2:end,:]
    h_l = X[1:end-1,:]

    increments = (h_u .- h_l).^2

    QV = zeros(Float64,num_timesteps,N)
    for i in 2:num_timesteps
        QV[i,:] = QV[i-1,:] + increments[i-1,:]
    end
    return QV
end

function QV_loss(pred,a,num_timesteps,N,QV)
    integrand = 1 ./ ((pred).^2)
    H = zeros(Float64, num_timesteps,N)
    for i in 1:N
        H[:,i]= integrate_increments(integrand[:,i],QV[:,i])
    end
    HH = collect([mean(row) for row in eachrow(H)])
    loss = integrate((HH .- a).^2,a)
    return loss
end


function QVloss(tree,dataset,options,num_timesteps,N,a,QV)
    prediction, flag = eval_tree_array(tree, dataset.X, options)
    # For 2D input
    pred = reshape(prediction,num_timesteps,N)
    if !flag
        return Inf
    end
    return QV_loss(pred,a,num_timesteps,N,QV)
end

function MGFloss2(tree,dataset,options,S,num_timesteps,N,t,pred_g,Ig)
    prediction, flag = eval_tree_array(tree, dataset.X, options)
    # For 2D input
    pred = reshape(prediction,num_timesteps,N)
    if !flag
        return Inf
    end
    return MGF_loss2(pred,t,S,num_timesteps,N,pred_g,Ig)
end

function MGF_loss2(pred,t,S,num_timesteps,N,pred_g,Ig)
    integrand = pred ./ pred_g
    It = zeros(Float64, num_timesteps,N)
    for i in 1:N
        It[:,i] = integrate_increments(integrand[:,i],t)
    end
    loss = 0
    for s in -S:1:S
        Qt = find_Qt(Ig,s,It)
        integrand = (exp.(0.5*((s/S)^2) * t) .- Qt).^2
        loss += integrate(integrand,t)
    end
    return loss/(2S+1)
end

function find_Qt(Ig,s,It)
    Q = exp.((s/S)*(Ig.-It))
    Qt = collect([mean(row) for row in eachrow(Q)])
    return Qt
end

function find_Ig(pred_g,X,num_timesteps,N)
    Ig = zeros(Float64,num_timesteps,N)
    for i in 1:N
        Ig[:,i] = integrate_increments(1 ./ (pred_g[:,i]), X[:,i])
    end
    return Ig
end