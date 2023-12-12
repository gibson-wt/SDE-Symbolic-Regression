function diff(func, u, l)
    return func.(u) .- func.(l)
end

function multiplyx(func)
    g(t) = t * func(t)
    return g
end

function integrate(integrand, t)
    u = t[2:end]
    l = t[1:end-1]
    integrand_u = integrand[2:end]
    integrand_l = integrand[1:end-1]
    return sum(0.5* (integrand_u .+ integrand_l) .* (u .- l))
end

function integrate_increments(integrand, integrator)
    n = length(integrator)
    integrates = zeros(Float64, n)
    for i in 2:n
        integrates[i] = integrates[i-1] + integrand[i-1]*(integrator[i]-integrator[i-1])
    end
    return collect(integrates)
end

function integrate_xf(iphi, iiphi, integrand, t)
    integrand_u = integrand[2:end]
    integrand_l = integrand[1:end-1]
    u = t[2:end]
    l = t[1:end-1]

    a = (integrand_u .- integrand_l) ./ (u .- l)
    b = integrand_l .- a .* l

    c = diff(multiplyx(iphi), u, l)
    d = diff(iiphi, u, l)
    e = diff(iphi, u, l)

    return sum(a .* c .+ a .* d .+ b .* e)
end

function integrate_x2f2(phi,iphi,iiphi,integrand,t)
    integrand_u = integrand[2:end]
    integrand_l = integrand[1:end-1]
    u = t[2:end]
    l = t[1:end-1]

    a = (integrand_u .- integrand_l) ./ (u .- l)
    b = integrand_l .- a .* l

    cubic(z) = z^3
    quadratic(z) = z^2
    linear(z) = z

    c = a.^2 .* diff(cubic,u,l) .+ 3* a .* b .* diff(quadratic,u,l) .+ 3* b.^2 .* diff(linear,u,l)
    d = diff(multiplyx(multiplyx(phi)),u,l)
    e = diff(multiplyx(iphi),u,l)
    f = diff(iiphi,u,l)
    g = diff(multiplyx(phi),u,l)
    h = diff(iphi,u,l)
    i = diff(phi,u,l)

    return sum(1/6 .* c .- a.^2 /2 .* d .+ a.^2 .* (e .+ f) .- a .* b .* (g .- h) .- b.^2 /2 .* i)
end

function integrate_xf2(phi,iphi,integrand,t)
    integrand_u = integrand[2:end]
    integrand_l = integrand[1:end-1]
    u = t[2:end]
    l = t[1:end-1]

    a = (integrand_u .- integrand_l) ./ (u .- l)
    b = integrand_l .- a .* l

    quadratic(z) = z^2
    linear(z) = z

    c = 1/4*diff(quadratic,u,l)
    d = 0.5*diff(linear,u,l)
    e = 0.5*diff(phi,u,l)
    f = 0.5*diff(iphi,u,l)
    g = 0.5*diff(multiplyx(phi),u,l)

    return sum(a .* c .+ b .* d .- b .* e .- a .* g .+ a .* f)

end


