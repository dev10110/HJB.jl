# this file contains some standard hamiltonians



function sup_u(b, U::S) where {S<:AbstractPolyhedron}
    # given a control set U
    # determine sup_{u \in U} (b * u)

    return maximum(b * u for u in vertices(U))
end

function sup_u(b, U::S; ϵ = 1e-9) where {S<:Ball2}
    # given a control set U
    # determine sup_{u \in U} (b * u)

    # check that b has sufficient norm
    if norm(b) < ϵ
        return 0
    end

    # check that U is centered at 0
    if norm(U.center) > ϵ
        error("U must be centered at 0. Got U.center = $(U.center)")
    end


    # notice
    # sup_u (b' u) = b' * (b / norm(b)  * norm(u)) = norm(b) * norm(u)
    return norm(b) * U.radius

end

function getHamiltonian_CBF(f, g, U::S, γ = 1.0) where {S<:LazySet}

    # defined by Dev, based on Lygeros and Choi

    function H(t, x, V, DxV)

        # H() = min( 0 , sup_{u \in U} ( DxV' * (f(t, x) + g(t, x) u)) + γ V)

        H0 = DxV' * f(t, x) + sup_u(DxV' * g(t, x), U) + γ * V

        return min(0, H0)

    end

    return H
end

function getHamiltonian_Viability(f, g, U::S) where {S<:LazySet}

    # defined in Lygeros

    function H(t, x, V, DxV)

        # H() = min( 0 , sup_{u \in U} ( DxV' * (f(t, x) + g(t, x) u)) )

        H0 = DxV' * f(t, x) + sup_u(DxV' * g(t, x), U)

        return min(0, H0)

    end

    return H

end
