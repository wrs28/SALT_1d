


    function solve_SPA1(inputs, ω; z₀₊=0.001im, z₀₋=0.001im, F = 1., u=[], η=[], φ₊=[], φ₋=[])

        x = inputs["x_ext"][inputs["x_inds"]]
        y = inputs["u_ext"][inputs["u_inds"]]
        ∂_ext = inputs["∂_ext"]
        N_ext = inputs["N_ext"]

        D₀ = copy(inputs["D₀"])
        a = copy(inputs["a"])
        inputs["D₀"] = 0.
        inputs["a"] = 1.

        ϕ,dummy = scatteredFields(inputs,inputs["ω₀"],isNonLinear = false)
        X_inds = [i for i in inputs["x_inds"], j in inputs["u_inds"]]
        Y_inds = [j for i in inputs["x_inds"], j in inputs["u_inds"]]
        φ = ϕ[Y_inds[:]*N_ext[1] + X_inds[:]]

        inputs["D₀"] = copy(D₀)
        inputs["a"] = copy(a)

        η,u = computeCF(inputs,ω,1);

        function γ()
            return inputs["γ⊥"]/(ω-inputs["ω₀"]+1im*inputs["γ⊥"])
        end

        r = whichRegion((x,y),∂_ext,inputs["geometry"])
        F = inputs["F_ext"][r[:]]

        function f!(z,fvec)

            b = inputs["a"]
            Z = z[1]+1im*z[2]

            numerator = u.*F.*(Z*u+b*φ)
            denominator = 1+abs2(Z*u+b*φ)
            term = (inputs["D₀"]*γ()./η).*trapz(numerator./denominator,inputs["dr"])

            fvec[1] = real(term[1] - Z)
            fvec[2] = imag(term[1] - Z)

            return term,Z

        end

        result = nlsolve(f!,[real(z₀),imag(z₀)])

        z = result.zero[1]+1im*result.zero[2]

        ψ = inputs["a"]*φ+z*u

        return ψ

    end



function solve_SPA(inputs,ω; z₀ = 0.0im)

        x = inputs["x_ext"][inputs["x_inds"]]
        y = inputs["u_ext"][inputs["u_inds"]]
        ∂_ext = inputs["∂_ext"]
        N_ext = inputs["N_ext"]

        D₀ = copy(inputs["D₀"])
        a = copy(inputs["a"])
        inputs["D₀"] = 0.
        inputs["a"] = 1.

        ϕ,dummy = solve_scattered(inputs,inputs["ω₀"],isNonLinear = false)
        X_inds = [i for i in inputs["x_inds"], j in inputs["u_inds"]]
        Y_inds = [j for i in inputs["x_inds"], j in inputs["u_inds"]]
        φ = ϕ[Y_inds[:]*N_ext[1] + X_inds[:]]

        inputs["D₀"] = copy(D₀)
        inputs["a"] = copy(a)

        η,u = computeCFS(inputs,ω,1);

        function γ()
            return inputs["γ⊥"]/(ω-inputs["ω₀"]+1im*inputs["γ⊥"])
        end

        r = whichRegion((x,y),∂_ext,inputs["geometry"])
        F = inputs["F_ext"][r[:]]

        function f!(z,fvec)

            b = inputs["a"]
            Z = z[1]+1im*z[2]

            numerator = u.*F.*(Z*u+b*φ)
            denominator = 1+abs2(Z*u+b*φ)
            term = (inputs["D₀"]*γ()./η).*trapz(numerator./denominator,inputs["dr"])

            fvec[1] = real(term[1]/Z - 1)
            fvec[2] = imag(term[1]/Z - 1)

            return term,Z

        end

        result = nlsolve(f!,[real(z₀),imag(z₀)])

        z = result.zero[1]+1im*result.zero[2]

        ψ = inputs["a"]*φ+z*u

        return ψ

    end
