module continuous_util_jl

using Plots, LRMoE

export
       plot_simulated_obs,
       plot_LRMoE_fit,
       plot_simulated_T_delay,
       plot_simulated_T_claim_T_delay,
       plot_observed_losses

function plot_simulated_obs(obs)
    return histogram(obs; bins=0:2:200,
                     xlabel="Observations", ylabel="Frequency",
                     title="Distribution of Observations",
                     legend=false, size=(750, 500))
end

function plot_LRMoE_fit(obs, X_mat, LRMoE_fitted)
    plt_series = 0:1:200
    pred_probs = predict_class_prior(X_mat, LRMoE_fitted.model_fit.α).prob
    experts_dens = hcat([exp.(LRMoE.expert_ll_exact.(e, plt_series)) for e in LRMoE_fitted.model_fit.comp_dist[1,:]]...)
    pred_dens = mean(experts_dens * pred_probs', dims=2)
    p = histogram(obs; bins=0:2:200, label = "Data",
                  xlabel="Observations", ylabel="Density",
                  title="Distribution of Observations and LRMoE Model Fit",
                  legend=true, size=(750, 500), normalize=true)
    plot!(plt_series, pred_dens, linewidth = 3, label = "LRMoE Fit")
    return p
end

function plot_simulated_T_delay(T_delay)
    return histogram(T_delay; bins=0:2:200,
                     xlabel="Reporting Delay", ylabel="Frequency",
                     title="Simulated Distribution of Reporting Delay",
                     legend=false, size=(750, 500))
end

function plot_simulated_T_claim_T_delay(T_accident, T_delay)
    plot_idx = vcat([findall(T_accident + T_delay .> 365)[1], findall(T_accident + T_delay .<= 365)[2],
                     findall(T_accident + T_delay .> 365)[3], findall(T_accident + T_delay .<= 365)[4],]...)
    p = plot(; size=(800, 450),
             xlims=(0, 450), ylims=(0.5, 4.5),
             xlabel="Time (Days)",
             legend=:topleft)
    for (idx, obs) in enumerate(plot_idx)
        scatter!([T_accident[obs], T_accident[obs] + T_delay[obs]], [idx, idx]; label="Claim $idx")
    end
    vline!([365]; label="Valuation Date")
    title!("Samples of Claim and Delay Times")
    return p
end

function plot_observed_losses(y)
    return histogram(y; bins=0:2:200,
                     xlabel="Observed Losses", ylabel="Frequency",
                     title="Distribution of Observed Losses",
                     legend=false, size=(750, 500))
end

end