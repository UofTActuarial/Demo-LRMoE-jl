module cnesor_truncation_util_jl

using Plots

export
       plot_simulated_T_delay,
       plot_simulated_T_claim_T_delay,
       plot_observed_losses

function plot_simulated_T_delay(T_delay)
    return histogram(T_delay; bins=0:2:200,
                     xlabel="Reporting Delay", ylabel="Frequency",
                     title="Simulated Distribution of Reporting Delay",
                     legend=false, size=(750, 500))
end

function plot_simulated_T_claim_T_delay(T_accident, T_delay; plot_idx=[5, 10, 2500, 9998])
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
                     title="Simulated Distribution of Observed Losses",
                     legend=false, size=(750, 500))
end

end