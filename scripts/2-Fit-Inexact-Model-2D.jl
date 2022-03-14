using DrWatson
@quickactivate "LRMoEjl Demo"

using DataFrames, Distributions, JLD2, LRMoE, Plots, Random

# Load saved complete data
X_obs, Y_obs = load(datadir("simulation-incomplete-data.jld2"), "X_obs", "Y_obs")

# Propose a model guess
α_guess = fill(0.0, 2, ncol(X_obs))
model_guess = [PoissonExpert(10.0) ZINegativeBinomialExpert(0.50, 40, 0.80);
                LogNormalExpert(3.0, 1.0) InverseGaussianExpert(15.0, 15.0)]

# Fit a model
model_fit = fit_LRMoE(Y_obs, X_obs, α_guess, model_guess;
                      exact_Y=false, ecm_iter_max=100)
# Should converge in 6 runs with default ϵ

# Model summary
summary(model_fit)

# Get fitted parameters
model_fit.model_fit.α
model_fit.model_fit.comp_dist

# Model performance
model_fit.loglik
model_fit.loglik_np
model_fit.AIC
model_fit.BIC

# Misspecified model: other experts
α_guess = fill(0.0, 2, ncol(X_obs))
model_guess = [ZIPoissonExpert(0.50, 10.0) ZIPoissonExpert(0.50, 20.0);
               LogNormalExpert(2.0, 0.5) LogNormalExpert(1.0, 1.0)]

# Fit the model
model_fit_alt = fit_LRMoE(Y_obs, X_obs, α_guess, model_guess;
                          exact_Y=false, ecm_iter_max=100)
# Should converge in 26 runs with default ϵ

# Model summary
summary(model_fit_alt)

# Get fitted parameters
model_fit_alt.model_fit.α
model_fit_alt.model_fit.comp_dist

# Model performance
model_fit_alt.loglik
model_fit_alt.loglik_np
model_fit_alt.AIC
model_fit_alt.BIC

# Compare two models
mean.(model_fit.model_fit.comp_dist)
mean.(model_fit_alt.model_fit.comp_dist)

var.(model_fit.model_fit.comp_dist)
var.(model_fit_alt.model_fit.comp_dist)

# Some visualization
# Dimension 1
plot_series = collect(0:1:50)
# Empirical Distribution
empirical_dist = [sum(Y_obs[:, 2] .== i) for i in plot_series] ./ length(Y_obs[:, 2])
# True Model
comp_density = hcat([exp.(LRMoE.expert_ll_exact.(expert, plot_series))
                     for expert in model_fit.model_fit.comp_dist[1, :]]...)
latent_class_prob = predict_class_prior(Matrix(X_obs), model_fit.model_fit.α).prob
fitted_density_person = comp_density * latent_class_prob'
fitted_density = mean(fitted_density_person; dims=2)
# Alternative Model
comp_density = hcat([exp.(LRMoE.expert_ll_exact.(expert, plot_series))
                     for expert in model_fit_alt.model_fit.comp_dist[1, :]]...)
latent_class_prob = predict_class_prior(Matrix(X_obs), model_fit_alt.model_fit.α).prob
fitted_density_person = comp_density * latent_class_prob'
fitted_density_alt = mean(fitted_density_person; dims=2)

bar(plot_series .- 0.50, empirical_dist; bar_width=0.25, label="Data")
bar!(plot_series, fitted_density; bar_width=0.25, label="True Model")
bar!(plot_series .+ 0.5, fitted_density_alt; bar_width=0.25, label="Alt Model")
title!("Fitted vs Data: Dimension 1 of Y")
savefig(plotsdir("fitted-vs-data-incomplete-model-dimension-1.png"))

# Save results
jldsave(datadir("fitted-incomplete-2d-model.jld2"); model_fit=model_fit)
jldsave(datadir("fitted-incomplete-2d-model-alternative.jld2");
        model_fit=model_fit_alt)
