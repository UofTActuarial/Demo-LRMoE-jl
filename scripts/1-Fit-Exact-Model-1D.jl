using DrWatson
@quickactivate "LRMoEjl Demo"

using DataFrames, Distributions, JLD2, LRMoE, Plots, Random

# Load saved complete data
X, Y_complete = load(datadir("simulation-complete-data.jld2"), "X", "Y_complete")

# Propose a model guess
α_guess = fill(0.0, 2, ncol(X))
model_guess = [LogNormalExpert(3.0, 0.5) InverseGaussianExpert(18, 22)]

# Fit a model
Y_2 = reshape(Y_complete[:, 2], :, 1) # Fitting function needs a matrix
model_fit_dim2 = fit_LRMoE(Y_2, X, α_guess, model_guess;
                           exact_Y=true, ecm_iter_max=100)
# Should converge in 53 runs with default ϵ

# Model summary
summary(model_fit_dim2)

# Get fitted parameters
model_fit_dim2.model_fit.α
model_fit_dim2.model_fit.comp_dist

# Model performance
model_fit_dim2.loglik
model_fit_dim2.loglik_np
model_fit_dim2.AIC
model_fit_dim2.BIC

# Generate numbers for visualization
plot_series = collect(0.0:0.05:250)
comp_density = hcat([pdf.(expert, plot_series) for expert in model_fit_dim2.model_fit.comp_dist]...)
latent_class_prob = predict_class_prior(Matrix(X), model_fit_dim2.model_fit.α).prob
fitted_density_person = comp_density * latent_class_prob'
fitted_density = mean(fitted_density_person, dims=2)
# Plot data vs fitted distribution
histogram(Y_complete[:, 2]; bins=200, normalize = true, label="Data")
plot!(plot_series, comp_density[:,1], linewidth=2, label="Component 1")
plot!(plot_series, comp_density[:,2], linewidth=2, label="Component 2")
plot!(plot_series, fitted_density, linewidth=2, label="Fitted Density")
title!("Fitted vs Data: Dimension 2 of Y")
savefig(plotsdir("fitted-vs-data-complete-1d-model-dimension-2.png"))

# "Misspecified" Model: local minimum
α_guess = fill(0.0, 2, ncol(X))
model_guess = [LogNormalExpert(2.0, 0.3) InverseGaussianExpert(15, 30)]

model_fit_dim2_alt = fit_LRMoE(Y_2, X, α_guess, model_guess;
                           exact_Y=true, ecm_iter_max=100)
# Should converge in 89 runs with default ϵ

# Model summary
summary(model_fit_dim2_alt)

# Get fitted parameters
model_fit_dim2_alt.model_fit.α
model_fit_dim2_alt.model_fit.comp_dist

# Model performance
model_fit_dim2_alt.loglik
model_fit_dim2_alt.loglik_np
model_fit_dim2_alt.AIC
model_fit_dim2_alt.BIC

# Check if there's label switching
mean.(model_fit_dim2.model_fit.comp_dist)
mean.(model_fit_dim2_alt.model_fit.comp_dist)

var.(model_fit_dim2.model_fit.comp_dist)
var.(model_fit_dim2_alt.model_fit.comp_dist)

# Generate numbers for visualization
plot_series = collect(0.0:0.05:250)
comp_density = hcat([pdf.(expert, plot_series) for expert in model_fit_dim2_alt.model_fit.comp_dist]...)
latent_class_prob = predict_class_prior(Matrix(X), model_fit_dim2_alt.model_fit.α).prob
fitted_density_person = comp_density * latent_class_prob'
fitted_density = mean(fitted_density_person, dims=2)
# Plot data vs fitted distribution
histogram(Y_complete[:, 2]; bins=200, normalize = true, label="Data")
plot!(plot_series, comp_density[:,1], linewidth=2, label="Component 1")
plot!(plot_series, comp_density[:,2], linewidth=2, label="Component 2")
plot!(plot_series, fitted_density, linewidth=2, label="Fitted Density")
title!("Fitted vs Data: Dimension 2 of Y")
savefig(plotsdir("fitted-vs-data-complete-1d-model-dimension-2-alternative.png"))

# Save results
jldsave(datadir("fitted-complete-1d-model-dimension-2.jld2"); model_fit = model_fit_dim2)
jldsave(datadir("fitted-complete-1d-model-dimension-2-alternative.jld2"); model_fit = model_fit_dim2_alt)