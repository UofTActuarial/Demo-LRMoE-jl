using DrWatson
@quickactivate "LRMoEjl Demo"

# Prerequisites (already included by DrWatson):
# JLD2, LRMoE, DataFrames, Distributions, Plots, Random

using DataFrames, Distributions, LRMoE, Plots, Random

# Random seed for reproducible results
Random.seed!(7777)
sample_size = 10_000

# Simulate covariates
intercept = fill(1.0, sample_size)
sex = rand(Binomial(1, 0.50), sample_size)
aged = rand(Uniform(20, 80), sample_size)
agec = rand(Uniform(0, 10), sample_size)
region = rand(Binomial(1, 0.50), sample_size)

X = DataFrame(; intercept=intercept, sex=sex,
              aged=aged, agec=agec, region=region)

# True model
α = [-0.5 1.0 -0.05 0.1 1.25;
      0.0 0.0   0.0 0.0  0.0]

comp_dist = [PoissonExpert(6.0)         ZINegativeBinomialExpert(0.20, 30, 0.50);
             LogNormalExpert(4.0, 0.3)  InverseGaussianExpert(20, 20)]

Y_complete = LRMoE.sim_dataset(α, X, comp_dist)

# A view of simulated complete DataFrame
plot(; size=(800, 600))
histogram(Y_complete[:, 1]; bins=100, label="")
title!("Dimension 1 of Y")
savefig(plotsdir("histogram-full-data-dimension-1.png"))

plot(; size=(800, 600))
histogram(Y_complete[:, 2]; bins=100, label="")
title!("Dimension 2 of Y")
savefig(plotsdir("histogram-full-data-dimension-2.png"))

# Save complete data
jldsave(datadir("simulation-complete-data.jld2"); X=X, Y_complete=Y_complete)

# Artificailly create incomplete data by censoring and truncation
# First block: 1~6000
X_obs = X[1:6000, :]

tl_1 = fill(0.0, 6000)
yl_1 = Y_complete[1:6000, 1]
yu_1 = Y_complete[1:6000, 1]
tu_1 = fill(Inf, 6000)

tl_2 = fill(0.0, 6000)
yl_2 = Y_complete[1:6000, 2]
yu_2 = Y_complete[1:6000, 2]
tu_2 = fill(Inf, 6000)

# Second block: 6001~8000
keep_idx = Y_complete[6001:8000, 2] .>= 5
keep_length = sum(keep_idx) # 1846 out of 2000

append!(X_obs, X[6001:8000, :][keep_idx, :])

append!(tl_1, fill(0.0, keep_length))
append!(yl_1, Y_complete[6001:8000, 1][keep_idx])
append!(yu_1, Y_complete[6001:8000, 1][keep_idx])
append!(tu_1, fill(Inf, keep_length))

y_temp = Y_complete[6001:8000, 2][keep_idx]
append!(tl_2, fill(5.0, keep_length))
append!(yl_2, Y_complete[6001:8000, 2][keep_idx])
append!(yu_2, Y_complete[6001:8000, 2][keep_idx])
append!(tu_2, fill(Inf, keep_length))

# Third block: 8001~10000
append!(X_obs, X[8001:10000, :])

append!(tl_1, fill(0.0, 2000))
append!(yl_1, Y_complete[8001:10000, 1])
append!(yu_1, Y_complete[8001:10000, 1])
append!(tu_1, fill(Inf, 2000))

y_temp = Y_complete[8001:10000, 2]
censor_idx = y_temp .>= 100.0 # 33 out of 2000
yl_temp = copy(y_temp)
yl_temp[censor_idx] .= 100
yu_temp = copy(y_temp)
yu_temp[censor_idx] .= Inf
append!(tl_2, fill(0.0, 2000))
append!(yl_2, yl_temp)
append!(yu_2, yu_temp)
append!(tu_2, fill(Inf, 2000))

# Put things together
Y_obs = DataFrame(; tl_1=tl_1, yl_1=yl_1, yu_1=yu_1, tu_1=tu_1,
                  tl_2=tl_2, yl_2=yl_2, yu_2=yu_2, tu_2=tu_2)

# A view of incomplete data
Y_obs[[1, 6010, 7913], :]

# Save incomplete data
jldsave(datadir("simulation-incomplete-data.jld2"); X_obs=X_obs, Y_obs=Y_obs)
