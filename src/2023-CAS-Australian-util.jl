module australian_auto_util_jl

using CategoricalArrays, DataFrames, Distributions, DrWatson, GLM, LRMoE, StatsModels,
      StatsPlots

export
       load_aus_auto_jld,
       generate_aus_auto_LRMoE_data,
       predict_aus_auto_glm_distribution,
       predict_aus_auto_LRMoE_distribution,
       generate_pmf_comparison,
       plot_pmf_comparison

function load_aus_auto_jld()
    # load raw dataframe
    filename = datadir("2023-CAS-demo", "ausprivauto0405.jld2")
    df = load(filename, "df")
    # set categorical variables
    return transform!(df,
                      :VehAge => (x -> CategoricalArray(x;
                      levels=["youngest cars", "young cars", "old cars", "oldest cars"],
                      ordered=true))
                      => :VehAge,
                      :Gender => (x -> CategoricalArray(x;
                      levels=["Female", "Male"],
                      ordered=true))
                      => :Gender,
                      :DrivAge => (x -> CategoricalArray(x;
                      levels=["youngest people", "young people", "working people",
                      "older work. people", "old people", "oldest people"],
                      ordered=true))
                      => :DrivAge)
end

function generate_aus_auto_LRMoE_data(fml, df)
    df_fml_schema = StatsModels.apply_schema(fml, StatsModels.schema(fml, df))
    # get y and X
    y, X = StatsModels.modelcols(df_fml_schema, df)
    X = hcat(fill(1, length(y)), X)
    # convert y to a matrix, which is needed for LRMoE
    y = reshape(y, length(y), 1)
    # keep track of the column names
    y_col, X_col = StatsModels.coefnames(df_fml_schema)
    X_col = ["Intercept"; X_col]
    return y, X, y_col, X_col
end

function predict_aus_auto_glm_distribution(model, df)
    df_pred = DataFrame()
    λ_pred = predict(model, df; offset=log.(df.Exposure))
    # 0-3 claims
    for n in [0, 1, 2, 3]
        prob_n = mean(map(λ -> pdf(Poisson(λ), n),
                          predict(model, df; offset=log.(df.Exposure))))
        append!(df_pred, DataFrame(; ClaimNb=n, ClaimNb_p_GLM=prob_n))
    end
    # 4+ claims grouped together
    prob_4plus = 1 - sum(df_pred.ClaimNb_p_GLM[1:4])
    append!(df_pred, DataFrame(; ClaimNb=4, ClaimNb_p_GLM=prob_4plus))
    return df_pred
end

function predict_aus_auto_LRMoE_distribution(model, X, exposure)
    df_pred = DataFrame()
    # predicted latent component probabilities
    prob_component = predict_class_prior(X, model.α).prob
    # predicted latent component λ, scaled by policy exposure
    λ_component = mean.(model.comp_dist) .* exposure
    # 0-3 claims
    for n in [0, 1, 2, 3]
        pmf_component = map(λ -> pdf(Poisson(λ), n), λ_component)
        pmf_weighted = sum(pmf_component .* prob_component; dims=2)
        prob_n = mean(pmf_weighted)
        append!(df_pred, DataFrame(; ClaimNb=n, ClaimNb_p_LRMoE=prob_n))
    end
    # 4+ claims grouped together
    prob_4plus = 1 - sum(df_pred.ClaimNb_p_LRMoE[1:4])
    append!(df_pred, DataFrame(; ClaimNb=4, ClaimNb_p_LRMoE=prob_4plus))
    return df_pred
end

function generate_pmf_comparison(df_empirical, df_glm, df_LRMoE)
    df_pmf_comparison = innerjoin(df_empirical, df_glm; on=:ClaimNb)
    df_pmf_comparison.pct_error_GLM = round.((df_pmf_comparison.ClaimNb_p_GLM -
                                              df_pmf_comparison.ClaimNb_Freq) ./
                                             df_pmf_comparison.ClaimNb_Freq * 100,
                                             digits=2)
    df_pmf_comparison = innerjoin(df_pmf_comparison, df_LRMoE; on=:ClaimNb)
    df_pmf_comparison.pct_error_LRMoE = round.((df_pmf_comparison.ClaimNb_p_LRMoE -
                                                df_pmf_comparison.ClaimNb_Freq) ./
                                               df_pmf_comparison.ClaimNb_Freq * 100,
                                               digits=2)
    return df_pmf_comparison
end

function plot_pmf_comparison(df_pmf_comparison)
    logpmf_Emp = log10.(df_pmf_comparison.ClaimNb_Freq)
    logpmf_GLM = log10.(df_pmf_comparison.ClaimNb_p_GLM)
    logpmf_LRMoE = log10.(df_pmf_comparison.ClaimNb_p_LRMoE)

    return groupedbar([logpmf_GLM logpmf_Emp logpmf_LRMoE];
                      bar_position=:dodge,
                      bar_width=0.5,
                      xticks=([0, -1, -2, -3, -4, -5], [0, -1, -2, -3, -4, -5]),
                      yticks=([1, 2, 3, 4, 5], ["0", "1", "2", "3", "4+"]),
                      xlabel="log(Probability Mass Function)",
                      ylabel="Claim Count",
                      grid=false,
                      label=["GLM" "Empirical" "LRMoE"],
                      legend=:bottomleft,
                      title="Distribution of Claim Frequency",
                      orientation=:horizontal,
                      size=(750, 500))
end

end
