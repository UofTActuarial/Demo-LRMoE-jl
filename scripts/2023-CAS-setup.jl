using DrWatson
@quickactivate "LRMoEjl Demo"

using CategoricalArrays, DataFrames, JLD2, RCall

R"""
library(CASdatasets)
data(ausprivauto0405)
"""

df = @rget ausprivauto0405

save_subfolder = datadir("2023-CAS-demo")

if !(isdir(datadir(save_subfolder)))
    mkpath(datadir(save_subfolder))
end

filename = datadir(save_subfolder, "ausprivauto0405.jld2")
jldsave(filename; df=df)