module WassersteinMetric

using JuMP
using Clp
using Graphs
using ProgressMeter

export get_wasserstein_metric
export getwsdm

include("wasserstein.jl")

end # module
