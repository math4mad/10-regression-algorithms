"""
A model type for constructing a ϵ-support vector regressor, based on LIBSVM.jl, and implementing the MLJ model interface.
"""

import MLJ:predict,fitted_params,fit!
import LIBSVM
using GLMakie, MLJ,CSV,DataFrames,StatsBase
include("data-processing.jl")

function data_process()
    df=get_data()
    y, X = unpack(df, ==(:Price); rng=123);
    (X, Xtest), (y, ytest) = partition((X, y), 0.7, multi=true);
    return (X, Xtest), (y, ytest) 
end

(X, Xtest), (y, ytest)=data_process()


#= work flow=#

function train_model(X,y)
    EpsilonSVR = @load EpsilonSVR pkg=LIBSVM            
    model = EpsilonSVR(kernel=LIBSVM.Kernel.Polynomial)
    mach = machine(model, X, y) |> fit!
    return mach
end

mach=train_model(X,y)
yhat = predict(mach, Xtest)

#RMS=round(rmsd(yhat, ytest), sigdigits=4)
l2(yhat, ytest)|>mean  #1.186392


