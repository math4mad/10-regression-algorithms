"""
Neural Network Regression 的配置方法和其他回归方法不同,要注意
"""

import MLJ:fit!,predict
using MLJ
import MLJFlux
using Flux


#====================data processing=============================#

    include("data-processing.jl")

    df=get_data()

    #first(df,10)
    y, X = unpack(df, ==(:Price); rng=123);

    (X, Xtest), (y, ytest) = partition((X, y), 0.7, multi=true);
#====================data processing end=========================#


#=====================NeuralNetwork Reg  workflow================#

    builder = MLJFlux.@builder begin
        init=Flux.glorot_uniform(rng)
        Chain(
            Dense(n_in, 64, relu, init=init),
            Dense(64, 32, relu, init=init),
            Dense(32, n_out, init=init),
        )
    end


    NeuralNetworkRegressor = @load NeuralNetworkRegressor pkg=MLJFlux
    model = NeuralNetworkRegressor(
        builder=builder,
        rng=123,
        epochs=20
    )


    pipe = Standardizer |> TransformedTargetModel(model, target=Standardizer)
    mach = machine(pipe, X, y)
    fit!(mach, verbosity=2)
    report(mach).transformed_target_model_deterministic.model.training_losses
    evaluate!(mach, resampling=CV(nfolds=5), measure=l2)
    fit!(mach) ## train on `(X, y)`
    yhat = predict(mach, Xtest)
    l2(yhat, ytest)  |> mean
#====================NeuralNetwork Reg   workflow end=============#


