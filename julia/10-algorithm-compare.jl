"""
比较始终算法的 RMSD
目前的 MLJ  里没有 Polynomail 方法, 所以只有 9 种

"""

import MLJ:predict,fitted_params,fit!
using GLMakie, MLJ,CSV,DataFrames,StatsBase
include("data-processing.jl")

df=get_data()
y, X = unpack(df, ==(:Price); rng=123);
(X, xtest), (y, ytest) = partition((X, y), 0.7, multi=true);


#======================= model load=================================================#
"""
    LinearMolesCollections()

    ## 定义返回模型字典的函数
    ```
    models=Dict(:LinearRegressor=>LinearRegressor,:RobustRegressor=>RobustRegressor,
                :RidgeRegressor=>RidgeRegressor,:LassoRegressor=>LassoRegressor,
                :ElasticNetRegressor=>ElasticNetRegressor,:SGDRegressor=>SGDRegressor,
                :NeuralNetworkRegressor=>NeuralNetworkRegressor,:RandomForestRegressor=>RandomForestRegressor,
                :NuSVRRegressor=>NuSVRRegressor
    )
    ```
   return  models,models_keys,models_vals,num_dict

   `👨‍🚀math4mads`
"""
function LinearMolesCollections()
    LinearRegressor = @load LinearRegressor pkg=MLJLinearModels
    RobustRegressor = @load RobustRegressor pkg=MLJLinearModels
    RidgeRegressor = @load RidgeRegressor pkg=MLJLinearModels
    LassoRegressor = @load LassoRegressor pkg=MLJLinearModels
    ElasticNetRegressor = @load ElasticNetRegressor pkg=MLJLinearModels
    SGDRegressor = @load SGDRegressor pkg=MLJScikitLearnInterface
    NeuralNetworkRegressor = @load NeuralNetworkRegressor pkg=MLJFlux
    RandomForestRegressor = @load RandomForestRegressor pkg=DecisionTree
    NuSVRRegressor= @load NuSVR pkg=LIBSVM

    #= models=Dict(:LinearRegressor=>LinearRegressor,:RobustRegressor=>RobustRegressor,
                :RidgeRegressor=>RidgeRegressor,:LassoRegressor=>LassoRegressor,
                :ElasticNetRegressor=>ElasticNetRegressor,:SGDRegressor=>SGDRegressor,
                :NeuralNetworkRegressor=>NeuralNetworkRegressor,:RandomForestRegressor=>RandomForestRegressor,
                :NuSVRRegressor=>NuSVRRegressor
    ) =#
    
   #NeuralNetworkRegressor 的方法和其他回归方法配置不同
    models=Dict(:LinearRegressor=>LinearRegressor,:RobustRegressor=>RobustRegressor,
                :RidgeRegressor=>RidgeRegressor,:LassoRegressor=>LassoRegressor,
                :ElasticNetRegressor=>ElasticNetRegressor,:SGDRegressor=>SGDRegressor,
                :RandomForestRegressor=>RandomForestRegressor,
                :NuSVRRegressor=>NuSVRRegressor
    )

    models_keys=keys(models)
    models_vals=values(models)
    num_dict=Dict(zip([1:9...],models_vals))

    return models,models_keys,models_vals,num_dict
end
#======================= model load end   ==========================================#

models,models_keys,_,_=LinearMolesCollections()

function _fit(data::Tuple,key)
         X,y,xtest,ytest=data
        #@info "$(key) model"
        model=models[key]()
        mach = machine(model, X, y)|>fit!
        yhat=predict(mach, xtest)
        RMS=round(rmsd(yhat, ytest), sigdigits=4)
        return  RMS
end
data=(X,y,xtest,ytest)
RMS=[_fit(data,key) for key in models_keys]

for (k,v) in zip(models_keys,RMS)
    @info "$(k)=>$(v)"
end















