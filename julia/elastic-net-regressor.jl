import MLJ:predict,fitted_params,fit!
using GLMakie, MLJ,CSV,DataFrames,StatsBase
include("data-processing.jl")

df=get_data()

first(df,10)

y, X = unpack(df, ==(:Price); rng=123);
xtest=df[1:3:end,1:5]
ytest=df[1:3:end,6]

Regressor = @load ElasticNetRegressor pkg=MLJLinearModels
model=Regressor()
mach = machine(model, X, y)|>fit!
fitted_params(mach)
yhat=predict(mach, xtest)



function plot_residue(yhat,ytest)
    res=yhat.-ytest
    fig=Figure(resolution=(1200,500))
    ax=Axis(fig[1,1])
    hlines!(ax, [0],color=(:red, 0.5))
    stem!(ax,res)
    fig
    #save("ushousing-robust-regression-residue.png",fig)
end

#plot_residue(yhat,ytest)

RMS=round(rmsd(yhat, ytest), sigdigits=4)