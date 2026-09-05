"""
based on https://www.kaggle.com/code/faressayah
/practical-introduction-to-10-regression-algorithm

绘制协方差矩阵图
"""

using GLMakie, MLJ,CSV,DataFrames,StatsBase
include("data-processing.jl")

df=get_data()

"""
    plot_corr(df)
    数据的相关矩阵
TBW
"""
function plot_cor(df)
    label=names(df)|>Array
    _,cols=size(df)
    colors = [:orange, :lightgreen, :purple,:lightblue,:red,:green]
    fig = Figure(resolution=(cols*350, cols*350))
    function plot_diag(i)

        ax = Axis(fig[i, i])
        #push!(axs, ax)
        density!(ax, df[:, i]; color=(colors[i], 0.5),
                strokewidth=1.25, strokecolor=colors[i])
    end

    function plot_cor(i, j)
        ax = Axis(fig[i, j])
        scatter!(ax, df[:, i], df[:, j]; color=colors[j])
    end
    
    
    function plot_pair()
        [(i == j ? plot_diag(i) : plot_cor(i, j)) for i in 1:cols, j in 1:cols]
    end
    
    function add_xy_label()
        for i in 1:cols
            Axis(fig[cols, i], xlabel=label[i],)
            Axis(fig[i, 1], ylabel=label[i],)
        end
    end

    plot_pair()
    add_xy_label()
    fig

    #save("us-housing-cor.png",fig)

end
#plot_corr(df)

export plot_cor;


