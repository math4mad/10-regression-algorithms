"""
based on https://www.kaggle.com/code/faressayah
/practical-introduction-to-10-regression-algorithm
"""

using GLMakie, MLJ,CSV,DataFrames,StatsBase


function data_prepare(str)
    fetch(str) = str |> d -> CSV.File("./DataSource/$str.csv") |> DataFrame |> dropmissing
    to_ScienceType(d)=coerce(d,Count=>Continuous)
    df = fetch(str)|>to_ScienceType
    return df
end

str="usa_housing"
df=data_prepare(str)

rename!(df, [1 =>:AreaIncome, 2 =>:HouseAge,3 =>:HouseRooms,4 =>:HouseBedromms,5 =>:AreaPopulation])

select!(df,1:6)

"""
    get_data()
    生成 ushouse dataframe data
TBW
"""
function get_data()
    return df
end

export get_data