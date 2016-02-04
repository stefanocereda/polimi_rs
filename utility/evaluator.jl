using DataFrames
#TODO ALL

rec_file = "./result/mergedToSend.csv"
test_file = "./data/test.csv"

const goodVote = 1 #testing purpose

function is_good(item::Int64, user::Int64, test::DataFrame)
    try
        rated = test[test[:userId] .== user, [:itemId,:rating]]
        this = rated[rated[:itemId] .== item, [:rating]]

        if this[1,1] >= goodVote
            return true
        else
            return false
        end
    catch #rating not present means bad evaluation
        return false
    end
end

function evaluate(recommendations::DataFrame, test::DataFrame)

    sum = 0
    num = 0

    for rec in eachrow(recommendations)
        user = rec[:userId]
        items = split(rec[:testItems], " ")

        for item in items
            num = num + 1
            if is_good(parse(Int64,item), user, test)
                sum = sum + 1
            end
        end
    end

    mean = (sum*0.20)/num
    return mean
end

recommendations_df = readtable(rec_file)
test_df = readtable(test_file)
println(evaluate(recommendations_df, test_df))

println("\a")
