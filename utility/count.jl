#Given the mergedToSend and a prevision return the number of recommendations taken from that prevision

const result = "./result/mergedToSend.csv"
const scores = "./previsions/scores.csv"

using DataFrames

scores_df = readtable(scores)
result_df = readtable(result)

for prev in scores_df[:filename]
    prev_df = readtable(prev)

    tot = 0
    for user in result_df[:userId]
        prevItems = prev_df[prev_df[:userId] .== user, :itemId]
        recItems = result_df[result_df[:userId] .== user, :testItems][1]

        for item in split(recItems, " ")
            if parse(Int, item) in prevItems
                tot +=1
            end
        end
    end

    tot1 = length(result_df[1])*5
    tot2 = length(prev_df[1])

    println(prev, ":", tot, " ", tot1, " ", tot2)
end
