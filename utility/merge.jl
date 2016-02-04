numProcs = 4
addprocs(numProcs)
@everywhere using DataFrames

const test_file = "./data/test.csv"
const scores_file = "./previsions/scores_onlyBR.csv" #_onlyBR
const rec_file = "./result/mergedToSend.csv"
const aggregating_function = sum#something like maximum, sum, mean, cazzoneso...
@everywhere const max_items = 5 #how many items to recommend

function main()
    scores = readtable(scores_file)

    mergedPrevisions = mergePrevisions(scores)

    usersToDo = sort(unique(readtable(test_file)[:userId]))
    bestPrevisions = takeBestPrevisions(mergedPrevisions, usersToDo)
    finalResult = convertToResult(bestPrevisions)

    writetable(rec_file, finalResult)
end

###Given a dataframe that contains (file_name, score) open all the files and compute a mixed dataframe of the form (user, item, score) where the score takes into account both the score of the file (how good is the algorithm that produced it) and the score found in the prevision dataframe (how much the algorithm trust that specific prevision).
function mergePrevisions(scores)
    #open everything and concatenate
    previsions = DataFrame()
    min_score = minimum(scores[:score])

    for score_entry in eachrow(scores)
        filename = score_entry[:filename]
        score = score_entry[:score]

        orig = readtable(filename)

        #normalize so that the top is 10
        scale = 10 / maximum(orig[:rating])

        previsions = vcat(previsions, DataFrame(userId = orig[:userId], itemId = orig[:itemId], rating = orig[:rating] * scale * score))
    end

    #merge the scores
    aggregate(previsions, [:userId, :itemId], aggregating_function)
    #rename the last column
    names!(previsions, [:userId, :itemId, :rating])
end

###For every user take the n (5) best recommendations
function takeBestPrevisions(previsions, users)
    result = @sync @parallel (merge) for user in users
        tmp = Dict{Int64, Vector{Int64}}() #user -> items[]

        user_prev = previsions[previsions[:userId] .== user, :]
        sort!(user_prev, cols = :rating, by = x -> -x) #sort in decreasing rating order
        tmp[user] = user_prev[1:max_items, :itemId].data
        tmp
    end

    result
end

###Given a dictionary user -> items[5] return a dataframe (user, "item1 item2 item3 item4 item5")
function convertToResult(previsions)
    result = DataFrame(userId = Int64[], testItems = AbstractString[])

    for user in keys(previsions)
        items = ""
        for item in previsions[user]
            items = string(items, item, " ")
        end
        items = items[1 : end-1] #remove the last space

        push!(result, [user, items])
    end

    result
end

main()
