const train_file = "./data/train.csv"
const test_file = "./data/test.csv"
const prev_file = "./previsions/tesPop.csv"

using DataFrames

#Given the original urm compute and remove global avg, user offset and item bias
function removeAverages(urm)
    #GLOBAL OFFSET
    urm1 = DataFrame(userId = Int64[], itemId = Int64[], rating = Float64[])
    global_mean = mean(urm[:rating])
    for line in eachrow(urm)
        push!(urm1, [line[:userId]; line[:itemId]; line[:rating] - global_mean])
    end
    urm = urm1

    #ITEM BIAS
    urm1 = DataFrame(userId = Int64[], itemId = Int64[], rating = Float64[])
    i_biases = Dict{Int64, Float64}()
    avg = aggregate(urm, :itemId, mean)
    for line in eachrow(avg)
        i_biases[line[:itemId]] = line[:rating_mean]
    end
    for line in eachrow(urm)
        item = line[:itemId]
        push!(urm1, [line[:userId]; item; line[:rating] - i_biases[item]])
    end
    urm = urm1

    #USER OFFSET
    urm1 = DataFrame(userId = Int64[], itemId = Int64[], rating = Float64[])
    u_biases = Dict{Int64, Float64}()
    avg = aggregate(urm, :userId, mean)
    for line in eachrow(avg)
        u_biases[line[:userId]] = line[:rating_mean]
    end
    for line in eachrow(urm)
        user = line[:userId]
        push!(urm1, [user; line[:itemId]; line[:rating] - u_biases[user]])
    end
    urm = urm1

    (global_mean, i_biases, u_biases, urm)
end

function computePopularity(urm)
    res = Dict{Int64, Int64}()

    for line in eachrow(urm)
        item = line[:itemId]
        pop = get(res, item, 0)
        pop += 1
        res[item] = pop
    end

    res
end

function computeUser(user, averages, popularities)
    glob = averages[1]
    itm = averages[2]
    usr = averages[3][user]

    positiveRatings = DataFrame(itemId = Int64[], rating = Float64[], pop=Int64[])

    for item in keys(itm)
        prev = glob + itm[item] + usr
        if prev > 8
            push!(positiveRatings, [item, prev, popularities[item]])
        end
    end

    result = Dict{Int64, Float64}()
    sort!(positiveRatings, cols=[:pop])
    for i in 1:5
        line = positiveRatings[i,:]
        result[line[:itemId][1]] = line[:rating][1]
    end

    result
end


#for all the usersToDo, select the 5 least popular items with a positive predicted rating
function giveRatings(usersToDo, averages, popularities)
    result = DataFrame(userId=Int64[], itemId=Int64[], rating=Int64[])

    for user in usersToDo
        prev = computeUser(user, averages, popularities)
        for item in keys(prev)
            push!(result, [user, item, prev[item]])
        end
    end

    result
end


urm = readtable(train_file)
avg = removeAverages(urm)
pop = computePopularity(urm)
test = readtable(test_file)
test = sort(unique(test[:userId].data))
result = giveRatings(test, avg, pop)
