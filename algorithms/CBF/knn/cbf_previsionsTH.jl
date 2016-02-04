#Using the item-item similarity computed before, we provide missing ratings for test users.

const numCores = 4
addprocs(numCores - 1)

using JLD
@everywhere using DataFrames

const train_file = "./data/train.csv"
const models_file = "./data/CBF/knn/Dicts.jld"
const simil_file = "./data/CBF/knn/simil3_v2.jld"

@everywhere const K = 6 #PARAMETER = how many neighbor to consider (< 20 or change how similarity is computed)
@everywhere const maxK = 10 #We assume that when we open the similarities each item have this number of similarities
@everywhere const shrink = 4 #PARAMETER = shrink when computing the prediction
@everywhere const ratingsToKeep = 5

test_file = "./data/test.csv"
prev_file = "./previsions/CBF/knn/cbf_364_v2.csv"


#Given an array of tuples containing the similarities of an item, modify it keeping only the first K entries
#OK
function keepNearest!(sim)
    deleteat!(sim, (K+1):maxK)
end

#Compute the previsions for the given user. Given an user we look at his rated items, for each item we consider similar items
@everywhere function predictUser(user, user_urm, similarities, popularity, popTh)
    previsions = Dict{Int64, Tuple{Float64, Float64}}() #item->(num,den)

    for item in keys(user_urm) #rated items
        i_rat = user_urm[item]

        sim = get(similarities, item, [])
        for sim_entry in sim #items similar to rated one. Array (int, number)
            oth_it = sim_entry[1]
            sim_val = sim_entry[2]

            if (oth_it in keys(user_urm))#already rated
                continue
            end
            if get(popularity, oth_it, 0) >= popTh
                continue
            end

            current_prev = get(previsions, oth_it, (0.0, 0.0))
            num = current_prev[1]
            num += i_rat * sim_val #sim * rating
            den = current_prev[2]
            den += sim_val
            previsions[oth_it] = (num, den)
        end
    end

    #now divide
    result = Dict{Int64, Float64}()
    for item in keys(previsions)
        prev = previsions[item]
        prev = prev[1]/(prev[2]+shrink)
        result[item] = prev
    end

    keepRatings(result)
end

@everywhere function keepRatings(previsions)
    keep = Array{Tuple{Int64, Float64},1}() #(item, value)

    for item in keys(previsions)
        value = previsions[item]

        #if we have space push in the array
        if length(keep) < ratingsToKeep
            push!(keep, (item, value))
        else #otherwise search the worst
            worst = searchWorst(keep)
            #and eventually discard
            if keep[worst][2] < value
                deleteat!(keep, worst)
                push!(keep, (item, value))
            end
        end
    end

    keep
end


@everywhere function searchWorst(arr)
#2 = rating
#3 = popularity
    lowest = 10000
    pos = 0

    for i = 1:ratingsToKeep
        if arr[i][2] < lowest
            lowest = arr[i][2]
            pos = i
        end
    end

    pos
end


#Provide ratings for all the given users
#OK
@everywhere function provideRatings(usersToDo, urm, similarities, popularity, popTh)
    result = DataFrame(userId = Int64[], itemId = Int64[], rating = Float64[])

    for user in usersToDo
        previsions = predictUser(user, urm[user], similarities, popularity, popTh[user])

        for prev in previsions
            push!(result, [user, prev[1], prev[2]])
        end
    end

    result
end

function parallelize(test, urm, similarities, popularity, popTh)
    test_length = length(test)
    test_length /= numCores

    workers = Array(RemoteRef{Channel{Any}}, numCores-1)

    for i in 2:numCores
        low = ceil(Int64, (i-1)*test_length+1)
        high = ceil(Int64,test_length*i)
        workers[i-1] = @spawn provideRatings(test[low:high], urm, similarities, popularity, popTh)
    end

    #the first trunk for the running thread
    low = 1
    high =  ceil(Int64,test_length)
    ratings = provideRatings(test[low:high], urm, similarities, popularity, popTh)

    #now collect
    for i in 1:(numCores-1)
        ratings = vcat(ratings, fetch(workers[i]))
    end

    return ratings
end


test = readtable(test_file)
test = sort(unique(test[:userId].data))

urm = load(models_file, "models")
pops = load(models_file, "popularities")
popTh = load(models_file, "pop_thresholds")

similarities = load(simil_file, "similarities")
for item in keys(similarities)
    keepNearest!(similarities[item])
end

result = parallelize(test, urm, similarities, pops, popTh)
nuser = length(unique(result[:userId]))
ntest = length(test)
nprev = length(result[1])
println("Provided ",nprev," previsions, to ",nuser," users on ",ntest," test users.")
writetable(prev_file, result)
