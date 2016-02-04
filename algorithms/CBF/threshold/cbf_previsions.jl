#Using the item-item similarity computed before, we provide missing ratings for test users.

const numCores = 4
addprocs(numCores - 1)

using JLD
@everywhere using DataFrames

const train_file = "./data/train.csv"
models_file = "./data/CBF/threshold/Dicts.dat"
simil_file = "./data/CBF/threshold/simil3t01.dat"

@everywhere const shrink = 0.05 #PARAMTER = shrink when computing the prediction
@everywhere const ratingsToKeep = 5
@everywhere const min_rate_to_save = 7.5
@everywhere const ordering = 3 #3 = order on pop, 2 = order on rating

test_file = "./data/test.csv"
prev_file = "./previsions/CBF/threshold/Ps3t01s005m75TH.csv"


#Compute the previsions for the given user. Given an user we look at his rated items, for each item we consider similar items
@everywhere function predictUser(user, user_urm, similarities, popularity, pop_th)
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
            if get(popularity, oth_it, 0) > pop_th
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
    result = Dict{Int64, Tuple{Float64, Int64}}()
    for item in keys(previsions)
        prev = previsions[item]
        prev = prev[1]/(prev[2]+shrink)
        if (prev > min_rate_to_save)
            result[item] = (prev, get(popularity, item, 0))
        end
    end

    keepRatings(result)
end

@everywhere function keepRatings(previsions)
    keep = Array{Tuple{Int64, Float64, Int64},1}() #(item, rating, popularity)

    for item in keys(previsions)
        rate = previsions[item][1]
        pop = previsions[item][2]

        #if we have space push in the array
        if length(keep) < ratingsToKeep
            push!(keep, (item, rate, pop))
        else #otherwise search the worst
            worst = searchWorst(keep)
            #and eventually discard
            if keep[worst][3] < pop
                deleteat!(keep, worst)
                push!(keep, (item, rate, pop))
            end
        end
    end

    keep
end

@everywhere function searchWorst(arr)
    lowest = 10000
    pos = 0

    for i = 1:ratingsToKeep
        if isnan(arr[i][ordering])
            return i
        end
        if arr[i][ordering] < lowest
            lowest = arr[i][ordering]
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

function computeSimThresh(urm, pop)
    thr = Dict{Int64, Int64}()

    for user in keys(urm)
        min = 100000
        for item in keys(urm[user])
            if urm[user][item] < 8
                continue
            end
            if pop[item] < min
                min = pop[item]
            end
        end
        thr[user] = min
    end
    thr
end


urm = readtable(train_file)
popularities = computePopularity(urm)

test = readtable(test_file)
test = sort(unique(test[:userId].data))

urm = load(models_file, "urm")
popTh = computeSimThresh(urm, popularities)

similarities = load(simil_file, "similarities")

previsions = parallelize(test, urm, similarities, popularities, popTh)
writetable(prev_file, previsions)
