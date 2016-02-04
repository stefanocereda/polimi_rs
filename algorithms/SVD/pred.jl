#Using the computed vectors, compute previsions

const numProcs = 4
addprocs(numProcs)

@everywhere const ratingsToKeep = 5
@everywhere const min_rat_to_save = 5

using DataFrames
using JLD

const models_file = "./data/SVD/values_15_7_100_1000_01_002_0001.jld" #forse è meglio togliere 1.5
const test_file = "./data/test.csv"
const previsions_file = "./previsions/SVD/out.csv"

function main()
    urm_dict = load(models_file, "urm")
    pop_dict = load(models_file, "pop")
    pop_th_dict = load(models_file, "pop_th")
    global_avg = load(models_file, "g_avg")
    items_avg = load(models_file, "i_avg")
    users_avg = load(models_file, "u_avg")
    u_vects = load(models_file, "u_vect")
    i_vects = load(models_file, "i_vect")

    test_users = unique(readtable(test_file)[:userId])

    previsions = @sync @parallel (vcat) for user in test_users
        tmp = Tuple{Int64, Dict{Int64, Float64}}[] #(user, item->value)[]

        push!(tmp, (user, provideUser(global_avg+users_avg[user], items_avg, u_vects[user], i_vects, pop_dict, pop_th_dict[user], keys(urm_dict[user]))))

        tmp
    end

    prev = prevToDF(previsions)

    nuser = length(unique(prev[:userId]))
    ntest = length(test_users)
    nprev = length(prev[1])
    println("Provided ",nprev," previsions, to ",nuser," users on ",ntest," test users.")

    writetable(previsions_file, prev)
end

###Compute recommendations for the given user.
@everywhere function provideUser(bias, items_avg, user_vect, i_vects, pop_dict, pop_th, ratedItems)
    userPrevisions = Tuple{Int64, Float64}[] #item, value

    for item in keys(i_vects)
        if get(pop_dict, item, 0) >= pop_th
            continue
        end
        if item in ratedItems
            continue
        end

        rating = bias + items_avg[item] + dot(user_vect, i_vects[item])

        if rating > min_rat_to_save
            insert_prevision!(userPrevisions, item, rating)
        end
    end

    previsionsToDict(userPrevisions)
end

###Given an array of prevision return a dict item->value
#As value we consider similarity*popularity
@everywhere function previsionsToDict(vect)
    ret = Dict{Int64, Float64}()
    for prev in vect
        ret[prev[1]] = prev[2]
    end
    ret
end

@everywhere function insert_prevision!(vector, key, value)
    if length(vector) < ratingsToKeep
        push!(vector, (key, value))
    else #otherwise search the worst
        worst = searchWorst(vector)
        #and eventually discard
        if vector[worst][2] < value
            deleteat!(vector, worst)
            push!(vector, (key, value))
        end
    end
end

#Search the worst element according to sim*pop
@everywhere function searchWorst(arr)
    lowest = 10000
    pos = 0

    for i = 1:ratingsToKeep
        if isnan(arr[i][2])
            return i
        end
        if arr[i][2] < lowest
            lowest = arr[i][2]
            pos = i
        end
    end

    pos
end

function prevToDF(previsions)
    ret = DataFrame(userId = Int64[], itemId = Int64[], rating = Float64[])
    for prev in previsions
        user = prev[1]
        dict = prev[2]
        for item in keys(dict)
            push!(ret, [user, item, dict[item]])
        end
    end
    ret
end

main()
