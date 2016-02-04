#Using the item-item similarity computed before, we provide missing ratings for test users.
#best 0.0003047619047619048 with 3-10-5


const numCores = 4
addprocs(numCores - 1)

using JLD
@everywhere using DataFrames

models_file = "./data/ICF/Dicts.dat"
simil_file = "./data/ICF/simil7.dat"

@everywhere const K = 7 #PARAMETER = how many neighbor to consider (< 20 or change how similarity is computed)
@everywhere const maxK = 20 #We assume that when we open the similarities each item have this number of similarities
@everywhere const minRateToKeep = 7 #minimum vote to keep a prevision
@everywhere const ratingsToKeep = 5
@everywhere const shrink = 5 #PARAMTER = shrink when computing the prediction

test_file = "./data/test.csv"
prev_file = "./previsions/ICF/out.csv"


#Given an array of tuples containing the similarities of an item, modify it keeping only the first K entries
#OK
function keepNearest!(sim:: Array{Tuple{Int64, Float64}})
    deleteat!(sim, (K+1):maxK)
end

#Compute the previsions for the given user. Given an user we look at his rated items, for each item we consider similar items
@everywhere function predictUser(user::Int64, user_urm::Dict{Int64, Float64}, similarities::Dict{Int64, Array{Tuple{Int64, Float64}}}, pop, popTh, bias)
    previsions = Dict{Int64, Tuple{Float64, Float64}}() #item->value

    for rated_item in keys(user_urm)
        known_rating = user_urm[rated_item]

        similars = get(similarities, rated_item, [])
        for sim_line in similars #(item, similarity)
            item = sim_line[1]
            sim_val = sim_line[2]

            if item in keys(user_urm)
                continue
            end
            if (get(pop, item, 0) > popTh)
                continue
            end

            cur_prev = get(previsions, item, (0,0)) #Current prevision

            n = cur_prev[1]
            n += known_rating * sim_val #rating * similarity

            d = cur_prev[2]
            d += sim_val

            previsions[item] = (n,d)
        end
    end


    #now divide
    result = Dict{Int64, Tuple{Float64, Int64}}()
    for item in keys(previsions)
        prev = previsions[item]
        score = prev[1]/(prev[2]+shrink) + bias
        if score > minRateToKeep
            result[item] = (score, get(pop, item, 0))
        end
    end

    keepRatings(result)
end

@everywhere function keepRatings(previsions)
    keep = Array{Tuple{Int64, Float64, Int64},1}() #(item, rating, popularity)

    for item in keys(previsions)
        try
            foo = previsions[item]
        catch
            continue #why the fuck sometimes it fails?
        end
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
        if isnan(arr[i][2]*arr[i][3])
            return i
        end
        if arr[i][3] < lowest
            lowest = arr[i][3]
            pos = i
        end
    end

    pos
end


#Provide ratings for all the given users
#OK
@everywhere function provideRatings(usersToDo::Array{Int64}, urm::Dict{Int64, Dict{Int64, Float64}}, similarities::Dict{Int64, Array{Tuple{Int64, Float64}}}, pop, popTh, glb_avg, u_biases)
        result = DataFrame(userId = Int64[], itemId = Int64[], rating = Float64[])

        for user in usersToDo
            previsions = predictUser(user, urm[user], similarities, pop, popTh[user], glb_avg + u_biases[user])#Array (item , rating, pop)

            for rating in previsions
                push!(result, [user, rating[1], rating[2]])
            end
        end

        result
end

function parallelize(test::Array{Int64}, urm::Dict{Int64, Dict{Int64, Float64}}, similarities::Dict{Int64, Array{Tuple{Int64, Float64}}}, pop, popTh, glb_avg, u_biases)
    test_length = length(test)
    test_length /= numCores

    workers = Array(RemoteRef{Channel{Any}}, numCores-1)

    for i in 2:numCores
        low = ceil(Int64, (i-1)*test_length+1)
        high = ceil(Int64,test_length*i)
        workers[i-1] = @spawn provideRatings(test[low:high], urm, similarities, pop, popTh, glb_avg, u_biases)
    end

    #the first trunk for the running thread
    low = 1
    high =  ceil(Int64,test_length)
    ratings = provideRatings(test[low:high], urm, similarities, pop, popTh, glb_avg, u_biases)

    #now collect
    for i in 1:(numCores-1)
        ratings = vcat(ratings, fetch(workers[i]))
    end

    return ratings
end


test = readtable(test_file)
test = sort(unique(test[:userId].data))

urm = load(models_file, "nor_models")#user->item->rating
pop = load(models_file, "popularities")
popTh = load(models_file, "pop_thresholds")
glb_avg = load(models_file, "glb_avg")
u_biases = load(models_file, "u_biases")

similarities = load(simil_file, "similarities")#item->item->similarity
for item in keys(similarities)
    keepNearest!(similarities[item])
end

prev = parallelize(test, urm, similarities, pop, popTh, glb_avg, u_biases)

nuser = length(unique(prev[:userId]))
ntest = length(test)
nprev = length(prev[1])
println("Provided ",nprev," previsions, to ",nuser," users on ",ntest," test users.")

writetable(prev_file, prev)
