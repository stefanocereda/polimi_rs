#Using the user-user similarity computed from user models, we provide missing ratings for test users.

using JLD
@everywhere using DataFrames

train_file = "./data/train.csv"
test_file = "./data/test.csv"
sim_file = "./data/UCF/similaritiesSH7.dat" #Parameter = the shrinkage used computing the similarity
models_file = "./data/UCF/userModels.dat"

@everywhere const K = 7 #Parameter = how many neighbor to consider (< 20 or change how similarity is computed)
@everywhere const maxK = 20 #We assume that when we open the similarities each user have this number of similarities
@everywhere const minRateToKeep = 7 #minimum vote to keep a prevision
@everywhere const ratingsToKeep = 5
@everywhere const shrink = 5

prev_file = "./previsions/UCF/Sh7K7min7Sh5K5_b.csv"

#Given an array of tuples containing the similarities of an user, modify it keeping only the first K entries
#OK
function keepNearest!(sim:: Array{Tuple{Int64, Float64}})
    deleteat!(sim, (K+1):maxK)
end

#Compute the previsions for the given user
#OK
@everywhere function provideRatings(user::Int64, models::Dict{Int64, Dict{Int64, Float64}}, similarity::Array{Tuple{Int64, Float64}}, pop, popTh, bias)
    previsions = Dict{Int64, Tuple{Float64, Float64}}()
    user_model = models[user]

    for sim_line in similarity
        other_user = sim_line[1]
        sim_value = sim_line[2]

        other_model = models[other_user]
        for item in keys(other_model)
            if item in keys(user_model)
                continue
            end
            if (get(pop, item, 0) > popTh)
                continue
            end

            value = get(previsions, item, (0,0))
            previsions[item] = (value[1]+other_model[item]*sim_value, value[2] + sim_value)
        end
    end

    #now divide
    result = Dict{Int64, Tuple{Float64, Int64}}()
    for item in keys(previsions)
        prev = previsions[item]
        score = (prev[1]/(prev[2]+shrink))+bias
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


function dictToDF(dict)
    ret = DataFrame(userId = Int64[], itemId = Int64[], rating = Float64[])
    for user in keys(dict)
        for rating in dict[user]
            push!(ret, [user, rating[1], rating[2]])
        end
    end
    ret
end

urm = readtable(train_file)
test = readtable(test_file)
models = load(models_file, "nor_models")
popularity = load(models_file, "popularities")
popTh = load(models_file, "pop_thresholds")
glb_avg = load(models_file, "glb_avg")
u_biases = load(models_file, "u_biases")
similarities = load(sim_file, "similarity")

test = sort(unique(test[:userId].data))
for user in keys(similarities)
    keepNearest!(similarities[user])
end

previsions = Dict{Int64, Array{Tuple{Int64, Float64, Int64},1}}()
for user in test
    previsions[user] = provideRatings(user, models, similarities[user], popularity, popTh[user], u_biases[user] + glb_avg)
end

result = dictToDF(previsions)
nuser = length(unique(result[:userId]))
ntest = length(test)
nprev = length(result[1])
println("Provided ",nprev," previsions, to ",nuser," users on ",ntest," test users.")

writetable(prev_file, result)
