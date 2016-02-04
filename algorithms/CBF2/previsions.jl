###For every user we consider its "taste vector", and compare it with the vector of
#all the available items comparing the similarity. The recommendations are based on
#the similarity and on the popularity of the items (otherwise it becomes hard to
#recommend a movie that the user actually saw)

const numProcs = 4
addprocs(numProcs)

@everywhere const shrink = 8
@everywhere const ratingsToKeep = 5
@everywhere const min_sim_to_save = 0
#QUESTA È IN FONDO AL FILE (ALTRIMENTI NON SA COSA È)
#@everywhere const similarity = pearson or cosine

using DataFrames
using JLD

const models_file = "./data/CBF2/Dicts_vote_v2.jld" #forse è meglio togliere 1.5
const test_file = "./data/test.csv"
const previsions_file = "./previsions/CBF2/vote_s8_v2.csv"

function main()
    itemVectors = load(models_file, "itemWeights")
    userVectors = load(models_file, "userWeights")
    popularity = load(models_file, "popularities")
    pop_th = load(models_file, "pop_th")
    urm_dict = load(models_file, "urm_dict")

    test_users = unique(readtable(test_file)[:userId])

    previsions = @sync @parallel (vcat) for user in test_users
        tmp = Tuple{Int64, Dict{Int64, Float64}}[] #(user, item->value)[]

        uV = get(userVectors, user, Dict{Int64, Float64}())

        push!(tmp, (user, provideUser(uV, itemVectors, popularity, pop_th[user], keys(urm_dict[user]))))

        tmp
    end

    prev = prevToDF(previsions)

    nuser = length(unique(prev[:userId]))
    ntest = length(test_users)
    nprev = length(prev[1])
    println("Provided ",nprev," previsions, to ",nuser," users on ",ntest," test users.")

    writetable(previsions_file, prev)
end

###Compute 5 recommendations for the given user.
#We start by evaluating the similarity with all the items that have a popularity below the threshold and the user did not rate already. Then we keep the top 5 items.
@everywhere function provideUser(u_vect, itemVectors, pop, pop_th, ratedItems)
    #if we do not know the user's taste we cannot predict
    if length(keys(u_vect)) == 0
        return Dict{Int64,Float64}()
    end

    items = keys(itemVectors)

    userPrevisions = Tuple{Int64, Float64}[] #item, value
    for item in items
        if get(pop, item, 0) >= pop_th
            continue
        end
        if item in ratedItems
            continue
        end

        sim_val = similarity(u_vect, itemVectors[item])

        if sim_val > min_sim_to_save
            insert_prevision!(userPrevisions, item, sim_val)#sim_val * get(pop, item, 0))
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

###Compute the cosine similarity between two dictionaries
@everywhere function cosine_similarity(d1::Dict{Int64, Float64}, d2::Dict{Int64, Float64})
    num = 0.0
    den1 = 0.0
    den2 = 0.0

    for k in keys(d1)
        if k in keys(d2)
            num += d1[k]*d2[k]
        end
        den1 += d1[k]^2
    end
    for k in keys(d2)
        den2 += d2[k]^2
    end

    (num)/(sqrt(den1 * den2) + shrink)
end

#Compute the pearson correlation between two dictionaries
@everywhere function pearsonCorrelation(d1::Dict{Int64, Float64}, d2::Dict{Int64, Float64})
    num = 0.0
    den1 = 0.0
    den2 = 0.0

    for k in keys(d1)
        if k in keys(d2)
            num += d1[k]*d2[k]
            den1 += d1[k]^2
            den2 += d2[k]^2
        end
    end

    num/(shrink + sqrt(den1*den2))
end


#Insert item, value in the given array discarding the worst prevision
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


@everywhere const similarity = pearsonCorrelation #cosine_similarity, pearsonCorrelation. Pearson seems better
main()
