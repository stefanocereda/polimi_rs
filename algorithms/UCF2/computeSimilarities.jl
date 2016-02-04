###Compute the similarity between users. We define 2 similarities: the first is the usual pearson correlation between ratings, the second is a cosine similarity between taste vectors. Now we are considering the average similarity.

const numProcs = 4
addprocs(numProcs)

using DataFrames
using JLD

const models_file = "./data/UCF2/Dicts_v2.jld"
const test_file = "./data/test.csv"
const sims_file = "./data/UCF2/Sims_att_v2.jld"

@everywhere const simsToKeep = 10
@everywhere const shrink = 10
#at the end there is also similarity = pearson/cosine


function main()
    urm_dict = load(models_file, "urm_dict")
    u_b_dict = load(models_file, "userBiases")
    u_w_dict = load(models_file, "userModels")

    removeBiases!(urm_dict, u_b_dict)

    testUsers = sort(unique(readtable(test_file)[:userId]))
    trainUsers = sort(collect(keys(urm_dict)))

    sims = @sync @parallel (merge) for user1 in testUsers
        tmp = Dict{Int64, Array{Tuple{Int64, Float64},1}}() #user -> (user, sim)[]
        tmp[user1] = Tuple{Int64, Float64}[]

        for user2 in trainUsers
            if user1 == user2
                continue
            end

            #the similarity can be computed by considering ratings or models (tf-idf of features). The secons one seems better (sparse dataset)
            simil = similarity(u_w_dict[user1], u_w_dict[user2], shrink)

            insertSim!(tmp[user1], user2, simil)
        end

        tmp
    end

    save(sims_file, "similarities", sims)
end

#Remove the user biases from the urm
function removeBiases!(urm_dict, u_b_dict)
    for user in keys(urm_dict)
        map(x -> urm_dict[user][x] -= u_b_dict[user], keys(urm_dict[user]))
    end
end

#Insert the value in a dictionary keeping the best entries
@everywhere function insertSim!(vector, key, value)
    if length(vector) < simsToKeep
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

#Search the lowest element
@everywhere function searchWorst(arr)
    lowest = 10000
    pos = 0

    for i = 1:simsToKeep
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

###Compute the cosine similarity between two dictionaries
@everywhere function cosineSimilarity(d1::Dict{Int64, Float64}, d2::Dict{Int64, Float64}, shrink)
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
@everywhere function pearsonCorrelation(d1::Dict{Int64, Float64}, d2::Dict{Int64, Float64}, shrink)
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

@everywhere similarity = pearsonCorrelation

main()
