###USER SIMILARITY EVALUATOR###
#We open the previously created user models, and for each couple of user we compute their similarity. The similarity is based on equally rated items.

const numCores = 4 #36
addprocs(numCores)

using JLD

model_file = "./data/UCF/userModels.dat"
sim_file = "./data/UCF/similaritiesSH7.dat"

@everywhere const simToKeep = 20 #how many similarities to keep for each user. This should not be considered as a parameter, it is the MAX value for k in KNN
@everywhere const shrink = 7 #shrink factor in the pearson correlation

#Given an array of couples return the position of the element with the lowest value as second elemnt of tuple. We assume the array has simToKeep values. Here we also check for NaN
@everywhere function searchWorst(arr::Array{Tuple{Int64, Float64}})
    lowest = 1000.0
    pos = 0

    for i = 1:simToKeep
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

#Compute the pearson correlation between two user models
#OK
@everywhere function pearsonCorrelation(mod1::Dict{Int64, Float64}, mod2::Dict{Int64, Float64})
    num = 0.0
    den1 = 0.0
    den2 = 0.0

    for i in keys(mod1)
        if i in keys(mod2)
            num += mod1[i]*mod2[i]
            den1 += mod1[i]^2
            den2 += mod2[i]^2
        end
    end

    num/(shrink + sqrt(den1*den2))
end



#Given a user compute the similarities with all the other users keeping only the top simToKeep
#OK
@everywhere function computeUserSimilarities(user::Int64, models::Dict{Int64, Dict{Int64, Float64}})
    similarities = Tuple{Int64, Float64}[] #(user,similarity)

    for other_user in keys(models)
        if user == other_user
            continue
        end

        value = pearsonCorrelation(models[user], models[other_user])

        #if we have space push in the array
        if length(similarities) < simToKeep
            push!(similarities, (other_user, value))
        else #otherwise search the worst
            worst = searchWorst(similarities)
            #and eventually discard
            if similarities[worst][2] < value
                deleteat!(similarities, worst)
                push!(similarities, (other_user, value))
            end
        end
    end

    sort(similarities, by=x->x[2])
end


models = load(model_file, "nor_models")
glb_avg = load(model_file, "glb_avg")
u_biases = load(model_file, "u_biases")

users = keys(models)
similarities = Dict{Int64, Array{Tuple{Int64, Float64}}}()
for user in users
    similarities[user] = computeUserSimilarities(user, models)
end
save(sim_file, "similarity", similarities)
