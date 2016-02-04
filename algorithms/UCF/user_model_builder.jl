###USER MODEL BUILDER###
###We build a dictionary that for each user contains a dictionary mapping from items to ratings.
###Tested: ok
train_file = "./data/train.csv"
model_file = "./data/UCF/userModels.dat"

const K_variance = 5

using DataFrames
using JLD

function build_models(urm::DataFrame)
    models = Dict{Int64, Dict{Int64, Float64}}() #user->item->rating

    for line in eachrow(urm)
        user = line[:userId]
        item = line[:itemId]
        rating = line[:rating]

        model = get(models, user, Dict{Int64, Float64}())
        model[item] = rating
        models[user] = model
    end

    models
end

function computeUserBiases(models::Dict{Int64, Dict{Int64, Float64}}, glb_avg)
    result = Dict{Int64, Float64}()

    #start with a global average
    glb_sum = 0
    glb_cnt = 0
    for user in keys(models)
        for item in keys(models[user])
            glb_sum += models[user][item]
            glb_cnt += 1
        end
    end
    glb_mean = glb_sum/glb_cnt

    #and compute the averages
    for user in keys(models)
        ratings = values(models[user])
        result[user] = (glb_mean*K_variance + sum(ratings))/(K_variance + length(ratings)) - glb_avg
    end

    result
end

function normalizeModels!(models, biases, glb_avg)
    for user in keys(models)
        for item in keys(models[user])
            models[user][item] -= glb_avg + biases[user]
        end
    end
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

function computePopThresh(models, pop)
    thr = Dict{Int64, Int64}()

    for user in keys(models)
        min = 100000
        for item in keys(models[user])
            if models[user][item] < 8
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

models = build_models(urm)
pop = computePopularity(urm)
popTh = computePopThresh(models, pop)
glb_avg = mean(urm[:rating])
biases = computeUserBiases(models, glb_avg)
normalizeModels!(models, biases, glb_avg)

save(model_file, "nor_models", models, "popularities", pop, "pop_thresholds", popTh, "glb_avg", glb_avg, "u_biases", biases)
