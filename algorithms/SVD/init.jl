###http://sifter.org/~simon/journal/20061211.html
using DataFrames
using JLD

const minGoodVote = 8
const shrink_items_mean = 15
const shrink_users_mean = 7
const features = 100
const n_iterations = 1000
const initial_vector = 0.1
const K_reg = 0.02
const lrate = 0.001

const train_file = "./data/train.csv"
const test_file = "./data/test.csv"
const save_file = "./data/SVD/values_15_7_100_1000_01_002_0001.jld"

function main()
    urm_df = readtable(train_file)
    urm_orig_dict = urmToDict(urm_df)

    #compute the popularities
    println("Computing popularities...")
    pop_dict = computePopularity(urm_df)
    pop_th_dict = computePopularityThresholds(urm_orig_dict, pop_dict)

    #compute and remove the averages
    println("Computing averages...")
    global_avg = removeGlobalAverage!(urm_df)
    items_avg = removeItemAverage!(urm_df)
    users_avg = removeUserAverage!(urm_df)

    #now we use the averages to compute the initial predicted ratings
    println("Initializing predictions...")
    predicted_ratings_dict = initializeRatings(urm_df, global_avg, items_avg, users_avg)

    #and start the learning!
    #check here
    println("Learning...")
    vectors = learnVectors(predicted_ratings_dict, urm_orig_dict, unique(urm_df[:itemId]))

    #save the vectors
    println("Saving...")
    save(save_file, "urm", urm_orig_dict, "pop", pop_dict, "pop_th", pop_th_dict, "g_avg", global_avg, "i_avg", items_avg, "u_avg", users_avg, "u_vect", vectors[1], "i_vect", vectors[2])
end

#From DF to Dict
function urmToDict(urm_df)
    urm_dict = Dict{Int64, Dict{Int64, Float64}}()

    for urm_line in eachrow(urm_df)
        user = urm_line[:userId]
        item = urm_line[:itemId]
        rating = urm_line[:rating]

        try
            urm_dict[user]
        catch
            urm_dict[user] = Dict{Int64, Float64}()
        end

        urm_dict[user][item] = rating
    end

    return urm_dict
end

#Compute the popularities for each item
function computePopularity(urm_df)
    popularity = Dict{Int64, Int64}()

    for line in eachrow(urm_df)
        item = line[:itemId]
        pop = get(popularity, item, 0)
        pop += 1
        popularity[item] = pop
    end

    popularity
end

#compute the thresholds for each user
function computePopularityThresholds(urm_dict, pop_dict)
    thr = Dict{Int64, Int64}()

    for user in keys(urm_dict)
        min = 100000
        for item in keys(urm_dict[user])
            if urm_dict[user][item] < minGoodVote
                continue
            end
            if pop_dict[item] < min
                min = pop_dict[item]
            end
        end
        thr[user] = min
    end

    return thr
end

#Compute and subtract the global average
function removeGlobalAverage!(urm_df)
    avg = mean(urm_df[:rating])
    urm_df[:rating] -= avg
    return avg
end

#Given the urm without the global average compute the item biases and remove them. The item biases are assumed to have an average bias of 0.
function removeItemAverage!(urm_df)
    items = unique(urm_df[:itemId])
    items_avg = Dict{Int64, Float64}()

    for item in items
        ratings = urm_df[urm_df[:itemId] .== item, :rating]
        items_avg[item] = sum(ratings) / (length(ratings) + shrink_items_mean)
    end

    for urm_line in eachrow(urm_df)
        urm_line[:rating] -= items_avg[urm_line[:itemId]]
    end

    return items_avg
end

#Given the urm compute the user biases and remove them. The user biases are assumed to have an average bias of 0 (urm already without global and item biases).
function removeUserAverage!(urm_df)
    users = unique(urm_df[:userId])
    users_avg = Dict{Int64, Float64}()

    for user in users
        ratings = urm_df[urm_df[:userId] .== user, :rating]
        users_avg[user] = sum(ratings) / (length(ratings) + shrink_users_mean)
    end

    for urm_line in eachrow(urm_df)
        urm_line[:rating] -= users_avg[urm_line[:userId]]
    end

    return users_avg
end

function clip(value)
    return min(max(value, 0),10)
end

#For every urm entry compute global+item+user biases and insert into a dictionary user->item->rating which is returned
function initializeRatings(urm_df, glb, itm_avg, usr_avg)
    urm_pred_dict = Dict{Int64, Dict{Int64, Float64}}()

    for urm_line in eachrow(urm_df)
        user = urm_line[:userId]
        item = urm_line[:itemId]

        try
            urm_pred_dict[user]
        catch
            urm_pred_dict[user] = Dict{Int64, Float64}()
        end

        u_dict = urm_pred_dict[user]
        u_dict[item] = clip(glb + itm_avg[item] + usr_avg[user])
    end

    return urm_pred_dict
end

#Actual training, for each feature scan several times the whole urm and adjust weights
function learnVectors(predicted_ratings_dict, urm_dict, items)
    user_vects = Dict{Int64, Vector{Float64}}()
    item_vects = Dict{Int64, Vector{Float64}}()

    #initialize the vectors
    for user in keys(urm_dict)
        user_vects[user] = zeros(Float64, features)
        for feature in 1:features
            user_vects[user][feature] = initial_vector
        end
    end
    for item in items
        item_vects[item] = zeros(Float64, features)
        for feature in 1:features
            item_vects[item][feature] = initial_vector
        end
    end

    #do the training, repeat every feature a lot of times
    for feature in 1:features
        println("Computing feature ", feature)
        for iteration in 1:n_iterations

            #scan the whole urm
            for user in keys(predicted_ratings_dict)
                u_val = user_vects[user][feature]
                for item in keys(predicted_ratings_dict[user])
                    i_val = item_vects[item][feature]

                    #compute the new vectors
                    err = urm_dict[user][item] - predicted_ratings_dict[user][item]
                    user_vects[user][feature] += lrate * (err * i_val - K_reg * u_val);
                    item_vects[item][feature] += lrate * (err * u_val - K_reg * i_val);

                    #update the predicted rating clipping in 0-10 range
                    predicted_ratings_dict[user][item] = clip( predicted_ratings_dict[user][item] + user_vects[user][feature] * item_vects[item][feature])
                end
            end
        end
    end

    return (user_vects, item_vects)
end

main()
