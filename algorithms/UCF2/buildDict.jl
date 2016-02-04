###BUILD NEEDED DICTIONARIES
#UCF is based on the similarity between users. In order to computer the similarity of the users we will compare their ratings (removing the user biases). We will also consider the similarity between their "taste vectors" as computed with CBF2.
#Therefore, we will compute the taste as in CBF2, translate the URM into a dictionary removing the averages and will also compute popularities and thresholds.


#WARNING: user models are computed with the same code used in CBF2, if there are improovement there they need to be copied here

const numProcs = 4
addprocs(numProcs)

@everywhere using DataFrames
using JLD

const icm_file = "./data/icm.csv"
const urm_file = "./data/train.csv"
const save_file = "./data/UCF2/Dicts_v2.jld"

@everywhere const minGoodVote = 8

function main()
    icm_df = readtable(icm_file)
    icm_dict = ICM_df_to_dict(icm_df)

    urm = readtable(urm_file)
    urm_dict = urmToDict(urm)

    userWeights = @spawn computeUserWeights(computeUserIDF(icm_dict, urm), computeUserTF(icm_dict, urm))

    userBiases = @spawn computeUserBiases(urm_dict)

    pops = @spawn computePopularity(urm, urm_dict)
    pops_c = fetch(pops)
    popularities = pops_c[1]
    pop_th = pops_c[2]

    save(save_file, "urm_dict", urm_dict, "userBiases", fetch(userBiases), "userModels", fetch(userWeights), "popularities",popularities, "pop_th", pop_th)
end

###Transform the ICM from a DF to a dictionary item -> [features]
@everywhere function ICM_df_to_dict(icm_df)
    ret = Dict{Int64, Vector{Int64}}()

    #initialize
    items = unique(icm_df[:itemId])
    for i in items
        ret[i] = Int64[]
    end

    for line in eachrow(icm_df)
        item = line[:itemId]
        feature = line[:featureId]

        vect = ret[item]
        push!(vect, feature)
    end

    ret
end

function urmToDict(urm_df)
    urm_dict = Dict{Int64, Dict{Int64, Float64}}() #user->item->rating

    for line in eachrow(urm_df)
        user = line[:userId]
        item = line[:itemId]
        rating = line[:rating]

        cur = get(urm_dict, user, Dict{Int64, Float64}())
        cur[item] = rating
        urm_dict[user] = cur
    end

    urm_dict
end

###For each feature we compute log(num of good ratings / num of good ratings on items with relevant feature).
#The given urm is assumed to contain only positive ratings
#let's try log(tot ratings / tot ratings with feature)
@everywhere function computeUserIDF(icm_dict, urm_df)
    idf_dict = Dict{Int64, Float64}()

    #initialize
    features = unique(foldl(vcat, values(icm_dict)))
    for f in features
        idf_dict[f] = 0
    end

    #start by counting ratings for each feature
    for line in eachrow(urm_df)
        item = line[:itemId]
        features = get(icm_dict, item, [])
        rating = line[:rating]

        for f in features
            idf_dict[f] += rating
        end
    end

    #and normalize
    sum_ratings = sum(urm_df[:rating])
    for f in keys(idf_dict)
        idf_dict[f] = log10(sum_ratings/(idf_dict[f]+1))
    end

    idf_dict
end

###For all the users build a dictionary containing the term frequency of each feature.
#We take the user, consider its positive ratings, increment the connected features.
#The negative ratings decrement the score.
#The result is then normalized dividing by the maximum tf of the user.
@everywhere function computeUserTF(icm_dict, urm_df)
    tf_dict = Dict{Int64, Dict{Int64, Float64}}() #user->feature->tf

    #initialize
    users = unique(urm_df[:userId])

    for user in users
        tf_dict[user] = Dict{Int64, Float64}()
    end

    for user in users
        #count the frequencies for good ratings
        goodRatings = urm_df[urm_df[:userId] .== user, :]

        for rat in eachrow(goodRatings)
            item = rat[:itemId]
            rating = rat[:rating]

            connectedfeatures = get(icm_dict, item, [])

            for feature in connectedfeatures
                try
                    tf_dict[user][feature] += rating
               catch
                    tf_dict[user][feature] = rating
                end
            end
        end

        #now consider the maximum
        try
            max_tf = maximum(values(tf_dict[user]))

            #and divide everything by it
            for feature in keys(tf_dict[user])
                tf_dict[user][feature] /= max_tf
            end
        catch
            #if tf_dict[user] is empty we cannot build user dictionary, we simply leave it empty
        end
    end

    tf_dict
end

###For all the users build a dictionary with the weight of the relative features
@everywhere function computeUserWeights(idf_dict, tf_dict)
    #We start by computing the denominator of the normalization term for each user
    #It is just sqrt(sum(tf-idf^2))
    norm_facts = Dict{Int64, Float64}()

    users = keys(tf_dict)
    for user in users
        features = keys(tf_dict[user])

        squared_sum = 0.0
        for f in features
            squared_sum += (tf_dict[user][f]*idf_dict[f])^2
        end

        norm_facts[user] = sqrt(squared_sum)
    end


    #Now we build the dictionarys: user->feature->tf-idf/norm_fact
    weights = Dict{Int64, Dict{Int64, Float64}}()

    #initialize the dicts
    for user in users
        weights[user] = Dict{Int64, Float64}()
    end

    #compute
    for user in users
        features = keys(tf_dict[user])
        for f in features
            weights[user][f] = (tf_dict[user][f]*idf_dict[f])#/norm_facts[user] ###As before, not normalizing seems to be better
        end
    end

    println("Done computing user weights")

    weights
end

###Compute a dictionary user->user_bias
@everywhere function computeUserBiases(urm_dict)
    biases = Dict{Int64, Float64}()

    for user in keys(urm_dict)
        biases[user] = mean(values(urm_dict[user]))
    end

    biases
end


###Compute the popularity of each item and the popularity threshold of each user
@everywhere function computePopularity(urm_df, urm_dict)
    popularity = Dict{Int64, Int64}()

    for line in eachrow(urm_df)
        item = line[:itemId]
        pop = get(popularity, item, 0)
        pop += 1
        popularity[item] = pop
    end


    #now thresholds
    thr = Dict{Int64, Int64}()

    for user in keys(urm_dict)
        min = 100000
        for item in keys(urm_dict[user])
            if urm_dict[user][item] < minGoodVote
                continue
            end
            if popularity[item] < min
                min = popularity[item]
            end
        end
        thr[user] = min
    end

    println("Done computing popularities")

    (popularity, thr)
end

main()
