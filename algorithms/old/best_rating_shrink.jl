#comments are wrong as this was changed to provide previsions instead of recommendations

#not parallelize this, bug
#const nCore = 4
#addprocs(nCore)

@everywhere using DataFrames

train_file = "./data/train.csv"
test_file = "./data/test.csv"
prev_file = "./previsions/old/br_s2.csv"
@everywhere  ratingsToKeep = 5
const shrink = 3
const minGoodVote = 8

function main()
    urm_df = readtable(train_file)
    urm_dict = urmToDict(urm_df)
    pop = computePopularity(urm_df)
    pop_th = computePopularityThreshold(urm_dict, pop)
    meanRatings = aggregate(urm_df, :itemId, x->sum(x)/(length(x)+shrink))

    test = unique(readtable(test_file)[:userId])
    previsions = @sync @parallel (merge) for user in test
        tmp = Dict{Int64, Dict{Int64, Float64}}() #user->item->value

        tmp[user] = provideUser(urm_dict[user], pop, pop_th[user], meanRatings)

        tmp
    end

    prev = dictToDF(previsions)

    nuser = length(unique(prev[:userId]))
    ntest = length(test)
    nprev = length(prev[1])
    println("Provided ",nprev," previsions, to ",nuser," users on ",ntest," test users.")

    writetable(prev_file, prev)
end


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

function computePopularityThreshold(urm_dict, popularity)
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

    thr
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

function previsionsToDict(vect)
    ret = Dict{Int64, Float64}()
    for prev in vect
        ret[prev[1]] = prev[2]*prev[3]
    end
    ret
end

@everywhere function provideUser(urm_dict_user, pop, pop_th, totRatings)
    ratedItems = keys(urm_dict_user)

    userPrevisions = Tuple{Int64, Float64}[] #item, mean of votes, popularity

    for line in eachrow(totRatings)
        item = line[:itemId]
        item_pop = get(pop, item, 0)
        item_mean = line[:3]

        if item_pop >= pop_th
            continue
        end
        if item in ratedItems
            continue
        end

        insert_prevision!(userPrevisions, item, item_mean)
    end

    previsionsToDict(userPrevisions)
end

@everywhere function previsionsToDict(vect)
    ret = Dict{Int64, Float64}()
    for prev in vect
        ret[prev[1]] = prev[2]
    end
    ret
end

@everywhere function insert_prevision!(vector, k, v)
    if length(vector) < ratingsToKeep
        push!(vector, (k, v))
    else #otherwise search the worst
        worst = searchWorst(vector)
        #and eventually discard
        if vector[worst][2] < v
            deleteat!(vector, worst)
            push!(vector, (k, v))
        end
    end
end

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

function dictToDF(dict)
    ret = DataFrame(userId = Int64[], itemId = Int64[], rating = Float64[])
    for user in keys(dict)
        for rating in dict[user]
            push!(ret, [user, rating[1], rating[2]])
        end
    end
    ret
end

main()
