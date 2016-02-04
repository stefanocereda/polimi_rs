#We open the urm_deaveraged and compute item-item similarities

const numCores = 4 #36
addprocs(numCores - 1)

@everywhere const simToKeep = 20 #how many similarities to keep for each item. This should not be considered as a parameter, it is the MAX value for k in KNN
@everywhere const shrink = 7 #shrink factor in the adjusted cosine similarity

model_file = "./data/ICF/Dicts.dat"
simil_file = "./data/ICF/simil7.dat"

using JLD


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

#Given two dictionaries, compute their adjusted cosine similarity
@everywhere function adjusted_cosine_similarity(d1::Dict{Int64, Float64}, d2::Dict{Int64, Float64})
    #dicts = user -> rating without average (user bias)
    num = den1 = den2 = 0

    for user in keys(d1)
        if user in keys(d2)
            num += d1[user]*d2[user]
            den1 += d1[user]^2
            den2 += d2[user]^2

        end
    end

    num/(shrink + sqrt(den1*den2))
end

#Given an item compute the similarities with all the other items keeping only the top simToKeep
#OK
@everywhere function computeItemSimilarities(item::Int64, urm::Dict{Int64, Dict{Int64, Float64}})
    similarities = Tuple{Int64, Float64}[] #(item,similarity)

    for other_item in keys(urm)
        if item == other_item
            continue
        end

        value = adjusted_cosine_similarity(urm[item], urm[other_item])

        #if we have space push in the array
        if length(similarities) < simToKeep
            push!(similarities, (other_item, value))
        else #otherwise search the worst
            worst = searchWorst(similarities)
            #and eventually discard
            if similarities[worst][2] < value
                deleteat!(similarities, worst)
                push!(similarities, (other_item, value))
            end
        end
    end

    similarities
end

#Compute similarities for all the given items
#OK
@everywhere function computeSimilarities(items::Array{Int64}, urm::Dict{Int64, Dict{Int64, Float64}})
    result = Dict{Int64, Array{Tuple{Int64, Float64}}}()

    for item in items
        sim = computeItemSimilarities(item, urm)
        result[item] = sort(sim, by=x->-x[2])
    end

    result
end

function parallelize(urm::Dict{Int64, Dict{Int64, Float64}})
    items = collect(keys(urm))

    num_items = length(items)
    num_items /= numCores

    workers = Array(RemoteRef{Channel{Any}}, numCores-1)

    for i in 2:numCores
        low = ceil(Int64, (i-1)*num_items+1)
        high = ceil(Int64,num_items*i)
        workers[i-1] = @spawn computeSimilarities(items[low:high], urm)
    end

    #the first trunk for the running thread
    low = 1
    high =  ceil(Int64,num_items)
    simil = computeSimilarities(items[low:high], urm)

    #now collect
    for i in 1:(numCores-1)
        simil = merge(simil, fetch(workers[i]))
    end

    return simil
end

urm = load(model_file, "tr_nor_models") #item->user->(rating-bias)
sim = parallelize(urm)
save(simil_file, "similarities", sim)
