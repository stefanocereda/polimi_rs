const train_file = "./data/train.csv"
const test_file = "./data/test.csv"
const prev_file = "./previsions/svds/svds.csv"

const numCores = 4
addprocs(numCores - 1)

const num_sing_val = 50
const max_iterations = 200
const tolerance = 0.3
@everywhere const ratingsToKeep = 5

@everywhere using DataFrames

#Given the original urm compute and remove global avg, user offset and item bias
function removeAverages(urm)
    #GLOBAL OFFSET
    urm1 = DataFrame(userId = Int64[], itemId = Int64[], rating = Float64[])
    global_mean = mean(urm[:rating])
    for line in eachrow(urm)
        push!(urm1, [line[:userId]; line[:itemId]; line[:rating] - global_mean])
    end
    urm = urm1

    #ITEM BIAS
    urm1 = DataFrame(userId = Int64[], itemId = Int64[], rating = Float64[])
    i_biases = Dict{Int64, Float64}()
    avg = aggregate(urm, :itemId, mean)
    for line in eachrow(avg)
        i_biases[line[:itemId]] = line[:rating_mean]
    end
    for line in eachrow(urm)
        item = line[:itemId]
        push!(urm1, [line[:userId]; item; line[:rating] - i_biases[item]])
    end
    urm = urm1

    #USER OFFSET
    urm1 = DataFrame(userId = Int64[], itemId = Int64[], rating = Float64[])
    u_biases = Dict{Int64, Float64}()
    avg = aggregate(urm, :userId, mean)
    for line in eachrow(avg)
        u_biases[line[:userId]] = line[:rating_mean]
    end
    for line in eachrow(urm)
        user = line[:userId]
        push!(urm1, [user; line[:itemId]; line[:rating] - u_biases[user]])
    end
    urm = urm1

    (global_mean, i_biases, u_biases, urm)
end

#Forall the (user,item) found in the given urm the initial prediction is the sum of the averages
function buildInitPred(urm, avg)
    glb = avg[1]
    itm = avg[2]
    usr = avg[3]
    pred = Dict{Int64, Dict{Int64, Float64}}()

    for line in eachrow(urm)
        user = line[:userId]
        u_dict=
        try
            pred[user]
        catch
            pred[user] = Dict{Int64, Int64}()
            pred[user]
        end

        item = line[:itemId]
        u_dict[item] = glb + itm[item] + usr[user]
    end

    pred
end


function do_svd(urm, umax, imax)
    A = zeros(umax,imax)
    A = sparse(A)
    for line in eachrow(urm)
        user = line[:userId]
        item = line[:itemId]
        rat = line[:rating]
        A[user,item] = rat
    end

    svds(A, nsv=num_sing_val, tol=tolerance, maxiter=max_iterations)
end


#Translate the given urm into a dictionary
function urmToDict(urm)
    res = Dict{Int64, Dict{Int64, Float64}}()

    for line in eachrow(urm)
        user = line[:userId]
        u_dict =
        try
            res[user]
        catch
            res[user] = Dict{Int64, Int64}()
            res[user]
        end

        item = line[:itemId]
        u_dict[item] = line[:rating]
    end

    res
end

@everywhere function searchWorst(arr::Array{Tuple{Int64, Float64}})
    lowest = 1000.0
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

@everywhere function provideRatings(usersToDo, vectors, avg, urm)
    U = vectors[1]
    S = diagm(vectors[2])
    V = vectors[3]
    glob = avg[1]
    itm = avg[2]
    usr = avg[3]

    result = Dict{Int64, Array{Tuple{Int64, Float64}}}() #(user, item, rating)
    for user in usersToDo
        result[user] = Tuple{Int64,Float64}[]
    end

    items = keys(itm)

    for user in usersToDo
        for item in items
            if !(item in keys(urm[user])) #not already rated
                rat = glob + itm[item] + usr[user]
                rat = rat + (U[user,:]*S*V[item,:]')[1]

                #if we have space push in the array
                if length(result[user]) < ratingsToKeep
                    push!(result[user], (item, rat))
                else #otherwise search the worst
                    worst = searchWorst(result[user])
                    #and eventually discard
                    if result[user][worst][2] < rat
                        deleteat!(result[user], worst)
                        push!(result[user], (item, rat))
                    end
                end
            end
        end
    end

    previsions = DataFrame(userId = Int64[], itemId = Int64[], rating = Float64[])
    for u in keys(result)
        for entry in result[u]
            push!(previsions, [u, entry[1], entry[2]])
        end
    end

    previsions
end

function parallelize(test, vectors, avg, urm)
    test_length = length(test)
    test_length /= numCores

    workers = Array(RemoteRef{Channel{Any}}, numCores-1)

    for i in 2:numCores
        low = ceil(Int64, (i-1)*test_length+1)
        high = ceil(Int64,test_length*i)
        workers[i-1] = @spawn provideRatings(test[low:high], vectors, avg, urm)
    end

    #the first trunk for the running thread
    low = 1
    high =  ceil(Int64,test_length)
    ratings = provideRatings(test[low:high], vectors, avg, urm)

    #now collect
    for i in 1:(numCores-1)
        ratings = vcat(ratings, fetch(workers[i]))
    end

    return ratings
end




urm = readtable(train_file)
umax = maximum(urm[:userId].data)
imax = maximum(urm[:itemId].data)
avg = removeAverages(urm)
initial_pred = buildInitPred(urm, avg)
println("Start building vectors...")
vectors = do_svd(avg[4], umax, imax)

test = readtable(test_file)
test = sort(unique(test[:userId].data))
urm = urmToDict(urm)
println("Start providing ratings...")
result = parallelize(test, vectors, avg, urm)

writetable(prev_file, result)
