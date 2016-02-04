using DataFrames
orig_train_file = "./data/orig/train.csv"
my_train_file = "./test/data/train.csv"
my_test_file = "./test/data/test.csv"

const minVote = 8
const minGoodVotes = 6
const testPercent = 0.2
const testThrow = 0.0

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

function atLeastSixPositive(urm)
    goodUsers = Int64[]

    users = unique(urm[:userId])
    for user in users
        if countPositiveRatings(user, urm) >= minGoodVotes
            push!(goodUsers, user)
        end
    end

    goodUsers
end

function countPositiveRatings(user, urm)
    length(urm[(urm[:userId] .== user) & (urm[:rating] .>= minVote), :][1])
end

function filterTestUsers(urm, testUsers)
    train = DataFrame(userId = Int64[], itemId = Int64[], rating = Int64[])

    for line in eachrow(urm)
        if !(line[:userId] in testUsers)
            push!(train, [line[:userId], line[:itemId], line[:rating]])
        end
    end

    train
end

function buildTest!(orig_urm, train, testUsers)
    res = DataFrame(userId = Int64[], itemId = Int64[], rating = Int64[])
    popularity = computePopularity(orig_urm)

    for user in testUsers
        positiveRatedItems = orig_urm[(orig_urm[:userId] .== user) & (orig_urm[:rating] .>= minVote), [:itemId, :rating]]

        unpopular = DataFrame(itemId = Int64[], popularity = Int64[], rating = Int64[])
        for line in eachrow(positiveRatedItems)
            push!(unpopular, [line[:itemId], popularity[line[:itemId]], line[:rating]])
        end
        sort!(unpopular, cols=[:popularity])
        unpopular=unpopular[1:5,:]

        for line in eachrow(unpopular)
            push!(res, [user, line[:itemId], line[:rating]])
        end

        for line in eachrow(orig_urm[orig_urm[:userId] .== user, :])
            if !(line[:itemId] in unpopular[:itemId])
                push!(train, [user, line[:itemId], line[:rating]])
            end
        end
    end

    res
end


##START
orig_urm = readtable(orig_train_file)

#first, the users who have less than 6 positive rating (>= 8) are removed
goodUsers = atLeastSixPositive(orig_urm)

#then, 20% of the remaining users are selected randomly as test users, and 80% of the users as train users
testUsers = Int64[]
for user in goodUsers
    if (rand() < testPercent)
        push!(testUsers, user)
    end
end

#the ratings of the train users are considered as train set
train = filterTestUsers(orig_urm, testUsers)

#for each test user, 5 most unpopular items with positive rating, are selected and their ratings are considered as test set, and the rest of the ratings are added to the train set
test = buildTest!(orig_urm, train, testUsers)

#finally, 80% of the ratings in the train set are randomly selected and removed MA PERCHÈ?
#as I don't think this has a sense, and plus I'm getting a really small test set, I won't do it

assert(length(orig_urm[1]) == length(train[1])+length(test[1]))
for line in eachrow(test)
    user = line[:userId]
    item = line[:itemId]

    assert(length(train[(train[:userId] .== user) & (train[:itemId] .== item), 1]) == 0)
end


writetable(my_train_file, train)

writetable(my_test_file, test)
