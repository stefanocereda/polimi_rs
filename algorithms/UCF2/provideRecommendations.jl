###Provide ratings using similarities. For each user we consider its similar users, and average their ratings weightening with the similarity. User bias is then added

const numProcs = 4
addprocs(numProcs)

@everywhere const shrink = 1
@everywhere const ratingsToKeep = 5

const sim_file = "./data/UCF2/Sims_att_v2.jld"
const test_file = "./data/test.csv"
const models_file = "./data/UCF2/Dicts_v2.jld"
const previsions_file = "./previsions/UCF2/att_s1_v2.csv"

using DataFrames
using JLD

function main()
    urm_dict = load(models_file, "urm_dict")
    u_b_dict = load(models_file, "userBiases")
    sim_dict = load(sim_file, "similarities")
    pop_dict = load(models_file, "popularities")
    pth_dict = load(models_file, "pop_th")

    test_users = sort(unique(readtable(test_file)[:userId]))

    previsions = @sync @parallel (vcat) for user in test_users
        tmp = Tuple{Int64, Dict{Int64, Float64}}[] #(user, item->value)[]

        push!(tmp, (user, provideUser(user, sim_dict[user], urm_dict, u_b_dict, pop_dict, pth_dict[user])))

        tmp
    end

    prev = prevToDF(previsions)

    nuser = length(unique(prev[:userId]))
    ntest = length(test_users)
    nprev = length(prev[1])
    println("Provided ",nprev," previsions, to ",nuser," users on ",ntest," test users.")

    writetable(previsions_file, prev)
end

@everywhere function provideUser(user, similarities, urm_dict, u_b_dict, popularity, threshold)
    previsions_ndn = Dict{Int64, Tuple{Float64, Float64}}() #item -> (num/den)

    for sim_entry in similarities
        oth_user = sim_entry[1]
        sim_val = sim_entry[2]

        #consider the other user ratings
        items = keys(urm_dict[oth_user])
        for item in items
            if (popularity[item] >= threshold) || (item in keys(urm_dict[user]))
                continue
            end

            rating = urm_dict[oth_user][item] - u_b_dict[oth_user]

            updatePrev!(previsions_ndn, item, rating, sim_val)
        end
    end

    #now divide and add user bias
    previsions = Dict{Int64, Float64}()
    for item in keys(previsions_ndn)
        previsions[item] = (previsions_ndn[item][1] / (previsions_ndn[item][2] + shrink)) + u_b_dict[user]
    end

    keepBest(previsions)
end

@everywhere function updatePrev!(p_ndn, item, r, s)
    cur_prev = get(p_ndn, item, (0.0, 0.0))

    n = cur_prev[1] + r*s
    d = cur_prev[2] + s

    p_ndn[item] = (n,d)
end

@everywhere function keepBest(previsions)
    toKeep = Dict{Int64, Float64}()

    i = 0
    for prev in sort!(collect(previsions), by=x->-x[2])  #sort according to rating
        if i >= ratingsToKeep
            break
        end

        toKeep[prev[1]] = prev[2]
        i+=1
    end

    toKeep
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

main()
