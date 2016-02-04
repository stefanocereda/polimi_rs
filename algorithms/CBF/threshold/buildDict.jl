###We take the standard inputs and transform them as dictionary. URM becomes user->item->rating and ICM becomes item->attribute->weight

train_file = "./data/train.csv"
icm_file = "./data/icm.csv"
model_file = "./data/CBF/Dicts.dat"

using DataFrames
using JLD

#User -> item -> rating
function build_urm(urm::DataFrame)
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

#Feature -> idf
function build_idf(icm::DataFrame)
    #start by aggregating the dataframe computing idf
    num_item = length(groupby(icm, :itemId))

    function idf_fun(x)
        return log10(num_item/length(x))
    end

    idf_df = aggregate(icm, :featureId, idf_fun)

    #now it becomes a dictionary
    res = Dict{Int64, Float64}()
    for line in eachrow(idf_df)
        feature = line[:featureId]
        idf = line[:itemId_idf_fun]
        res[feature] = idf
    end

    res
end


#item -> feature -> weight
function build_icm(icm::DataFrame, idf::Dict{Int, Float64})
    #compute the normalization factors
    norms = Dict{Int64, Float64}() #item -> norm fact

    items = unique(icm[:itemId].data)
    for item in items
        features = icm[icm[:itemId] .== item, :featureId].data

        norm = 0.0
        for feat in features
            norm += idf[feat]^2
        end

        norm = sqrt(norm)

        norms[item] = norm
    end

    #and now build the dict
    res = Dict{Int64, Dict{Int64, Float64}}() #item->feature->weight
    for line in eachrow(icm)
        item = line[:itemId]
        feat = line[:featureId]

        val = idf[feat] / norms[item]

        old = get(res, item, Dict{Int64, Float64}())
        old[feat] = val
        res[item] = old
    end

    res
end

urm = readtable(train_file)
icm = readtable(icm_file)

ratings = build_urm(urm)#OK
idf = build_idf(icm)#OK
attributes = build_icm(icm, idf)#ok if the book is ok...

save(model_file, "urm", ratings, "icm", attributes)
