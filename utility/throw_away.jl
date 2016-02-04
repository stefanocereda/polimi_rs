using DataFrames

in_file = "./result/mergedToSend.csv"
out_file = "./result/mts_half.csv"

orig = readtable(in_file)
out = DataFrame(userId = Int64[], testItems = UTF8String[])

in_length = length(orig[1])
keep = in_length*0.75

for line in eachrow(orig[1:keep, :])
    push!(out, [line[:userId], line[:testItems]])
end
for line in eachrow(orig[keep+1:in_length, :])
    push!(out, [line[:userId], "1 2 3 4 5"])
end

writetable(out_file, out)
