prefix = "python3 matching_1_to_1.py"
nums = [3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 20]
modes = ["rank_welfare", "rank_stats", "write_data"]
changes = ["any", "priority", "swapped"]

for type_of_change in changes:
    for mode in modes:
        filename = "run_rankgraph_or_data_" + type_of_change + "_" + mode + "_" + ".txt"
        with open('./' + filename,'w') as f:
            for n in nums:
                if type_of_change == "priority":
                    for k in range(1, n):
                        new_cmd = prefix + " " + str(n) + " " + type_of_change + " 10000 " + str(k) + " " + mode
                        f.write("%s\n" % new_cmd)
                else:
                    new_cmd = prefix + " " + str(n) + " " + type_of_change + " 10000 0 " + mode
                    f.write("%s\n" % new_cmd)
        f.close()



