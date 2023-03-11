import pandas as pd
import numpy as np
import os
import sys
import csv
import matplotlib.pyplot as plt
import matplotlib.font_manager

plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Tahoma'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 12

def get_average_stdev(df, column_name):
    return round(df[column_name].mean(), 2), round(df[column_name].std(), 2)

def get_min(mins, column_name):
    return mins[column_name]

def get_max(maxes, column_name):
    return maxes[column_name]

def create_histogram_plot_freq(data, n, k,  x_axis, title, type_of_change):
    fig, ax = plt.subplots(figsize = (6,4))
    # Later in the code
    plt.locator_params(axis="both", integer=True, tight=True)
    data.plot(kind = "hist", bins = range(n + 1), color="maroon", linewidth = 0.5, edgecolor = "black")
    ax.set_xlabel(x_axis)
    ax.set_xticks(np.arange(n + 1))
    ax.set_ylabel("Frequency")
    plt.title(title + " for n = " + str(n) + " and k = " + str(k))
    plt.savefig("./plots/" + type_of_change + "/" + str(n) + "_" + str(k) + "_" + type_of_change + "_" + title + "_freq.png", bbox_inches='tight')
    plt.close()

def create_histogram_plot_pct(data, n, k, x_axis, title, type_of_change):
    fig, ax = plt.subplots(figsize = (6,4))
    # Later in the code
    ax.grid(False)
    plt.locator_params(axis="both", integer=True, tight=True)
    data.plot(kind = "hist", bins = range(n + 1), grid= False, color = "maroon", linewidth = 0.5, edgecolor = "black", weights=np.ones(len(data)) / len(data))
    ax.set_xlabel(x_axis)
    ax.set_xticks(np.arange(n + 1))
    ax.set_ylabel("Frequency")
    plt.title(title + " for n = " + str(n) + " and k = " + str(k))
    plt.savefig("./plots/" + type_of_change + "/" + str(n) + "_" + str(k) + "_" + type_of_change + "_" + title + '_pct.png', bbox_inches='tight')
    plt.close()

def create_general_histogram(data, n, k, x_axis, title, type_of_change, num_bins):
    fig, ax = plt.subplots(figsize = (6,4))
    # Later in the code
    plt.locator_params(axis="both", integer=True, tight=True)
    data.plot(kind = "hist", bins=num_bins, color = "maroon", linewidth = 0.5, edgecolor = "black")
    ax.set_xlabel(x_axis)
    ax.set_ylabel("Frequency")
    plt.title(title + " for n = " + str(n) + " and k = " + str(k))
    plt.savefig("./plots/" + type_of_change + "/" + str(n) + "_" + str(k) + "_" + type_of_change + "_" +  title + '_freq.png', bbox_inches='tight')
    plt.close()

def create_pct_histogram(data, n, k, x_axis, title, type_of_change):
    fig, ax = plt.subplots(figsize = (6,4))
    # Later in the code
    ax.grid(False)
    plt.locator_params(axis="both", integer=True, tight=True)
    data.plot(kind = "hist", grid= False, color = "maroon", edgecolor = "black", weights=np.ones(len(data)) / len(data))
    ax.set_xlabel(x_axis)
    ax.set_ylabel("Frequency")
    plt.title(title)
    plt.savefig("./plots/" + type_of_change + "/" + str(n) + "_" + str(k) + "_" + type_of_change + "_" + title + '_pct.png', bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    files = os.listdir("./logs/")
    files.sort()

    type_of_change = "priority"

    new_file = 'all_matching_data_' + type_of_change + ".csv"
    worsts_bests_data_file = 'priority_extra_data.csv'

    out_columns = ["n", "k", "Type of Change", "Changed", "Unchanged", "Num No Worse", "Num No Better",
                   "Worst Util Welfare", "Best Util Welfare", "Worst Util Count", "Best Util Count", "Worst Egal Welfare",
                   "Avg/Stdev Util Welfare", "Avg/Stdev Util Count", "Avg/Stdev Egal Welfare",
                   "Avg/Stdev Collateral Damage", "Avg/Stdev Consistent",
                   "Avg/Stdev Num Better", "Avg/Stdev Num Unchanged",
                   "Worst Util Welfare Envy",
                   "Worst Util Count Envy", 
                   "Worst Minority Num Worse", "Worst Majority Num Worse",
                   "Avg/Stdev Minority Util Welfare", "Avg/Stdev Majority Util Welfare",
                   "Avg/Stdev Util Welfare Envy", "Avg/Stdev Egal Welfare Envy",
                   "Avg/Stdev Majority Num Worse", "Avg/Stdev Minority Num Worse"]

    labels = []
    num_minority = []
    averages = {"util": [], "count": [], "egal": [], "collateral": [], "consistent": [],
                "better": [], "unchanged": []}
    stdevs = {"util": [], "count": [], "egal": [], "collateral": [], "consistent": [],
                "better": [], "unchanged": []}
    titles = {"util": "Utilitarian Welfare", 
              "count": "Num Worse Off",
              "egal": "Egalitarian Welfare",
              "collateral": "Collateral Damage",
              "consistent": "Num Consistent",
              "better": "Num Better Off",
              "unchanged": "Num Unchanged",
              "min_util": "Minority Utilitarian Welfare",
              "maj_util": "Majority Utilitarian Welfare",
              "worst_min_util": "Worst Minority Utilitarian Welfare",
              "worst_maj_util": "Worst Majority Utilitarian Welfare",
              "best_min_util": "Best Minority Utilitarian Welfare",
              "best_maj_util": "Best Majority Utilitarian Welfare",
              "min_egal": "Minority Egalitarian Welfare",
              "maj_egal": "Majority Egalitarian Welfare",
              "util_envy": "Utilitarian Welfare Envy",
              "egal_envy": "Egalitarian Welfare Envy",
              "min_count": "Minority Num Worse Off",
              "maj_count": "Majority Num Worse Off",
              "worst_util": "Worst Utilitarian Welfare", 
              "best_util": "Best Utilitarian Welfare", 
              }
    
    priority_avgs = ["min_util", "maj_util", "util_envy", "min_egal", "maj_egal", "min_count", "maj_count"]
    
    counts = {"n": [],
              "k": [],
              "num_changed": [],
              "num_unchanged": [],
              "worst_util": [],
              "best_util": [],
              "worst_min_util": [],
              "best_min_util": [],
              "worst_maj_util": [],
              "best_maj_util": [],
              "util_envy": [],
              "util_envy_for_min": [],
              "count_envy": []
              }

    for name in priority_avgs:
        averages[name] = []
        stdevs[name] = []

    with open(r'./' + new_file, 'w', newline='') as fp:
        csvwriter = csv.writer(fp, delimiter=',')
        csvwriter.writerow(out_columns)

        for filename in files:
            if "log" in filename and type_of_change in filename:
                f = os.path.join("./logs/", filename)
                
                params = filename.split('_')
                n = int(params[1])
                labels.append(n)
                type_of_change = params[2]
                num_its = int(params[3])
                k = int(params[4].split('.')[0])
                num_minority.append(k)
                
                df = pd.read_csv(f)
                df["consistent"] = df["rank_better_match_better"] + df["rank_worse_match_worse"] + df["rank_unchanged_match_unchanged"]
                df["util_envy"] = df["maj_util"] - df["min_util"]
                df["egal_envy"] = df["maj_egal"] - df["min_egal"]
                df["count_envy"] = df["maj_better"] - (-1 * df["rank_better_match_worse"])

                mins = df.min()
                maxes = df.max()

                changed_only = df[df["is_changed"] == True]
                print(len(changed_only))
                num_unchanged = df["is_changed"].value_counts()[False]
                counts["num_unchanged"].append(num_unchanged)
                num_changed = df["is_changed"].value_counts()[True]
                counts["num_changed"].append(num_changed)
                
                num_no_worse_off = len(changed_only[changed_only["num_worse"] == 0].index)
                num_no_better_off = len(changed_only[changed_only["num_better"] == 0].index)
                num_better_and_worse = num_changed - num_no_worse_off - num_no_better_off

                avg_util_welfare, std_util_welfare = get_average_stdev(df, "util")
                avg_util_welfare_changed, std_util_welfare_changed = get_average_stdev(changed_only, "util")
                worst_util_welfare = get_min(mins, "util")
                best_util_welfare = get_max(maxes, "util")
                counts["worst_util"].append(worst_util_welfare)
                counts["best_util"].append(best_util_welfare)

                avg_maj_util_welfare, std_maj_util_welfare = get_average_stdev(df, "maj_util")
                avg_maj_util_welfare_changed, std_maj_util_welfare_changed = get_average_stdev(changed_only, "maj_util")
                worst_maj_util_welfare = get_min(mins, "maj_util")
                best_maj_util_welfare = get_max(maxes, "maj_util")
                counts["best_maj_util"].append(best_maj_util_welfare)
                counts["worst_maj_util"].append(worst_maj_util_welfare)

                avg_min_util_welfare, std_min_util_welfare = get_average_stdev(df, "min_util")
                avg_min_util_welfare_changed, std_min_util_welfare_changed = get_average_stdev(changed_only, "min_util")
                worst_min_util_welfare = get_min(mins, "min_util")
                best_min_util_welfare = get_max(maxes, "min_util")
                counts["worst_min_util"].append(worst_min_util_welfare)
                counts["best_min_util"].append(best_min_util_welfare)

                avg_util_envy_welfare, std_util_envy_welfare = get_average_stdev(df, "util_envy")
                avg_util_envy_welfare_changed, std_util_envy_welfare_changed = get_average_stdev(changed_only, "util_envy")
                worst_util_envy_welfare = get_min(mins, "util_envy")
                worst_util_envy_welfare_for_mins = get_max(maxes, "util_envy")
                counts["util_envy"].append(worst_util_envy_welfare)
                counts["util_envy_for_min"].append(-1 * worst_util_envy_welfare_for_mins)

                avg_egal_welfare, std_egal_welfare = get_average_stdev(df, "max_worse")
                avg_egal_welfare_changed, std_egal_welfare_changed = get_average_stdev(changed_only, "max_worse")
                worst_egal_welfare = get_min(mins, "max_worse")

                avg_egal_envy_welfare, std_egal_envy_welfare = get_average_stdev(df, "egal_envy")
                avg_egal_envy_welfare_changed, std_egal_envy_welfare_changed = get_average_stdev(changed_only, "egal_envy")
                worst_egal_envy_welfare = get_min(mins, "egal_envy")

                avg_min_egal_welfare, std_min_egal_welfare = get_average_stdev(df, "min_egal")
                avg_min_egal_welfare_changed, std_min_egal_welfare_changed = get_average_stdev(changed_only, "min_egal")
                worst_min_egal_welfare = get_min(mins, "min_egal")

                avg_maj_egal_welfare, std_min_egal_welfare = get_average_stdev(df, "maj_egal")
                avg_maj_egal_welfare_changed, std_maj_egal_welfare_changed = get_average_stdev(changed_only, "maj_egal")
                worst_maj_egal_welfare = get_min(mins, "maj_egal")

                avg_num_worse, std_num_worse = get_average_stdev(df, "num_worse")
                avg_num_worse_changed, std_num_worse_changed = get_average_stdev(changed_only, "num_worse")
                worst_num_worse = get_max(maxes, "num_worse")
                best_num_better = get_max(maxes, "num_better")

                avg_min_num_worse, std_min_num_worse = get_average_stdev(df, "rank_better_match_worse")
                avg_min_num_worse_changed, std_min_num_worse_changed = get_average_stdev(changed_only, "rank_better_match_worse")

                avg_maj_num_worse, std_maj_num_worse = get_average_stdev(df, "maj_worse")
                avg_maj_num_worse_changed, std_maj_num_worse_changed = get_average_stdev(changed_only, "maj_worse")

                worst_count_envy = get_max(maxes, "count_envy")
                counts["count_envy"].append(worst_count_envy)
                worst_min_num_worse = get_max(maxes, "rank_better_match_worse")
                worst_maj_num_worse = get_max(maxes, "maj_worse")

                avg_num_unchanged, std_num_unchanged = get_average_stdev(df, "num_unchanged")
                avg_num_unchanged_changed, std_num_unchanged_changed = get_average_stdev(changed_only, "num_unchanged")

                avg_num_better, std_num_better = get_average_stdev(df, "num_better")
                avg_num_better_changed, std_num_better_changed = get_average_stdev(changed_only, "num_better")

                avg_consistent, std_consistent = get_average_stdev(df, "consistent")
                avg_consistent_changed, std_consistent_changed = get_average_stdev(changed_only, "consistent")

                avg_collateral_damage, std_collateral_damage = get_average_stdev(df, "rank_unchanged_match_worse")
                avg_collateral_damage_changed, std_collateral_damage_changed = get_average_stdev(changed_only, "rank_unchanged_match_worse")
                worst_collateral_damage = (maxes, "rank_unchanged_match_worse")

                create_histogram_plot_freq(changed_only["num_worse"], n, k, "# students", "Worsened Students", type_of_change)
                create_histogram_plot_pct(changed_only["num_worse"], n, k, "# students", "% Worsened Students", type_of_change)

                create_histogram_plot_freq(changed_only["num_better"], n, k, "# students", "Improved Students", type_of_change)
                create_histogram_plot_pct(changed_only["num_better"], n, k, "# students", "% Improved Students", type_of_change)

                num_bins = len(pd.unique(changed_only["util"]))
                create_general_histogram(changed_only["util"], n, k, "total rank change", "Utilitarian Welfare", type_of_change, num_bins)

                to_write_row = [n, k, type_of_change, num_changed, num_unchanged, num_no_worse_off, num_no_better_off,
                                    worst_util_welfare, best_util_welfare, worst_num_worse, best_num_better, worst_egal_welfare,
                                    str(avg_util_welfare_changed) + " (" + str(std_util_welfare_changed) + ")",
                                    str(avg_num_worse_changed) + " (" + str(std_num_worse_changed) + ")",
                                    str(avg_egal_welfare_changed) + " (" + str(std_egal_welfare_changed) + ")",
                                    str(avg_collateral_damage_changed) + " (" + str(std_collateral_damage_changed) + ")",
                                    str(avg_consistent_changed) + " (" + str(std_consistent_changed) + ")",
                                    str(avg_num_better_changed) + " (" + str(std_num_better_changed) + ")",
                                    str(avg_num_unchanged_changed) + " (" + str(std_num_unchanged_changed) + ")",
                                    worst_util_envy_welfare, worst_count_envy, worst_min_num_worse, worst_maj_num_worse,
                                    str(avg_min_util_welfare_changed) + " (" + str(std_min_util_welfare_changed) + ")",
                                    str(avg_maj_util_welfare_changed) + " (" + str(std_maj_util_welfare_changed) + ")",
                                    str(avg_util_envy_welfare_changed) + " (" + str(std_util_envy_welfare_changed) + ")",
                                    str(avg_egal_envy_welfare_changed) + " (" + str(std_egal_envy_welfare_changed) + ")",
                                    str(avg_maj_num_worse_changed) + " (" + str(std_maj_num_worse_changed) + ")",
                                    str(avg_min_num_worse_changed) + " (" + str(std_min_num_worse_changed) + ")",
                                    ]
                assert(len(to_write_row) == len(out_columns))

                csvwriter.writerow(to_write_row)
        fp.close()

    counts["n"] = labels
    counts["k"] = num_minority
    all_df = pd.DataFrame(counts)
    all_df = all_df.sort_values("n")

    all_df.to_csv(worsts_bests_data_file, index=False)

        