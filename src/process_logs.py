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

def get_average_stdev_pct(df, column_name, n):
    return round(df[column_name].mean() / n, 2), round(df[column_name].std() / (n ** 2), 2)

def get_min(mins, column_name):
    return mins[column_name]

def get_max(maxes, column_name):
    return maxes[column_name]

def create_histogram_plot_freq(data, n, x_axis, title, type_of_change, num_bins):
    fig, ax = plt.subplots(figsize = (6,4))
    # Later in the code
    plt.locator_params(axis="both", integer=True, tight=True)
    if num_bins == 0:
        data.plot(kind = "hist", bins = range(n + 1), color="maroon", linewidth = 0.5, edgecolor = "black")
    else:
        data.plot(kind = "hist", bins = num_bins, color="maroon", linewidth = 0.5, edgecolor = "black")
    ax.set_xlabel(x_axis)
    ax.set_xticks(np.arange(n + 1))
    ax.set_ylabel("Frequency")
    plt.title(title + " for n = " + str(n))
    plt.savefig("./plots/" + type_of_change + "/" + str(n) + "_" + type_of_change + "_" + title + "_freq.png", bbox_inches='tight', dpi=100)
    plt.close()

def create_histogram_plot_pct(data, n, x_axis, title, type_of_change):
    fig, ax = plt.subplots(figsize = (6,4))
    # Later in the code
    plt.locator_params(axis="both", integer=True, tight=True)
    data.plot(kind = "hist", bins = range(n + 1),  color = "maroon", linewidth = 0.5, edgecolor = "black", weights=np.ones(len(data)) / len(data))
    ax.set_xlabel(x_axis)
    ax.set_xticks(np.arange(n + 1))
    ax.set_ylabel("Frequency")
    plt.title(title + " for n = " + str(n))
    plt.savefig("./plots/" + type_of_change + "/" + str(n) + "_" + type_of_change + "_" + title + '_pct.png', bbox_inches='tight', dpi=100)
    plt.close()

def create_general_histogram(data, x_axis, title, type_of_change, num_bins):
    fig, ax = plt.subplots(figsize = (6,4))
    # Later in the code
    plt.locator_params(axis="both", integer=True, tight=True)
    data.plot(kind = "hist", bins=num_bins, color = "maroon", linewidth = 0.5, edgecolor = "black")
    ax.set_xlabel(x_axis)
    ax.set_ylabel("Frequency")
    plt.title(title + " for n = " + str(n))
    plt.savefig("./plots/" + type_of_change + "/" + str(n) + "_" + type_of_change + "_" +  title + '_freq.png', bbox_inches='tight', dpi=100)
    plt.close()

def create_pct_histogram(data, x_axis, title, type_of_change, num_bins):
    fig, ax = plt.subplots(figsize = (6,4))
    # Later in the code
    plt.locator_params(axis="both", integer=True, tight=True)
    data.plot(kind = "hist", bins = num_bins, color = "maroon", linewidth = 0.5, edgecolor = "black", weights=np.ones(len(data)) / len(data))
    ax.set_xlabel(x_axis)
    ax.set_ylabel("Frequency")
    plt.title(title + " for n = " + str(n))
    plt.savefig("./plots/" + type_of_change + "/" + str(n) + "_" + type_of_change + "_" + title + '_pct.png', bbox_inches='tight', dpi=100)
    plt.close()

def plot_pct_bar_chart(df, column_list, y_label, title, type_of_change, x_labels):
    fig, ax = plt.subplots(figsize = (6,4))
    x_pos = np.arange(len(x_labels))
    if len(column_list) == 1:
        df.plot(x = "n", y = column_list, kind = "bar", grid= False, linewidth = 0.5, edgecolor = "black", weights=np.ones(len(df)) / len(df))
    else:
        ax = df.plot(x = "n", y = column_list[0], color = "steelblue", kind = "bar", weights=np.ones(len(df)) / len(df))
        df.plot(x = "n", y = column_list[1], ax = ax, color = "maroon", weights=np.ones(len(df)) / len(df))
    ax.set_xlabel("# schools")
    ax.set_ylabel(y_label)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    plt.title(title + " for n = " + str(n))
    plt.savefig("./plots/" + type_of_change + "/" + str(n) + "_" + type_of_change + "_" +  title + '_pct.png', bbox_inches='tight', dpi=100)
    plt.close()

def plot_normal_bar_chart(df, column_list, y_label, title, type_of_change, x_labels):
    fig, ax = plt.subplots(figsize = (6,4))
    x_pos = np.arange(len(x_labels))
    data = df[column_list]
    if len(column_list) == 1:
        df.plot(x = "n", y = column_list, kind = "bar", color = 'maroon', linewidth = 0.5, edgecolor = "black")
    else:
        data.plot.bar(x = "n", stacked = True, color = ["steelblue", "maroon"])
        plt.legend(bbox_to_anchor=(1.0, 1.0))
    ax.set_xlabel("# schools")
    ax.set_ylabel(y_label)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    plt.title(title)
    plt.savefig("./plots/" + type_of_change + "/" + type_of_change + "_" + title + '.png', bbox_inches='tight', dpi=100)
    plt.close()

def plot_with_error_bars(averages, stdevs, x_labels, y_label, title, type_of_change):
    x_labels, stdevs, averages = map(list, zip(*sorted(zip(x_labels, stdevs, averages))))
    fig, ax = plt.subplots(figsize = (6,4))
    x_pos = np.arange(len(x_labels))
    (_, caps, _) = ax.errorbar(x_pos, averages, yerr=stdevs, linewidth = 0.5, color = "maroon", ecolor = "maroon", ls='-', fmt='o', markersize=6, capsize = 4)
    for cap in caps:
        cap.set_markeredgewidth(1)
    ax.set_xlabel("n")
    ax.set_ylabel(y_label)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    plt.title(title)
    plt.savefig("./plots/" + type_of_change + "/" + type_of_change + "_" + title + '_freq.png', bbox_inches='tight', dpi=100)
    plt.close()

if __name__ == "__main__":
    files = os.listdir("./logs/")
    files.sort()

    type_of_change = sys.argv[1]

    new_file = 'all_matching_data_' + type_of_change + ".csv"

    out_columns = ["n", "type", "Num Changed", "Num Unchanged", "Num No Worse", "Num No Better",
                   "Worst Util Welfare", "Best Util Welfare", "Worst Util Count", "Best Util Count", "Worst Egal Welfare",
                   "Avg/Stdev Util Welfare", "Avg/Stdev Pct Util Count", "Avg/Stdev Egal Welfare",
                   "Avg/Stdev Collateral Damage", "Avg/Stdev Pct Consistent",
                   "Avg/Stdev Pct Num Better", "Avg/Stdev Pct Num Unchanged"]

    labels = []
    num_minority = []
    averages = {"util": [], "count": [], "egal": [], "collateral": [], "consistent": [],
                "better": [], "unchanged": []}
    stdevs = {"util": [], "count": [], "egal": [], "collateral": [], "consistent": [],
                "better": [], "unchanged": []}
    titles = {"util": "Utilitarian Welfare", 
              "count": "Pct Worse Off",
              "egal": "Egalitarian Welfare",
              "collateral": "Collateral Damage",
              "consistent": "Pct Consistent",
              "better": "Pct Better Off",
              "unchanged": "Pct Unchanged",
              "min_util": "Minority Utilitarian Welfare",
              "maj_util": "Majority Utilitarian Welfare",
              "util_envy": "Utilitarian Welfare Envy",
              "egal_envy": "Egalitarian Welfare Envy",
              "min_count": "Minority Num Worse Off",
              "maj_count": "Majority Num Worse Off",
              "worst_util": "Worst Utilitarian Welfare",
              "best_util": "Best Utilitarian Welfare"
              }
    
    counts = {"n": [],
              "num_changed": [],
              "num_unchanged": [],
              "worst_util": [],
              "best_util": []
              }

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
                
                df = pd.read_csv(f)
                df["consistent"] = df["rank_better_match_better"] + df["rank_worse_match_worse"] + df["rank_unchanged_match_unchanged"]
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
                averages["util"].append(avg_util_welfare_changed)
                stdevs["util"].append(std_util_welfare_changed)
                counts["worst_util"].append(worst_util_welfare)
                counts["best_util"].append(best_util_welfare)

                avg_egal_welfare, std_egal_welfare = get_average_stdev(df, "max_worse")
                avg_egal_welfare_changed, std_egal_welfare_changed = get_average_stdev(changed_only, "max_worse")
                worst_egal_welfare = get_min(mins, "max_worse")
                averages["egal"].append(avg_egal_welfare_changed)
                stdevs["egal"].append(std_egal_welfare_changed)

                avg_num_worse, std_num_worse = get_average_stdev(df, "num_worse")
                avg_num_worse_changed, std_num_worse_changed = get_average_stdev_pct(changed_only, "num_worse", n)
                worst_num_worse = get_max(maxes, "num_worse")
                best_num_better = get_max(maxes, "num_better")
                averages["count"].append(avg_num_worse_changed)
                stdevs["count"].append(std_num_worse_changed)

                avg_num_unchanged, std_num_unchanged = get_average_stdev(df, "num_unchanged")
                avg_num_unchanged_changed, std_num_unchanged_changed = get_average_stdev_pct(changed_only, "num_unchanged", n)
                averages["unchanged"].append(avg_num_unchanged_changed)
                stdevs["unchanged"].append(std_num_unchanged_changed)

                avg_num_better, std_num_better = get_average_stdev(df, "num_better")
                avg_num_better_changed, std_num_better_changed = get_average_stdev_pct(changed_only, "num_better", n)
                averages["better"].append(avg_num_better_changed)
                stdevs["better"].append(std_num_better_changed)

                avg_consistent, std_consistent = get_average_stdev(df, "consistent")
                avg_consistent_changed, std_consistent_changed = get_average_stdev_pct(changed_only, "consistent", n)
                averages["consistent"].append(avg_consistent_changed)
                stdevs["consistent"].append(std_consistent_changed)

                avg_collateral_damage, std_collateral_damage = get_average_stdev(df, "rank_unchanged_match_worse")
                avg_collateral_damage_changed, std_collateral_damage_changed = get_average_stdev(changed_only, "rank_unchanged_match_worse")
                worst_collateral_damage = (maxes, "rank_unchanged_match_worse")
                averages["collateral"].append(avg_collateral_damage_changed)
                stdevs["collateral"].append(std_collateral_damage_changed)

                create_histogram_plot_freq(changed_only["num_worse"], n, "# students", "Worsened Students", type_of_change, 0)
                create_histogram_plot_pct(changed_only["num_worse"], n, "# students", "% Worsened Students", type_of_change)

                create_histogram_plot_freq(changed_only["num_better"], n, "# students", "Improved Students", type_of_change, 0)
                create_histogram_plot_pct(changed_only["num_better"], n, "# students", "% Improved Students", type_of_change)

                num_bins = len(pd.unique(changed_only["util"]))
                create_general_histogram(changed_only["util"], "total rank change", "Utilitarian Welfare", type_of_change, num_bins)

                to_write_row = [n, type_of_change, num_changed, num_unchanged, num_no_worse_off, num_no_better_off,
                                    worst_util_welfare, best_util_welfare, worst_num_worse, best_num_better, worst_egal_welfare,
                                    str(avg_util_welfare_changed) + " (" + str(std_util_welfare_changed) + ")",
                                    str(avg_num_worse_changed) + " (" + str(std_num_worse_changed) + ")",
                                    str(avg_egal_welfare_changed) + " (" + str(std_egal_welfare_changed) + ")",
                                    str(avg_collateral_damage_changed) + " (" + str(std_collateral_damage_changed) + ")",
                                    str(avg_consistent_changed) + " (" + str(std_consistent_changed) + ")",
                                    str(avg_num_better_changed) + " (" + str(std_num_better_changed) + ")",
                                    str(avg_num_unchanged_changed) + " (" + str(std_num_unchanged_changed) + ")",
                                    ]
                assert(len(to_write_row) == len(out_columns))
                csvwriter.writerow(to_write_row)

        counts["n"] = labels
        all_df = pd.DataFrame(counts)
        all_df = all_df.sort_values("n")

        plot_normal_bar_chart(all_df, ["n", "num_unchanged", "num_changed"], "Frequency", "Num Changed and Unchanged", type_of_change, labels)
        
        plot_normal_bar_chart(all_df, ["worst_util"], "Frequency", titles["worst_util"], type_of_change, labels)
        plot_normal_bar_chart(all_df, ["best_util"], "Frequency", titles["best_util"], type_of_change, labels)

        for key in averages.keys():
            plot_with_error_bars(averages[key], stdevs[key], labels, "Avg " + titles[key], "Avg " + titles[key] + " Over n", type_of_change)