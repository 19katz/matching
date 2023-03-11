import pandas as pd
import numpy as np
import os
import sys
import csv
import matplotlib.pyplot as plt


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

num_students = [3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 20]

def graph_min_maj_line_error_plots(n, avg1, stdev1, avg2, stdev2, x_labels, x_label, y_label, title):
    fig, ax = plt.subplots(figsize = (6,4))
    x_pos = np.arange(len(x_labels))
    (line, caps, _) = ax.errorbar(x_pos, avg1, yerr=stdev1, color = "maroon", ecolor = "maroon", fmt='o', ls="-", markersize=6, capsize = 4)
    line.set_label("minority")
    for cap in caps:
        cap.set_markeredgewidth(1)
    (line, caps, _) = ax.errorbar(x_pos, avg2, yerr=stdev2, color = "steelblue", ecolor = "steelblue", fmt='s', ls="-", markersize=6, capsize = 4)
    line.set_label("majority")
    for cap in caps:
        cap.set_markeredgewidth(1)
    ax.set_ylabel(y_label)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel(x_label)
    plt.title(title)
    plt.savefig("./plots/priority/" + str(n) + "_" + title + '.png', bbox_inches='tight', dpi=100)
    plt.close()

def graph_single_line_error_plot(n, averages, stdevs, x_labels, x_label, y_label, title):
    fig, ax = plt.subplots(figsize = (6,4))
    x_pos = np.arange(len(x_labels))
    (line, caps, _) = ax.errorbar(x_pos, averages, yerr=stdevs, color = "maroon", ecolor = "maroon", fmt='o', ls="-", markersize=6, capsize = 4)
    for cap in caps:
        cap.set_markeredgewidth(1)
    ax.set_ylabel(y_label)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel(x_label)
    plt.title(title)
    plt.savefig("./plots/priority/" + str(n) + "_" + title + '.png', bbox_inches='tight', dpi=100)
    plt.close()

def graph_single_line_plot(n, vals, x_labels, x_label, y_label, title):
    fig, ax = plt.subplots(figsize = (6,4))
    x_pos = np.arange(len(x_labels))
    ax.plot(x_pos, vals, color = "maroon", marker='o', ls="-", markersize=6)
    ax.set_ylabel(y_label)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel(x_label)
    plt.title(title)
    plt.savefig("./plots/priority/" + str(n) + "_" + title + '.png', bbox_inches='tight', dpi=100)
    plt.close()

def graph_double_line_plot(n, mins, majs, x_labels, x_label, y_label, title):
    fig, ax = plt.subplots(figsize = (6,4))
    x_pos = np.arange(len(x_labels))
    line = ax.plot(x_pos, mins, label = "minority", color = "maroon", marker='o', ls="-", markersize=6)
    line = ax.plot(x_pos, majs, label = "majority", color = "steelblue", marker='o', ls="-", markersize=6)
    ax.set_ylabel(y_label)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel(x_label)
    plt.title(title)
    plt.savefig("./plots/priority/" + str(n) + "_" + title + '.png', bbox_inches='tight', dpi=100)
    plt.close()

def plot_normal_bar_chart(df, column_list, y_label, title, type_of_change, x_labels):
    fig, ax = plt.subplots(figsize = (6,4))
    x_pos = np.arange(len(x_labels))
    data = df[column_list]
    if len(column_list) == 1:
        df.plot(x = "k", y = column_list, kind = "bar", color = 'maroon', linewidth = 0.5, edgecolor = "black")
    else:
        data.plot.bar(x = "k", stacked = True, color = ["steelblue", "maroon"])
        plt.legend(bbox_to_anchor=(1.0, 1.0))
    ax.set_xlabel("# schools")
    ax.set_ylabel(y_label)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    plt.title(title)
    plt.savefig("./plots/" + type_of_change + "/" + type_of_change + "_" + title + '.png', bbox_inches='tight', dpi=100)
    plt.close()

def get_avg_from_col(val):
    avg, stdev = val.split('(')
    avg = float(avg)
    return avg

def get_stdev_from_col(val):
    avg, stdev = val.split('(')
    stdev = float(stdev[:-1])
    return stdev


if __name__ == "__main__":
    df = pd.read_csv('./all_matching_data_priority.csv')
    names = []
    columns = df.columns
    for column in df.columns:
        if "Avg/Stdev" in column:
            name = column.split("Avg/Stdev ")[1]
            names.append(name)

            df["Avg " + name] = df[column].apply(get_avg_from_col)
            df["Stdev " + name] = df[column].apply(get_stdev_from_col)

            if "Num Worse" in name:
                if "Minority" in name:
                    df["Avg " + name.replace("Num", "Pct")] = df["Avg " + name]/df["k"]
                    df["Stdev " + name.replace("Num", "Pct")] = df["Stdev " + name]/(df["k"] ** 2)
                elif "Majority" in name:
                    df["Avg " + name.replace("Num", "Pct")] = df["Avg " + name]/(df["n"] - df["k"])
                    df["Stdev " + name.replace("Num", "Pct")] = df["Stdev " + name]/((df["n"] - df["k"]) ** 2)
    
    averages = ["Util Welfare", "Pct Worse"]
    other_averages = ["Util Welfare", "Util Count", "Egal Welfare",
                   "Collateral Damage", "Consistent",]
    single_vals = ["Worst Util Welfare", "Best Util Welfare", "Worst Util Count", "Best Util Count", "Worst Egal Welfare", "Worst Util Welfare Envy", "Worst Util Count Envy"]
    for num in num_students:
        cur_df = df[df["n"] == num]
        cur_df = cur_df.sort_values(by=["k"])
        
        for name in averages:
            graph_min_maj_line_error_plots(num, cur_df["Avg Minority " + name], cur_df["Stdev Minority " + name],
                                        cur_df["Avg Majority " + name], cur_df["Stdev Majority " + name],
                                        cur_df["k"], "# minority students", name, "Avg " + name + " for n = " + str(num))
    
        for name in single_vals:
            graph_single_line_plot(num, cur_df[name], cur_df["k"], "# minority students", name, name + " for n = " + str(num))
        
        for name in other_averages:
            graph_single_line_error_plot(num, cur_df["Avg " + name], cur_df["Stdev " + name],
                                        cur_df["k"], "# minority students", name, "Avg " + name + " for n = " + str(num))

        graph_double_line_plot(num, cur_df["Worst Minority Num Worse"], cur_df["Worst Majority Num Worse"], cur_df["k"], "# minority students", "Worst Util Count", "Worst Util Count for n = " + str(num))

    new_df = pd.read_csv('./priority_extra_data.csv')
    counts = {
              "num_changed": "Num Changed",
              "num_unchanged": "Num Changed",
              "worst_util": "Worst Utilitarian Welfare",
              "best_util": "Best Utilitarian Welfare",
              "worst_min_util": "Worst Minority Utilitarian Welfare",
              "best_min_util": "Best Minority Utilitarian Welfare",
              "worst_maj_util": "Worst Majority Utilitarian Welfare",
              "best_maj_util": "Best Majority Utilitarian Welfare",
              "util_envy": "Worst Utilitarian Welfare Envy by Majority Class to Minority Class",
              "util_envy_for_min": "Worst Utilitarian Welfare Envy by Minority Class to Majority Class",
              "count_envy": "Worst Minority and Majority Utilitarian Count Envy"
              }

    for num in num_students:
        print(num)
        cur_df = new_df[new_df["n"] == num]
        cur_df = cur_df.sort_values(by=["k"])
        
        graph_double_line_plot(num, cur_df["worst_min_util"], cur_df["worst_maj_util"], cur_df["k"], 
                               "# minority students", "Utilitarian Welfare", "Worst Minority and Majority Utilitarian Welfare for n = " + str(num))

        graph_double_line_plot(num, cur_df["best_min_util"], cur_df["best_maj_util"], cur_df["k"], 
                               "# minority students", "Utilitarian Welfare", "Best Minority and Majority Utilitarian Welfare for n = " + str(num))
        single_plots= {"worst_util": "Utilitarian Welfare", "best_util": "Utilitarian Welfare", "util_envy": "Utilitarian Welfare Envy", "util_envy_for_min": "Utilitarian Welfare Envy", "count_envy": "Utilitarian Count Envy"}
        for plot in single_plots.keys():
            graph_single_line_plot(num, cur_df[plot], cur_df["k"], "# minority students", single_plots[plot], counts[plot] + " for n = " + str(num))

        plot_normal_bar_chart(cur_df, ["k", "num_unchanged", "num_changed"], "Frequency", "Num Changed and Unchanged for n = " + str(num), "priority", range(0, num))