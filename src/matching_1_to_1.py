import copy  # deepcopy constructs a new compound object, recursively, inserts copies into it
import random
import math
import time
import itertools
import sys
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

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

class Person:
    # constructor to initialize the attributes of Person class
    def __init__(self, name, preferences):
        self.name = name
        self.partner = None
        self.preferences = preferences

    # return object representation
    def __repr__(self):
        if self.partner:
            return f'{self.name} ⚭ {self.partner}'
        else:
            return f'{self.name} ⌀'


class Alpha(Person):
    def __init__(self, name, preferences):
        # super() refers to parent class, and inherits methods
        super().__init__(name, preferences)
        # prefered person not asked yet
        # recursively copy
        self.not_asked = copy.deepcopy(preferences)

    def ask(self):
        # drop the first element which is the next preferred person
        return self.not_asked.pop(0)

    # for check_stability function
    def accept(self, suitor):
        return self.partner is None or(
            # check that the suitor is strictly preferred to the existing partner
            self.preferences.index(suitor) <
            self.preferences.index(self.partner)
        )


class Beta(Person):
    def __init__(self, name, preferences):
        super().__init__(name, preferences)
        # this person does not ask

    def accept(self, suitor):
        return self.partner is None or(
            # check that the suitor is strictly preferred to the existing partner
            self.preferences.index(suitor) <
            self.preferences.index(self.partner)
        )


def setup(preferences, proposing, accepting):
    """
    Initialize the set up and return a dictionary of alphas and betas.

    No one is matched at the beginning.
    """
    # modify the variable in a local context
    global alphas
    global betas

    alphas = {}
    # loop over the preferences proposing
    for key, value in preferences.get(proposing).items():
        alphas[key] = Alpha(key, value)

    betas = {}
    for key, value in preferences.get(accepting).items():
        betas[key] = Beta(key, value)


def run_da(preferences, proposing, accepting):
    """
    Run the deferred acceptance algo and print the match results.

    1) Each unengaged man propose to the woman he prefers most
    2) Each woman says "maybe" to the suitor she most prefers and "no" to all other suitors
    3) Continue while there are still unengaged men
    """
    # Friends came out in 1994
    random.seed(1994)
    setup(preferences, proposing, accepting)
    #print("Proposing: ", alphas)
    #print("Accepting: ", betas)
    #print()
    # all alphas are unmatched at the beginning
    unmatched = list(alphas.keys())

    while unmatched:
        # randomly select one of the alphas to choose next
        alpha = alphas[random.choice(unmatched)]
        # alpha ask his first choice
        beta = betas[alpha.ask()]
        #print(f'{alpha.name} asks {beta.name}')
        # if beta accepts alpha's proposal
        if beta.accept(alpha.name):
            #print(f'{beta.name} accepts')
            # # if beta has a partner
            if beta.partner:
                # this existing alpha partner is now an ex
                ex = alphas[beta.partner]
                #print(f'{beta.name} dumps {ex.name}')
                # this alpha person has no partner now :(
                ex.partner = None
                # add this alpha person back to the list of unmatched
                unmatched.append(ex.name)
            unmatched.remove(alpha.name)
            # log the match
            alpha.partner = beta.name
            beta.partner = alpha.name
        #else:
            #print(f'{beta.name} rejects')
            # move on to the next unmatched male
    #print()
    #print("Everyone is matched. This is a stable matching")
    return alphas, betas


def print_pairings(people):
    for p in people.values():
        if p.partner:
            print(
                f'{p.name} is paired with {p.partner} ({p.preferences.index(p.partner) + 1})')
        else:
            print(f'{p.name} is not paired')


def check_not_top_matches(matches):
    '''Generate a list of people who do not have their top matches'''
    not_top_matches = []
    for person in matches.keys():
        if matches[person].partner != matches[person].preferences[0]:
            not_top_matches.append(person)
    return not_top_matches

def check_stability(proposing, accepting, list_of_not_top_matches):
    for i in list_of_not_top_matches:
        more_preferred = proposing[i].preferences[:proposing[i].preferences.index(
            proposing[i].partner)]
        # check to see if it's reciprocated
        for j in more_preferred:
            # print reason why the female rejects
            if accepting[j].accept(proposing[i].name) == False:
                print(
                    f'{proposing[i].name} prefers {accepting[j].name} more, but {accepting[j].name} prefers {accepting[j].partner}.')
            else:
                print("This matching is NOT stable!")
                break
    print("Therefore, this matching is stable.")


def create_schools_and_students(num_schools, num_students):
    schools_list = []
    for i in range(num_schools):
        school_name = "c" + str(i + 1)
        schools_list.append(school_name)

    students_list = []
    for i in range(num_students):
        student_name = "s" + str(i + 1)
        students_list.append(student_name)
    
    return students_list, schools_list

def create_student_rankings(students_list, schools):
    student_preferences = {}
    for s in students_list:
        student_preferences[s] = list(np.random.permutation(schools))

    return student_preferences

def swap_random(seq):
    new_seq = seq.copy()
    idx = range(len(new_seq) - 1)
    i1 = random.sample(idx, 1)[0]
    new_seq[i1], new_seq[i1 + 1] = new_seq[i1 + 1], new_seq[i1]
    return new_seq

def create_school_rankings(ranking_mode, schools_list, students,  k):
    normal_school_preferences = {}
    aff_act_school_preferences = {}
    unchanged = []
    better = []
    worse = []
    minority_students = []

    for i in range(len(schools_list)):
        normal_prefs = list(np.random.permutation(students))
        if i == 0:
            if ranking_mode == "priority":
                c1_aff_act_prefs = normal_prefs.copy()
                minority_student_inds = random.sample(range(1, len(students)), k)
                minority_students = [c1_aff_act_prefs[i] for i in minority_student_inds]
                new_inds = []
                for j in range(0, len(c1_aff_act_prefs)):
                    cur_student = c1_aff_act_prefs[j]
                    #print(cur_student)
                    if cur_student in minority_students:
                        swap_min = 0
                        if len(new_inds) > 0:
                            swap_min = new_inds[-1] + 1
                            swap_max = j - 1
                        else:
                            swap_min = 0
                            swap_max = j - 1
                        new_ind = random.choice(range(swap_min, swap_max + 1))
                        #print(new_ind)
                        c1_aff_act_prefs.insert(new_ind, c1_aff_act_prefs.pop(j))
                        new_inds.append(new_ind)
                        #print(new_inds)
            elif ranking_mode == "swapped":
                c1_aff_act_prefs = swap_random(normal_prefs)
            else:
                c1_aff_act_prefs = list(np.random.permutation(students))
            aff_act_school_preferences[schools_list[i]] = list(c1_aff_act_prefs)
            normal_school_preferences[schools_list[i]] = list(normal_prefs)
            c1_normal_inds, c1_aff_act_inds, rank_change_welfare, highest_ranked_strictly_better = compare_school_rankings(list(normal_prefs), list(c1_aff_act_prefs), students)
        else:
            aff_act_school_preferences[schools_list[i]] = list(normal_prefs)
            normal_school_preferences[schools_list[i]] = list(normal_prefs)

    return normal_school_preferences, aff_act_school_preferences, minority_students, c1_normal_inds, c1_aff_act_inds, rank_change_welfare, highest_ranked_strictly_better

def compare_school_rankings(normal_prefs, aff_act_prefs, students_list):
    unchanged = []
    rank_change_welfare = {"better": [], "unchanged": [], "worse": []}
    
    c1_normal_inds = [normal_prefs.index(s) for s in students_list]
    c1_aff_act_inds = [aff_act_prefs.index(s) for s in students_list]

    better = []
    worse = []

    for i in range(len(c1_normal_inds)):
        if c1_aff_act_inds[i] > c1_normal_inds[i]:
            rank_change_welfare["worse"].append(students_list[i])
        elif c1_aff_act_inds[i] < c1_normal_inds[i]:
            rank_change_welfare["better"].append(students_list[i])
        else:
            rank_change_welfare["unchanged"].append(students_list[i])


    highest_ranked_strictly_better = ""
    highest_ind = len(students_list) - 1
    for s in rank_change_welfare["better"]:
        next_ind = aff_act_prefs.index(s) 
        if next_ind < highest_ind:
            highest_ind = next_ind
            highest_ranked_strictly_better = s

    return c1_normal_inds, c1_aff_act_inds, rank_change_welfare, highest_ranked_strictly_better

def find_num_worse_off(old_inds, new_inds, students_list, rank_change_welfare):
    new_match_welfare = {"better": [], "worse": [], "unchanged": []}
    total_index_diff = 0
    max_worse = 0
    num_worst_off = 0
    max_worse_is_better = False
    ind_diffs = []

    for i in range(len(old_inds)):
        old_index = old_inds[i]
        new_index = new_inds[i]

        ind_diff = old_index - new_index
        ind_diffs.append(ind_diff)
        
        total_index_diff += ind_diff
        if new_index > old_index:
            new_match_welfare["worse"].append(students_list[i])
            if new_index - old_index == len(students_list) - 1:
                num_worst_off += 1
            if ind_diff < max_worse:
                max_worse = ind_diff
            if students_list[i] in rank_change_welfare["better"]:
                max_worse_is_better = True
            else:
                max_worse_is_better = False
        elif new_index < old_index:
            new_match_welfare["better"].append(students_list[i])
        else:
            new_match_welfare["unchanged"].append(students_list[i])
    
    return new_match_welfare, ind_diffs, total_index_diff, max_worse, num_worst_off, max_worse_is_better

def find_overlaps(arr1, arr2):
    return list(set(arr1) & set(arr2))

def find_egal_welfare_priority(students_list, ind_diffs, minority_students):
    maj_max_worse = 0
    min_max_worse = 0
    for i in range(len(ind_diffs)):
        diff = ind_diffs[i]
        student = students_list[i]
        if student in minority_students and diff < min_max_worse:
            min_max_worse = diff
        elif student not in minority_students and diff < maj_max_worse:
            maj_max_worse = diff
    
    return maj_max_worse, min_max_worse



def prepare_result(type_of_change, students_list, students_preferences, schools_preferences, aff_act_schools_preferences, minority_students, rank_change_welfare, old_inds, new_inds, normal_match, aff_act_match, highest_ranked_strictly_better):

    new_match_welfare, ind_diffs, total_index_diff, max_worse, num_worst_off, max_worse_is_better = find_num_worse_off(old_inds, new_inds, students_list, rank_change_welfare)
    total_min_ind_diff = 0
    total_maj_ind_diff = 0

    if type_of_change == "priority":
        for i in range(len(ind_diffs)):
            if students_list[i] in minority_students:
                total_min_ind_diff += ind_diffs[i]
            else:
                total_maj_ind_diff += ind_diffs[i]

    keys = ["better", "worse", "unchanged"]
    overlaps = {}

    for key1 in keys:
        for key2 in keys:
            overlaps[(key1, key2)] = len(find_overlaps(rank_change_welfare[key1], new_match_welfare[key2]))
    
    min_maj_overlaps = {}
    majority_students = list(set(students_list) - set(minority_students))

    for key in keys:
        min_maj_overlaps[("min", key)] = len(find_overlaps(minority_students, new_match_welfare[key]))
        min_maj_overlaps[("maj", key)] = len(find_overlaps(majority_students, new_match_welfare[key]))

    highest_ranked_is_worse = (highest_ranked_strictly_better in new_match_welfare["worse"])
    highest_ranked_is_better = (highest_ranked_strictly_better in new_match_welfare["better"])
    
    if highest_ranked_is_worse:
        highest_ranked_status = "worse"
    elif highest_ranked_is_better:
        highest_ranked_status = "better"
    else:
        highest_ranked_status = "unchanged"

    string_to_print = "Student preferences: " + str(students_preferences) + "\n"
    string_to_print += "School preferences: " + str(schools_preferences) + "\n"
    string_to_print += "Affirmative Action School preferences: " + str(aff_act_schools_preferences) + "\n"
    # string_to_print += "Unchanged individuals: " + str(unchanged) + "\n"
    string_to_print += "Old Match: " + normal_match + "\n"
    string_to_print += "New Match: " + aff_act_match + "\n"
    # string_to_print += "Students made better: " + str(len(better)) + " " + str(better) + "\n"
    # string_to_print += "Students Worse Off: " + str(num_worse) + " " + str(worse_off) + "\n"
    # string_to_print += "Unchanged Students Worse Off: " + str(num_worse_off_and_unchanged) + " " + str(worse_off_and_unchanged) + "\n"
    # string_to_print += "Students Better Off: " + str(num_better) + " " + str(better_off) + "\n"
    # string_to_print += "Better off Students who are Better Off: " + str(num_better_off_and_better) + " " + str(better_off_and_better) + "\n"
    # string_to_print += "Better off Students who are Worse Off: " + str(num_worse_off_and_better) + " " + str(worse_off_and_better) + "\n"
    # string_to_print += "Highest Ranked Status: " + highest_ranked_status + "\n"
    # string_to_print += "Max Amount Worsened (for 1 student): " + str(max_worse) + "\n"
    # string_to_print += "Max student worse is not worse: " + str(max_worse_is_unchanged_or_better) + "\n"
    # string_to_print += "Amount Changed (total): " + str(total_index_diff) + "\n"

    return string_to_print, overlaps, min_maj_overlaps, new_match_welfare, ind_diffs, total_index_diff, total_min_ind_diff, total_maj_ind_diff, max_worse, max_worse_is_better, num_worst_off, highest_ranked_is_better, highest_ranked_is_worse

def write_list_to_file(file_name, string_list):
    with open(r'./' + file_name, 'w') as fp:
        for string in string_list:
            # write each item on a new line
            fp.write("%s \n" % string)

def write_to_csv(file_name, string_list):
    with open(r'./logs/' + file_name, 'w', newline='') as fp:
        csvwriter = csv.writer(fp, delimiter=',')
        csvwriter.writerow(string_list)

def find_highest_ranked_better_off_pref_c1(old_c1_comparison, students_list, normal_prefs, aff_act_prefs, better):
    highest_ranked_strictly_better = ""
    highest_ind = len(students_list) - 1
    for i in range(len(students_list)):
        s = students_list[i]
        if old_c1_comparison[i] and s in better:
            next_ind = aff_act_prefs.index(s) 
            if next_ind < highest_ind:
                highest_ind = next_ind
                highest_ranked_strictly_better = s
    
    return highest_ranked_strictly_better

def create_histogram_plot_pct(data, n, x_axis, title, type_of_change):
    fig, ax = plt.subplots(figsize = (6,4))
    # Later in the code
    print(len(data))
    print(Counter(data))
    plt.locator_params(axis="both", integer=True, tight=True)
    ax.hist(data, bins = list(np.arange(-1 * (n - 1), n + 1)), color = "darkred", linewidth = 0.5, edgecolor = "black", weights=np.ones(len(data)) / len(data))
    ax.set_xlabel(x_axis)
    ax.set_xticks(np.arange(-1 * (n - 1), n + 1))
    ax.set_ylabel("Percent Frequency")
    plt.title(title)
    plt.savefig("./plots/" + type_of_change + "/" + str(n) + "_" + title + '_pct.png', bbox_inches='tight', dpi=100)
    plt.close()

def create_histogram_plot_pct_min(data, n, k, x_axis, title, type_of_change):
    fig, ax = plt.subplots(figsize = (6,4))
    # Later in the code
    plt.locator_params(axis="both", integer=True, tight=True)
    ax.hist(data, bins = list(np.arange(-1 * (n), n + 1)), color = "darkred", linewidth = 0.5, edgecolor = "black", weights=np.ones(len(data)) / len(data))
    ax.set_xlabel(x_axis)
    ax.set_xticks(np.arange(-1 * n, n + 1))
    ax.set_ylabel("Percent Frequency")
    plt.title(title)
    plt.savefig("./plots/" + type_of_change + "/" + str(n) + "_" + str(k) + "_" + title + '_pct.png', bbox_inches='tight', dpi=100)
    plt.close()

def graph_dot_error_plot(n, k, averages, stdevs, x_labels, x_label, y_label, title, type_of_change):
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
    plt.savefig("./plots/" + type_of_change + "/" + str(n) + "_" + str(k) + "_" + title + '.png', bbox_inches='tight', dpi=100)
    plt.close()

if __name__ == '__main__':
    # preferences
    n = int(sys.argv[1])
    type_of_change = sys.argv[2]
    num_its = int(sys.argv[3])
    k = 0
    if type_of_change == "priority":
        k = int(sys.argv[4])
    file_name = "log_" + str(n) + "_" + type_of_change + "_" + str(num_its) + "_" + str(k) + ".csv"
    mode = sys.argv[5]

    all_ind_diffs = []

    print("NUMBER OF STUDENTS AND SCHOOLS: " + str(n))
    print("Type of change: " + type_of_change)
    total_worse_off = 0
    rank_change_welfare_change_df = pd.DataFrame({'rank_change': pd.Series(dtype='int'),
                                                  'welfare_change': pd.Series(dtype='int')})

    max_num_worse_off = 0
    worst_total_change = 0
    worst_single_change = 0
    num_changed = 0
    total_better_off_only = 0
    total_worse_off_only = 0
    highest_ranked_worse_list = []

    students_list, schools_list = create_schools_and_students(n, n)

    csv_header = ["id", "is_changed", "max_worse", "util", "num_worse", "num_better", "num_unchanged",
                    "rank_better", "rank_worse", "rank_unchanged", "rank_better_match_better",
                    "rank_better_match_unchanged", "rank_better_match_worse", "rank_unchanged_match_better",
                    "rank_unchanged_match_unchanged", "rank_unchanged_match_worse", "rank_worse_match_better",
                    "rank_worse_match_unchanged", "rank_worse_match_worse"]

    if type_of_change == "priority":
        csv_header.extend(["k", "maj_better", "maj_unchanged", "maj_worse",
                           "maj_util", "min_util", "maj_egal", "min_egal"])

    # with open(r'./logs/' + file_name, 'w', newline='') as fp:
    #     csvwriter = csv.writer(fp, delimiter=',')
    #     csvwriter.writerow(csv_header)

    for i in range(num_its):
        print(i)
        random.seed(time.time())
        people = {}
        aff_act_people = {}
        students_list, schools_list = create_schools_and_students(n, n)
        schools_preferences, aff_act_schools_preferences, minority_students, c1_normal_inds, c1_aff_act_inds, rank_change_welfare, highest_ranked_strictly_better = create_school_rankings(type_of_change, schools_list, students_list, k)
        students_preferences = create_student_rankings(students_list, schools_list)
        people["Students"] = students_preferences
        people["Schools"] = schools_preferences
        aff_act_people["Students"] = students_preferences
        aff_act_people["Schools"] = aff_act_schools_preferences

        alphas, betas = run_da(people, "Students", "Schools")
        normal_match = str(alphas)
        aff_act_c1_prefs = aff_act_schools_preferences["c1"]
        c1_prefs = schools_preferences["c1"]
        old_inds = [students_preferences[student].index(alphas[student].partner) for student in students_list]
        old_c1_comparison = [students_preferences[student].index("c1") < students_preferences[student].index(alphas[student].partner) for student in students_list]
        c1_old_partner = betas["c1"].partner
        c1_old_ind = aff_act_c1_prefs.index(c1_old_partner)
        c1_old_match_ind = schools_preferences["c1"].index(c1_old_partner)
        student_old_inds = [c1_prefs.index(s_i) for s_i in students_list]
        student_new_inds = [aff_act_c1_prefs.index(s_i) for s_i in students_list]
        rank_changes = [o_i - n_i for o_i, n_i in zip(student_old_inds, student_new_inds)]
        print(rank_changes)
        aff_act_alphas, aff_act_betas = run_da(aff_act_people, "Students", "Schools")
        aff_act_match = str(aff_act_alphas)
        new_inds = [students_preferences[student].index(aff_act_alphas[student].partner) for student in students_list]
        c1_new_partner = betas["c1"].partner
        c1_new_ind = aff_act_c1_prefs.index(c1_new_partner)
        
        highest_ranked_strictly_better = find_highest_ranked_better_off_pref_c1(old_c1_comparison, students_list, schools_preferences["c1"], c1_prefs, rank_change_welfare["better"])

        string_to_print, overlaps, min_maj_overlaps, new_match_welfare, ind_diffs, total_index_diff, total_min_ind_diff, total_maj_ind_diff, max_worse, max_worse_is_better, num_worst_off, highest_ranked_is_better, highest_ranked_is_worse = prepare_result(type_of_change, students_list, students_preferences,
            schools_preferences, aff_act_schools_preferences, minority_students, rank_change_welfare, old_inds, new_inds, normal_match, aff_act_match, highest_ranked_strictly_better)

        to_write_csv = []
        num_worse_off = len(new_match_welfare["worse"])
        num_better_off = len(new_match_welfare["better"])
        num_unchanged = len(new_match_welfare["unchanged"])
        num_ranked_worse = len(rank_change_welfare["worse"])
        num_ranked_better = len(rank_change_welfare["better"])
        num_rank_unchanged = len(rank_change_welfare["unchanged"])

        is_changed = False

        print(ind_diffs)
        
        if num_worse_off >= 1 or num_better_off >= 1:
            all_ind_diffs.extend(ind_diffs)
            is_changed = True
            num_changed += 1
            cur_it_rank_welfare_dict = {
                "rank_change": rank_changes,
                "welfare_change": ind_diffs
            }

            cur_it_rank_welfare_df = pd.DataFrame(data=cur_it_rank_welfare_dict)
            rank_change_welfare_change_df = pd.concat([rank_change_welfare_change_df, cur_it_rank_welfare_df])
        if num_worse_off == 0 and num_better_off >= 1:
            total_better_off_only += 1
        if num_worse_off >= 1 and num_better_off == 0:
            total_worse_off_only += 1
        if num_worse_off >= 1:
            total_worse_off += 1
            if num_worse_off > max_num_worse_off:
                max_num_worse_off = num_worse_off
        if total_index_diff != 0:
            if total_index_diff < worst_total_change:
                worst_total_change = total_index_diff
        if max_worse > 1:
            if max_worse > worst_single_change:
                worst_single_change = max_worse
        if highest_ranked_is_worse:
            highest_ranked_worse_list.append(string_to_print)
        
        if type_of_change == "priority":
            maj_max_worse, min_max_worse = find_egal_welfare_priority(students_list, ind_diffs, minority_students)
        
        to_write_csv = [i, is_changed, max_worse, total_index_diff, num_worse_off, num_better_off,
                        num_unchanged, num_ranked_better, num_ranked_worse, num_rank_unchanged,
                        overlaps[("better", "better")], overlaps[("better", "unchanged")],
                        overlaps[("better", "worse")], overlaps[("unchanged", "better")],
                        overlaps[("unchanged", "unchanged")], overlaps[("unchanged", "worse")],
                        overlaps[("worse", "better")], overlaps[("worse", "unchanged")],
                        overlaps[("worse", "worse")]]

        if type_of_change == "priority":
            to_write_csv.extend([k, min_maj_overlaps[("maj", "better")],
                                min_maj_overlaps[("maj", "unchanged")], min_maj_overlaps[("maj", "worse")], 
                                total_maj_ind_diff, total_min_ind_diff, maj_max_worse, min_max_worse])
        if is_changed and type_of_change == "priority":
            assert(overlaps[("better", "better")] == min_maj_overlaps[("min", "better")])
            assert(overlaps[("better", "unchanged")] == min_maj_overlaps[("min", "unchanged")])
            assert(overlaps[("better", "worse")] == min_maj_overlaps[("min", "worse")])
        assert(len(to_write_csv) == len(csv_header)) 
        if mode == "write_data":
            csvwriter.writerow(to_write_csv)
    print(highest_ranked_worse_list)  

    # write_list_to_file("max_num_worse.txt", by_worse_off[max_num_worse_off])
    # write_list_to_file("worst_total_change.txt", by_total_change[worst_total_change])
    # write_list_to_file("max_collateral_damage.txt", by_worse_off_unchanged[max_num_worse_unchanged])
    # write_list_to_file("worst_single_change.txt", by_worst_change[worst_single_change])
    # write_list_to_file("better_off_only.txt", better_off_only_list)
    # write_list_to_file("worse_off_only.txt", worse_off_only_list)
    # write_list_to_file("c1_worse_off.txt", c1_worse_off_list)
    # write_list_to_file("better_off_students_all_worse_off.txt", better_off_students_all_worse_off)
    # write_list_to_file("better_off_students_worse_off.txt", better_off_students_worse_off)
    # write_list_to_file("highest_ranked_worse_off.txt", highest_ranked_worse_list)
    # write_list_to_file("worst_single_change_not_worse.txt", worst_single_change_not_worse)

    if mode == "rank_stats":
        average_ind_diff = round(np.mean(all_ind_diffs), 4)
        stdev_ind_diff = round(np.std(all_ind_diffs), 2)
        rank_file_name = "rank_stats_" + str(type_of_change) + ".csv"
        if type_of_change == "priority":
            rank_header = ["n", "k", "Avg Change", "Stdev Change"]
            rank_row = [n, k, average_ind_diff, stdev_ind_diff]
        else:
            rank_header = ["n", "Avg Change", "Stdev Change"]
            rank_row = [n, average_ind_diff, stdev_ind_diff]

        with open(r'./' + rank_file_name, 'a', newline='') as rf:
            csvwriter = csv.writer(rf, delimiter=',')
            #csvwriter.writerow(rank_header)
            #csvwriter.writerow(rank_row)
        rf.close()

        if type_of_change == "priority":
            create_histogram_plot_pct_min(all_ind_diffs, n, k, "rank change", "Rank Changes for Students when n = " + str(n) + ", k = " + str(k), type_of_change)
        else:
            create_histogram_plot_pct(all_ind_diffs, n, "rank change", "Rank Changes for Students when n = " + str(n), type_of_change)
    
    print(rank_change_welfare_change_df.columns)
    if mode == "rank_welfare":
        unique_rank_changes = rank_change_welfare_change_df["rank_change"].unique()
        unique_rank_changes.sort()
        print(unique_rank_changes)
        averages = []
        stdevs = []
        for rank_change in unique_rank_changes:
            cur_df = rank_change_welfare_change_df[rank_change_welfare_change_df["rank_change"] == rank_change]["welfare_change"]
            averages.append(np.mean(cur_df))
            stdevs.append(np.std(cur_df))
        
        title = "Average Change in Match Ranking vs Preference Ranking Change for n = " + str(n)
        if type_of_change == "priority":
            title += ", k = " + str(k)

        graph_dot_error_plot(n, k, averages, stdevs, unique_rank_changes, "change in preference ranking", "Average Rank Change in Match", title, type_of_change)

    print("Max Num Worse Off: " + str(max_num_worse_off))
    print("Worst Total Change: " + str(worst_total_change))
    print("Num worse off: " + str(total_worse_off))
    print("Num unchanged: " + str(num_its - num_changed))
    #print("Num better off: " + str(num_changed - total_worse_off_only))
    #print("Num better off only: " + str(total_better_off_only))
    #print("Num worse off only: " + str(total_worse_off_only))
    #print("Num times C1 worse off: " + str(c1_worse_off))
    #print("Num times C1's old/new partners are better off: " + str(total_c1_old_new_better_off))
    #print("Num times C1's old/new partners are worse off: " + str(total_c1_old_new_worse_off))
    #print("Max num worst off: " + str(max_num_worst_off))
    #'''
