import csv
from random import randint, shuffle
import os
import numpy as np

csv_malware = "../sha256_family.csv"  # csv file with malware apps
feature_index_dir = 'features_indexes/'  # directory with indexed features for all apps

malware = []
benign = []


def create_list_of_apps():
    print("Creating list of malicious apps...")
    with open(csv_malware, 'r') as file:  # open malware csv file
        next(file)  # skip the header line
        reader = csv.reader(file, delimiter=',')  # read the csv malware families
        for row in reader:
            malware.append(row[0])  # append every row from the csv file into a list
    print("Malware apps found: ", len(malware))  # 5560
    print("Malware sample: ", malware[randint(0, len(malware) - 1)])  # print a random malware sample

    print("Creating list of benign apps...")
    for filename in os.listdir(feature_index_dir):  # read all apps
        if filename not in malware:  # if a SHA name not in malware list, append it to benign list
            benign.append(filename)
    print("Benign apps found: ", len(benign))  # 123453
    print("Benign app sample: ", benign[randint(0, len(benign) - 1)], )  # print a random benign app

    print("Total apps (Benign & Malicious) found: ", len(malware) + len(benign))  # 129013


malware_incremental_counter = 0
benign_incremental_counter = 0


def generate_set_incremental(set_size, malware_ratio):
    global malware_incremental_counter, benign_incremental_counter
    set = []  # list that will fill with app set

    print("Creating set with", set_size, "samples...")
    print("Malware ratio:", int(malware_ratio * 100), "%, totaling", int(set_size * malware_ratio), "apps in", set_size)
    print("Creating malware set...")

    while len(set) < (set_size * malware_ratio):
        app = malware[malware_incremental_counter]  # locate malware based on random index in malware list
        malware_incremental_counter += 1
        if malware_incremental_counter >= 5560:
            break
        if app not in set:
            set.append(app)  # append malware to set list

    print("Total malware apps in set: ", len(set))
    print("Malware sample in set: ", set[0])

    print("Creating benign set...")

    while len(set) < set_size:
        app = benign[benign_incremental_counter]  # locate benign based on random index in benign list
        benign_incremental_counter += 1
        if benign_incremental_counter >= 123453:
            break
        if app not in set:
            set.append(app)  # append benign to set list
    print(malware_incremental_counter)
    print("Total apps (malicious and benign) in set: ", len(set))
    return set


def generate_set(set_size, malware_ratio):
    set = []  # list that will fill with app set

    print("Creating set with", set_size, "samples...")
    print("Malware ratio:", int(malware_ratio * 100), "%, totaling", int(set_size * malware_ratio), "apps in", set_size)
    print("Creating malware set...")

    while len(set) < (set_size * malware_ratio):
        index = randint(0, len(malware) - 1)  # choose a random index between (0,5559)
        app = malware[index]  # locate malware based on random index in malware list
        if app not in set:
            set.append(app)  # append malware to set list

    print("Total malware apps in set: ", len(set))
    print("Malware sample in set: ", set[0])

    print("Creating benign set...")
    while len(set) < set_size:
        index = randint(0, len(benign) - 1)  # choose a random index between (0,129012)
        app = benign[index]  # locate benign based on random index in benign list
        if app not in set:
            set.append(app)  # append benign to set list

    print("Total apps (malicious and benign) in set: ", len(set))
    return set


def generate_input(set, total_features):
    print("performing one hot encoding...")
    # return a 2D array filled with zeros that will be used for the features of each app
    data = np.zeros((len(set), total_features), dtype=float)
    # return an array filled with zeros that will be used for the label of each app {0-benign 1-malicious}
    labels = np.zeros((len(set),), dtype=int)

    shuffle(set)  # shuffle the set
    for id_app, app in enumerate(set):  # iterate through set with a counter
        with open(feature_index_dir + app, 'r') as file:  # open apps in set
            for index in file:  # read line by line
                data[id_app][int(index)] = 1.0  # update corresponding element of the array with 1.0

        if app in malware:
            labels[id_app] = 1  # update corresponding label to 1 if it is malware
        else:
            labels[id_app] = 0

    #print(data)
    #print(labels)
    #print(data.shape)
    #print(labels.shape)
    return data, labels
