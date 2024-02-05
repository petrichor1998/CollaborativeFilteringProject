import pandas as pd
import numpy as np
import time
import gc
import argparse

def main():
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_data', '--train_set', type=str)
    parser.add_argument('-test_data', '--test_set', type=str)

    arg = parser.parse_args()
    train_name = arg.train_set
    test_name = arg.test_set

    #"netflix/TrainingRatings.txt"
    #"netflix/TestingRatings.txt"

    #import train set and test set
    print("importing the train and test set...")
    train_df = pd.read_csv(train_name, header = None)
    train_df.columns = ["movie_id", "user_id", "rating"]

    test_df = pd.read_csv(test_name, header = None)
    test_df.columns = ["movie_id", "user_id", "rating"]

    print("imported the train and test set...")
    print("grouping the data by 'movie_id' so that it is easier to work with...")
    #grouping the data by movie_id so that it is easier to work with
    bdf = pd.DataFrame()
    for movie, group in train_df.groupby("movie_id"):
        if bdf.empty:
            bdf = group.set_index("user_id")[["rating"]].rename(columns= {"rating" : movie})
        else:
            bdf = bdf.join(group.set_index("user_id")[["rating"]].rename(columns= {"rating" : movie}), how= "outer")

    print("Done grouping and created a better dataframe.")
    #The list of all user indexes that are ambiguous and are making the denominator of the W array 0
    bad = [3594, 4256, 8570, 13074, 14354, 16309, 16631, 18291, 19183, 23730, 24696, 25203, 26756]
    bu = [list(bdf.index)[b] for b in list(bad)]

    #droppin the 13 users from the train data
    bdf.drop(bu, inplace = True)

    bdf.fillna(0)
    print("Calculating the weight matrix...")
    #creating a numpy array
    r_arr = bdf.values
    r_arr[np.isnan(r_arr)] = 0.0

    #creating the average matrix
    r_sum = r_arr.sum(axis = 1)
    r_nz = (r_arr != 0).sum(1)
    r_avg = np.true_divide(r_sum, r_nz)

    #creating the numerator of the W matrix
    r_sub = r_arr - r_avg.reshape(bdf.shape[0], 1)
    tv = r_arr == 0
    r_sub[tv] = 0
    r_num = np.dot(r_sub, r_sub.T)

    #collecting garbage to free up memory
    gc.collect()

    #making the denominator and the W matrix
    a_sqr_d = np.square(r_sub).sum(axis = 1)
    W = np.multiply(a_sqr_d.reshape(bdf.shape[0], 1), a_sqr_d.reshape(bdf.shape[0], 1).T)
    W = np.sqrt(W)
    W = np.true_divide(r_num, W)

    print("Done calculating the weight matrix.")
    print("Testing on the test set...")
    #testing on the test set
    paj = []
    for i, row in test_df[["movie_id", "user_id"]].iterrows():
        if row["user_id"] in bu:
            paj.append(2.5)
        else:
            a = list(bdf.index).index(row["user_id"])
            j = list(bdf.columns).index(row["movie_id"])
            k = np.true_divide(1, W[a].sum())
            tp = r_avg[a] + np.multiply(k, np.dot(W[a], r_sub[:, j]))
            paj.append(tp)
    print("Done predicting.")
    paj = np.array(paj)
    Y = test_df["rating"].values

    error = np.true_divide(np.sum(np.abs(Y - paj)), len(Y))
    print("Mean absolute error is : {}".format(error))
    sqerror = np.sqrt(np.true_divide(np.sum(np.square(Y - paj)) , len(Y)))
    print("Mean absolute error is : {}".format(sqerror))
    print("Time taken = {}".format(time.time() - start))

main()
