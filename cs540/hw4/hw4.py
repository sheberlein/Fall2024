import numpy as np
import scipy
import matplotlib.pyplot as plt
import csv
from scipy.spatial import distance_matrix
from scipy.cluster.hierarchy import dendrogram

# Q3(hac function)) Model: ChatGPT, Prompt: I copied the description of the question and pasted it, and then used it to
# compare my code and fix some things that were not working correctly.
# Q3(hac function) Model: ChatGPT, Prompt: I copied the tie-breaking description from the project and asked for some
# code for it.


def load_data(filepath):
    data = []
    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(dict(row))
    return data

def calc_features(row):
    x1 = float(row["Population"])
    x2 = float(row["Net migration"])
    x3 = float(row["GDP ($ per capita)"])
    x4 = float(row["Literacy (%)"])
    x5 = float(row["Phones (per 1000)"])
    x6 = float(row["Infant mortality (per 1000 births)"])
    features = np.array([x1, x2, x3, x4, x5, x6])
    return features

def hac(features):
    n = len(features)

    # the distance matrix to be maintained.
    d_matrix = distance_matrix(features, features)

    # since we're doing single linkage, we need to replace the zeros with np.inf to help find the min value
    d_matrix = np.where(d_matrix == 0, np.inf, d_matrix)
    
    # create a (n-1) by 4 array
    Z = [[0 for x in range(4)] for y in range(n-1)]

    # iterate through, row by row.
    sizes = {}
    for m in range(n):
        sizes[m] = 1
    
    current_clusters = []
    for i in range(n):
        current_clusters.append(i)
    i = 0
    for row in Z:
        if (len(current_clusters) != 1):
            # for each row, determine which two clusters are closest and put their numbers in the first and second
            # elements of the row, Z[i, 0] and Z[i, 1]. The first element listed, Z[i, 0] should be the smaller 
            # of the 2 cluster indices.
            minimum_distance = np.inf
            lowest_index = -1
            highest_index = -1
            
            for iindex in current_clusters:
                for jindex in current_clusters:
                    if iindex < jindex:
                        d = d_matrix[iindex, jindex]
                        if (d < minimum_distance or (d == minimum_distance and iindex < lowest_index) or (d == minimum_distance and iindex == lowest_index and jindex < highest_index)):
                            minimum_distance = d
                            lowest_index = iindex
                            highest_index = jindex
            
            Z[i][0] = lowest_index
            Z[i][1] = highest_index
    
            # now, the single-linkage distance between the two clusters goes into the 3rd element of the row, Z[i, 2]
            Z[i][2] = minimum_distance
            # the total number of countries in the cluster goes into the fourth element, Z[i, 3]
            Z[i][3] = sizes[lowest_index] + sizes[highest_index]
            
            # now, update the data_with_indices to have the merged clusters
            # The index of the new cluster is n + current number of merges (i)
            
            # now we need to update the distance matrix
            new_row = {}
            for index in current_clusters:
                new_row[index] = min(d_matrix[index, lowest_index], d_matrix[index, highest_index])
            
            # remove the two clusters that were merged
            current_clusters.remove(lowest_index)
            current_clusters.remove(highest_index)
            
            # add the new cluster that is merged
            current_clusters.append(n + i)
            
            # temporarily change the infinities back to zero so that we can make a row/col of zeros to add
            d_matrix = np.where(d_matrix == np.inf, 0, d_matrix)
            
            # make a row/col of zeros so that we can add them
            row1 = [0 for x in range(n + i)]
            
            # now, go through the new row and update those values that we just added
            for item in new_row:
                row1[item] = new_row[item]
            
            # add on the row to the bottom
            d_matrix = np.vstack([d_matrix, row1])
            
            # add a zero so we can have a new column
            row1.append(0)
            
            # add on the column to the right
            d_matrix = np.column_stack([d_matrix, row1])
            
            sizes[n + i] = sizes[lowest_index] + sizes[highest_index]
            
            # change the zeros back to infinities.
            d_matrix = np.where(d_matrix == 0, np.inf, d_matrix)
            
            i = i + 1
    return np.array(Z)

def fig_hac(Z, names):
    # to return one plot, you need to have dendrogram before fig, for some reason
    dendrogram(Z, labels = names, leaf_rotation = 90)
    fig = plt.figure()
    fig.tight_layout()
    return fig

def normalize_features(features):
    # mean and standard deviation, altough never used
    means = np.mean(features, axis = 0)
    standard_deviations = np.std(features, axis = 0)
    
    
    # min and max value of the columns
    col_mins = np.min(features, axis = 0)
    col_maxes = np.max(features, axis = 0)
    
    # normalize
    numerator = features - col_mins
    denominator = col_maxes - col_mins
    normalize = numerator / denominator
    return normalize.tolist()
