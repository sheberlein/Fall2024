import csv
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if __name__=="__main__":
    # reading in data
    filename = sys.argv[1]
    learning_rate = float(sys.argv[2])
    iterations = int(sys.argv[3])
    
    # run the rest of the functions here
# QUESTION 2: VISUALIZE DATA
    fig, ax = plt.subplots()
    data = pd.read_csv(filename)
    plt.plot(data.year, data.days)
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Frozen Days")
    plt.savefig("data_plot.jpg")

# QUESTION 3: DATA NORMALIZATION
    print("Q3:")
    n = len(data.days)
    m = min(data.year)
    M = max(data.year)

    x_matrix = []

    for xi in range(n):
        year = data.loc[xi, "year"]
        x_matrix.append([(year - m) / (M - m), 1])
    x_matrix = np.array(x_matrix)
    X_normalized = x_matrix
    print(X_normalized)
    
    print("Q4:")
# QUESTION 4: CLOSED-FORM SOLUTION TO LINEAR REGRESSION
    Yq4 = data.days
    Xq4 = X_normalized
    xtx = np.matmul(Xq4.T, Xq4)
    inverse_xtx = np.linalg.inv(xtx)

    weights = np.matmul(inverse_xtx, np.matmul(Xq4.T, Yq4))
    print(weights)
    
    print("Q5a:")
# QUESTION 5: LINEAR REGRESSION WITH GRADIENT DESCENT
    a = learning_rate
    T = iterations
    X = X_normalized
    y = data.days
    wb = np.zeros(X.shape[1])
    loss_v_iteration = {}
    for t in range(T):
        if t % 10 == 0:
            print(wb)
        y_i_hat = X.dot(wb)
        gradient = (1 / n) * (np.dot(np.transpose(X), np.subtract(y_i_hat, y)))
        wb -= a * gradient
        loss_v_iteration[t] = (1 / (2 * n)) * np.sum((y_i_hat - y) ** 2)

    # plotting
    fig, ax = plt.subplots()
    plt.plot(loss_v_iteration.keys(), loss_v_iteration.values())
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    plt.savefig("loss_plot.jpg")
    print("Q5b: 0.5")
    print("Q5c: 350")

# QUESTION 6: PREDICTION
    w = weights[0]
    b = weights[1]
    data = pd.read_csv(filename)
    x = 2023
    y_hat = w * ((x - m)/(M - m)) + b
    print("Q6: " + str(y_hat))

# QUESTION 7: MODEL INTERPRETATION
    symbol = None
    if w > 0:
        symbol = ">"
    elif w < 0:
        symbol = "<"
    else:
        symbol = "="
    print("Q7a: " + symbol)
    
    print("Q7b: If w > 0, there is a direct relationship between year and days of ice cover. This means that as the year increases, so does the number of days of ice cover. If w < 0, there is an inverse relationship between year and days of ice cover. This means that as the year increases, the number of days of ice cover is expected to decrease. If w = 0, then as the year increases, the number of days of ice cover is expected to stay the same.")

# QUESTION 8: MODEL LIMITATIONS
    x_star = (-b / w) * (M - m) + m
    print("Q8a: " + str(x_star))
    print("Q8b: This is a relatively compelling prediction. It says that Lake Mendota will continue freezing over until the year 2463, which is about 400 years from now. With the current rate of global warming, this is not completely out of the question. One limitation that may cause this prediction to be inaccurate is the lack of multiple features. We are only using the year to predict the days of ice cover, so the data might not be completely accurate. Furthermore, there may be some outlier years that have unusually high or low values for days of ice cover, leading to inaccurate predictions.")
