import numpy as np
import pandas as pd

from utils.preprocess import preprocess
from model.neural_network import NeuralNetwork
from trainer.trainer import train
from diagnostics.diagnostics import report

from sklearn.model_selection import train_test_split


df = pd.read_csv(
    "data.csv"
)


X, y, scaler, cdf = preprocess(df)


X_train, X_test, y_train, y_test, cdf_train, cdf_test = train_test_split(
    X,
    y,
    cdf,
    test_size=0.1,
    random_state=42
)


print()

print(
    "Train:",
    X_train.shape
)

print(
    "Test:",
    X_test.shape
)


model = NeuralNetwork(
    input_size=X_train.shape[1]
)


train(
    model,
    X_train,
    y_train,
    lr = 0.02,
    epochs = 4000
)


Z1,A1,Z2,A2,Z3,A3,prediction = model.forward(
    X_test
)


report(
    model,
    prediction,
    y_test,
    A1,
    A2,
    A3
)




real_prediction = (
    prediction
    *
    scaler["y_std"]
    +
    scaler["y_mean"]
)


real_y = (
    y_test
    *
    scaler["y_std"]
    +
    scaler["y_mean"]
)



print(
    "\n========== 10 TEST PREDICTIONS =========="
)


for i in range(10):

    print()

    print(
        "Sample:",
        i
    )


    print(
        "Suburb:",
        cdf_test.iloc[i]["Suburb"]
    )


    print(
        "Type:",
        cdf_test.iloc[i]["Type"]
    )


    print(
        "Rooms:",
        cdf_test.iloc[i]["Rooms"]
    )


    print(
        "Distance:",
        cdf_test.iloc[i]["Distance"]
    )


    print(
        "Landsize:",
        cdf_test.iloc[i]["Landsize"]
    )


    print(
        "Predicted:",
        round(
            float(real_prediction[i][0]),
            2
        )
    )


    print(
        "Actual:",
        round(
            float(real_y[i][0]),
            2
        )
    )


    print(
        "Difference:",
        round(
            abs(
                float(real_prediction[i][0])
                -
                float(real_y[i][0])
            ),
            2
        )
    )