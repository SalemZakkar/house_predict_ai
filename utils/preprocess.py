import numpy as np
import pandas as pd


def preprocess(df):

    df = df.copy()
    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    df["SaleYear"] = df["Date"].dt.year.fillna(2017)

    df["SaleMonth"] = df["Date"].dt.month.fillna(1)

    df["Age"] = (
        df["SaleYear"] - df["YearBuilt"]
    ).clip(0, 150)


    numeric_cols = [
        "Price",
        "Rooms",
        "Distance",
        "Bedroom2",
        "Bathroom",
        "Car",
        "Landsize",
        "BuildingArea",
        "YearBuilt",
        "Propertycount",
        "Lattitude",
        "Longtitude",
        "SaleYear",
        "SaleMonth",
        "Age"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    required = [
        "Price",
        "Rooms",
        "Distance",
        "Bedroom2",
        "Bathroom",
        "Car",
        "Landsize",
        "BuildingArea",
        "YearBuilt",
        "Propertycount",
        "Lattitude",
        "Longtitude",
        "SaleYear",
        "SaleMonth",
        "Age"
    ]

    df = df.dropna(subset=required)



    df = df[
        (df.Price > 10000)
        &
        (df.Rooms > 0)
        &
        (df.Distance >= 0)
        &
        (df.Landsize > 0)
        &
        (df.BuildingArea > 0)
    ]

    df = df.reset_index(drop=True)

    df["Postcode"] = df["Postcode"].astype(str)

    categorical_cols = [
        "Suburb",
        "Type",
        "Method",
        "CouncilArea",
        "Regionname",
        "Postcode"
    ]

    df_cat = pd.get_dummies(
        df[categorical_cols],
        dtype=float
    )

    numeric_features = df[
        [
        "Rooms",
        "Distance",
        "Bedroom2",
        "Bathroom",
        "Car",
        "Landsize",
        "BuildingArea",
        "YearBuilt",
        "Propertycount",
        "Lattitude",
        "Longtitude",
        "SaleYear",
        "SaleMonth",
        "Age"
        ]
    ]

    X_num = numeric_features.values

    X_mean = X_num.mean(axis=0)
    X_std = X_num.std(axis=0)

    X_std[X_std == 0] = 1

    X_num = (X_num - X_mean) / X_std

    X = np.concatenate(
        [
            X_num,
            df_cat.values
        ],
        axis=1
    )

    # ==========================
    # TARGET
    # ==========================

    y = df[["Price"]].values.astype(float)

    y_mean = y.mean()
    y_std = y.std()

    if y_std == 0:
        y_std = 1

    y = (y - y_mean) / y_std

    scaler = {
        "X_mean": X_mean,
        "X_std": X_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "num_features": numeric_features.shape[1]
    }

    print("========== PREPROCESS ==========")
    print("Samples:", len(X))
    print("Features:", X.shape[1])
    print("Categorical:", len(df_cat.columns))

    return X, y, scaler, df