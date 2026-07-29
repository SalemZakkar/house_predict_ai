import numpy as np
import pandas as pd


X_mean = None
X_std = None

y_mean = None
y_std = None

numericNum = None


def clean(df):

    global numericNum

    df = df.copy()

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    df["SaleYear"] = (
        df["Date"]
        .dt.year
        .fillna(2017)
    )

    df["SaleMonth"] = (
        df["Date"]
        .dt.month
        .fillna(1)
    )

    df["Age"] = (
        df["SaleYear"] - df["YearBuilt"]
    ).clip(0, 150)


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


    for col in required:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )


    df = df.dropna(
        subset=required
    )


    df = df[
        (df.Price > 300000)
        # &
        # (df.Price < 1000000)
        &
        (df.Rooms > 0)
        &
        (df.Distance >= 0)
        &
        (df.Landsize > 10)
        &
        (df.BuildingArea > 9)
        &
        (df.BuildingArea < df.Landsize)
    ]


    df = df.reset_index(drop=True)


    df["Postcode"] = (
        df["Postcode"]
        .astype(str)
    )


    categorical_cols = [
        "Suburb",
        "Type",
        "Method",
        "CouncilArea",
        "Regionname",
        "SellerG",
        "Postcode",
        "Address"
    ]


    df_cat = pd.get_dummies(
        df[categorical_cols],
        dtype=float
    )


    numeric_cols = [
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

    numericNum = len(numeric_cols)


    X_num = df[numeric_cols].values


    X = np.concatenate(
        [
            X_num,
            df_cat.values
        ],
        axis=1
    )


    y = df[["Price"]].values.astype(float)

    return X, y, df



# =====================================
# SCALE
# =====================================

def scale(X, y):

    X = X.copy()
    y = y.copy()

    global X_mean, X_std
    global y_mean, y_std
    global numericNum


    X_num = X[:, :numericNum]
    X_cat = X[:, numericNum:]


    X_mean = X_num.mean(axis=0)
    X_std = X_num.std(axis=0)

    X_std[X_std == 0] = 1


    X_num = (
        X_num - X_mean
    ) / X_std


    X = np.concatenate(
        [
            X_num,
            X_cat
        ],
        axis=1
    )


    y_mean = y.mean()
    y_std = y.std()

    if y_std == 0:
        y_std = 1


    y = (
        y - y_mean
    ) / y_std


    return X, y



def scaleX(data):
    global X_mean, X_std
    global numericNum
    
    # Split data
    numeric = data[:numericNum]
    cat = data[ numericNum:]
    
    # Convert to float
    numeric = numeric.astype(float)
    
    # Scale numeric features
    numeric = (numeric - X_mean) / X_std
    
    # Combine back
    scaled_data = np.concatenate([numeric, cat], axis=0)
    
    return scaled_data

def scaleY(data):

    global y_mean, y_std

    return (
        data - y_mean
    ) / y_std



def realY(scaledPrice):

    global y_mean, y_std

    return (
        scaledPrice * y_std
        +
        y_mean
    )