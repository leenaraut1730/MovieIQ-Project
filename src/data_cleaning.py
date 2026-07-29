# ====================================
# MovieIQ Project - Data Cleaning
# ====================================

# Import Libraries

import pandas as pd
import numpy as np


# ------------------------------------
# STEP-1 : Load Dataset
# ------------------------------------

df = pd.read_csv("data/movies.csv")

print("\nDataset Loaded Successfully")
print("-" * 50)


# ------------------------------------
# STEP-2 : First Five Rows
# ------------------------------------

print("\nFirst Five Rows")
print(df.head())


# ------------------------------------
# STEP-3 : Dataset Shape
# ------------------------------------

print("\nDataset Shape")
print(df.shape)


# ------------------------------------
# STEP-4 : Dataset Information
# ------------------------------------

print("\nDataset Information")
print(df.info())


# ------------------------------------
# STEP-5 : Column Names
# ------------------------------------

print("\nColumn Names")
print(df.columns)


# Remove Extra Spaces

df.columns = df.columns.str.strip()


# ------------------------------------
# STEP-6 : Missing Values
# ------------------------------------

print("\nMissing Values")
print(df.isnull().sum())


print("\nMissing Values Percentage")

print((df.isnull().sum() / len(df)) * 100)


# ------------------------------------
# STEP-7 : Duplicate Values
# ------------------------------------

print("\nDuplicate Values")

print(df.duplicated().sum())


# Remove Duplicate Rows

df.drop_duplicates(inplace=True)

print("\nDuplicates Removed Successfully")


# ------------------------------------
# STEP-8 : Datatypes
# ------------------------------------

print("\nData Types")

print(df.dtypes)


# ------------------------------------
# STEP-9 : Convert Columns
# ------------------------------------

numeric_columns = [

    "budget",
    "revenue",
    "popularity",
    "runtime",
    "vote_average"

]


for column in numeric_columns:

    df[column] = pd.to_numeric(

        df[column],
        errors="coerce"

    )


# ------------------------------------
# STEP-10 : Missing Values After
# Datatype Conversion
# ------------------------------------

print("\nMissing Values After Conversion")

print(df.isnull().sum())


# ------------------------------------
# STEP-11 : Remove Missing Values
# ------------------------------------

df.dropna(inplace=True)

print("\nMissing Values Removed")


# ------------------------------------
# STEP-12 : Zero Values Check
# ------------------------------------

print("\nBudget = 0")

print((df["budget"] == 0).sum())


print("\nRevenue = 0")

print((df["revenue"] == 0).sum())


print("\nRuntime = 0")

print((df["runtime"] == 0).sum())


# ------------------------------------
# STEP-13 : Remove Zero Values
# ------------------------------------

df = df[df["budget"] > 0]

df = df[df["revenue"] > 0]

df = df[df["runtime"] > 0]


print("\nZero Values Removed Successfully")


# ------------------------------------
# STEP-14 : Genres Cleaning
# ------------------------------------

df["genres"] = (

    df["genres"]
    .astype(str)
    .str.strip()

)


# ------------------------------------
# STEP-15 : Create Success Column
# ------------------------------------

# Success =1

# Revenue > Budget


df["success"] = (

    df["revenue"] >

    df["budget"]

).astype(int)


print("\nSuccess Column Created")


# ------------------------------------
# STEP-16 : Successful Movies
# ------------------------------------

print("\nSuccess Count")

print(

    df["success"]

    .value_counts()

)


# Percentage

print("\nSuccess Percentage")

print(

    round(

        df["success"]
        .value_counts(normalize=True)

        * 100,

        2

    )

)


# ------------------------------------
# STEP-17 : Summary Statistics
# ------------------------------------

print("\nSummary Statistics")

print(

    df.describe()

)


# ------------------------------------
# STEP-18 : Final Dataset Shape
# ------------------------------------

print("\nFinal Dataset Shape")

print(df.shape)


# ------------------------------------
# STEP-19 : Final Missing Values
# ------------------------------------

print("\nFinal Missing Values")

print(

    df.isnull().sum()

)


# ------------------------------------
# STEP-20 : Save Cleaned Dataset
# ------------------------------------

df.to_csv(

    "data/cleaned_movies.csv",

    index=False

)


print("\nCleaned Dataset Saved Successfully")


# ------------------------------------
# STEP-21 : Final Output
# ------------------------------------

print("\nData Cleaning Completed Successfully")

print("-" * 50)

print(df.head())
