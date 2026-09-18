
# ============================================================
# TASK 1: EXPLORATORY DATA ANALYSIS ON TITANIC DATASET
# ============================================================

# ------------------------------------------------------------
# 1. IMPORT LIBRARIES
# ------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

# Plot style
sns.set_theme(style="whitegrid")

print("Libraries imported successfully!")


# ------------------------------------------------------------
# 2. LOAD DATASET
# ------------------------------------------------------------

# Make sure Titanic-Dataset.csv is in the same folder
df = pd.read_csv("Titanic-Dataset.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ------------------------------------------------------------
# 3. DISPLAY FIRST FIVE ROWS
# ------------------------------------------------------------

print("\nFirst five rows:")
display(df.head())


# ------------------------------------------------------------
# 4. DISPLAY RANDOM SAMPLE
# ------------------------------------------------------------

print("\nRandom sample of 10 rows:")
display(df.sample(10, random_state=42))


# ------------------------------------------------------------
# 5. DISPLAY DATASET INFORMATION
# ------------------------------------------------------------

print("\nDataset information:")
df.info()


# ------------------------------------------------------------
# 6. DISPLAY NUMBER OF ROWS AND COLUMNS
# ------------------------------------------------------------

print("\nDataset shape:")
print(df.shape)

print("\nNumber of rows:", df.shape[0])
print("Number of columns:", df.shape[1])


# ------------------------------------------------------------
# 7. DISPLAY DATA TYPES
# ------------------------------------------------------------

print("\nData types:")
print(df.dtypes)


# ------------------------------------------------------------
# 8. CHECK NON-NULL VALUES
# ------------------------------------------------------------

print("\nNon-null values in each column:")
print(df.notnull().sum())


# ------------------------------------------------------------
# 9. CHECK DUPLICATE ROWS
# ------------------------------------------------------------

print("\nNumber of duplicate rows:")
print(df.duplicated().sum())


# ------------------------------------------------------------
# 10. CHECK UNIQUE VALUES OF SURVIVED COLUMN
# ------------------------------------------------------------

if "Survived" in df.columns:
    print("\nUnique values in Survived column:")
    print(df["Survived"].unique())


# ------------------------------------------------------------
# 11. COUNT UNIQUE VALUES IN EACH COLUMN
# ------------------------------------------------------------

print("\nNumber of unique values in each column:")
print(df.nunique())


# ------------------------------------------------------------
# 12. DESCRIPTIVE STATISTICS
# ------------------------------------------------------------

print("\nDescriptive statistics:")
display(df.describe())


# ------------------------------------------------------------
# 13. MISSING VALUES COUNT
# ------------------------------------------------------------

print("\nMissing values in each column:")
missing_values = df.isnull().sum()
print(missing_values)


# ------------------------------------------------------------
# 14. MISSING VALUES PERCENTAGE
# ------------------------------------------------------------

missing_percentage = (df.isnull().sum() / len(df)) * 100

missing_table = pd.DataFrame({
    "Missing Values": missing_values,
    "Missing Percentage": missing_percentage
})

print("\nMissing values report:")
display(missing_table)


# ------------------------------------------------------------
# 15. VISUALIZE MISSING VALUES
# ------------------------------------------------------------

plt.figure(figsize=(10, 5))

sns.heatmap(
    df.isnull(),
    cbar=False,
    yticklabels=False,
    cmap="viridis"
)

plt.title("Missing Values Heatmap")
plt.xlabel("Columns")
plt.ylabel("Rows")
plt.show()


# ------------------------------------------------------------
# 16. REMOVE DUPLICATE ROWS
# ------------------------------------------------------------

before_duplicates = len(df)

df = df.drop_duplicates().copy()

after_duplicates = len(df)

print("\nDuplicate rows removed:", before_duplicates - after_duplicates)
print("Dataset shape after removing duplicates:", df.shape)


# ------------------------------------------------------------
# 17. FILL MISSING AGE VALUES
# ------------------------------------------------------------

if "Age" in df.columns:
    age_median = df["Age"].median()
    df["Age"] = df["Age"].fillna(age_median)

    print("\nMissing Age values filled with median:", age_median)


# ------------------------------------------------------------
# 18. FILL MISSING EMBARKED VALUES
# ------------------------------------------------------------

if "Embarked" in df.columns:
    embarked_mode = df["Embarked"].mode()[0]
    df["Embarked"] = df["Embarked"].fillna(embarked_mode)

    print("Missing Embarked values filled with mode:", embarked_mode)


# ------------------------------------------------------------
# 19. FILL MISSING FARE VALUES IF ANY
# ------------------------------------------------------------

if "Fare" in df.columns:
    fare_median = df["Fare"].median()
    df["Fare"] = df["Fare"].fillna(fare_median)

    print("Missing Fare values filled with median:", fare_median)


# ------------------------------------------------------------
# 20. CHECK MISSING VALUES AFTER CLEANING
# ------------------------------------------------------------

print("\nMissing values after cleaning:")
print(df.isnull().sum())


# ------------------------------------------------------------
# 21. OVERALL SURVIVAL RATE
# ------------------------------------------------------------

if "Survived" in df.columns:
    overall_survival_rate = df["Survived"].mean() * 100

    print("\nOverall survival rate:")
    print(round(overall_survival_rate, 2), "%")


# ------------------------------------------------------------
# 22. SURVIVAL RATE BY GENDER
# ------------------------------------------------------------

if "Sex" in df.columns and "Survived" in df.columns:

    survival_by_gender = df.groupby("Sex")["Survived"].mean() * 100

    print("\nSurvival rate by gender:")
    print(survival_by_gender)


# ------------------------------------------------------------
# 23. SURVIVAL RATE BY PASSENGER CLASS
# ------------------------------------------------------------

if "Pclass" in df.columns and "Survived" in df.columns:

    survival_by_class = df.groupby("Pclass")["Survived"].mean() * 100

    print("\nSurvival rate by passenger class:")
    print(survival_by_class)


# ------------------------------------------------------------
# 24. HISTOGRAM OF AGE
# ------------------------------------------------------------

if "Age" in df.columns:

    plt.figure(figsize=(8, 5))

    plt.hist(df["Age"], bins=20, edgecolor="black")

    plt.title("Age Distribution of Passengers")
    plt.xlabel("Age")
    plt.ylabel("Number of Passengers")
    plt.show()


# ------------------------------------------------------------
# 25. HISTOGRAM OF FARE
# ------------------------------------------------------------

if "Fare" in df.columns:

    plt.figure(figsize=(8, 5))

    plt.hist(df["Fare"], bins=20, edgecolor="black")

    plt.title("Fare Distribution")
    plt.xlabel("Fare")
    plt.ylabel("Number of Passengers")
    plt.show()


# ------------------------------------------------------------
# 26. BOXPLOT OF AGE
# ------------------------------------------------------------

if "Age" in df.columns:

    plt.figure(figsize=(8, 5))

    sns.boxplot(x=df["Age"])

    plt.title("Boxplot of Passenger Age")
    plt.xlabel("Age")
    plt.show()


# ------------------------------------------------------------
# 27. BOXPLOT OF FARE
# ------------------------------------------------------------

if "Fare" in df.columns:

    plt.figure(figsize=(8, 5))

    sns.boxplot(x=df["Fare"])

    plt.title("Boxplot of Passenger Fare")
    plt.xlabel("Fare")
    plt.show()


# ------------------------------------------------------------
# 28. CORRELATION MATRIX
# ------------------------------------------------------------

numeric_df = df.select_dtypes(include=np.number)

correlation_matrix = numeric_df.corr()

print("\nCorrelation matrix:")
display(correlation_matrix)


# ------------------------------------------------------------
# 29. CORRELATION HEATMAP
# ------------------------------------------------------------

plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    linewidths=0.5
)

plt.title("Correlation Heatmap")
plt.show()


# ------------------------------------------------------------
# 30. BARPLOT: GENDER VS SURVIVAL
# ------------------------------------------------------------

if "Sex" in df.columns and "Survived" in df.columns:

    plt.figure(figsize=(8, 5))

    sns.barplot(
        x="Sex",
        y="Survived",
        data=df,
        errorbar=None
    )

    plt.title("Survival Rate by Gender")
    plt.xlabel("Gender")
    plt.ylabel("Survival Rate")
    plt.show()


# ------------------------------------------------------------
# 31. COUNT PLOT OF SURVIVAL
# ------------------------------------------------------------

if "Survived" in df.columns:

    plt.figure(figsize=(8, 5))

    sns.countplot(
        x="Survived",
        data=df
    )

    plt.title("Number of Survived and Non-Survived Passengers")
    plt.xlabel("Survived: 0 = No, 1 = Yes")
    plt.ylabel("Number of Passengers")
    plt.show()


# ------------------------------------------------------------
# 32. FIVE HIGHEST-PAYING PASSENGERS
# ------------------------------------------------------------

if "Fare" in df.columns:

    topFares = df.sort_values(
        by="Fare",
        ascending=False
    )

    top5 = topFares.head(5)

    print("\nFive highest-paying passengers:")

    columns_to_display = [
        column for column in ["Name", "Sex", "Pclass", "Fare"]
        if column in top5.columns
    ]

    display(top5[columns_to_display])


# ------------------------------------------------------------
# 33. GRAPH OF FIVE HIGHEST-PAYING PASSENGERS
# ------------------------------------------------------------

if "Fare" in top5.columns:

    plt.figure(figsize=(12, 6))

    if "Name" in top5.columns and "Sex" in top5.columns:

        sns.barplot(
            x="Fare",
            y="Name",
            data=top5,
            hue="Sex",
            dodge=False
        )

        plt.ylabel("Passenger Name")

    else:

        sns.barplot(
            x="Fare",
            y=top5.index.astype(str)
        )

        plt.ylabel("Passenger Index")

    plt.title("Five Highest-Paying Passengers")
    plt.xlabel("Fare")
    plt.show()


# ------------------------------------------------------------
# 34. SURVIVAL COUNT BY GENDER
# ------------------------------------------------------------

if "Sex" in df.columns and "Survived" in df.columns:

    gender_survival_table = pd.crosstab(
        df["Sex"],
        df["Survived"]
    )

    print("\nSurvival count by gender:")
    display(gender_survival_table)


# ------------------------------------------------------------
# 35. SURVIVAL COUNT BY PASSENGER CLASS
# ------------------------------------------------------------

if "Pclass" in df.columns and "Survived" in df.columns:

    class_survival_table = pd.crosstab(
        df["Pclass"],
        df["Survived"]
    )

    print("\nSurvival count by passenger class:")
    display(class_survival_table)


# ------------------------------------------------------------
# 36. PASSENGER CLASS DISTRIBUTION
# ------------------------------------------------------------

if "Pclass" in df.columns:

    plt.figure(figsize=(8, 5))

    sns.countplot(
        x="Pclass",
        data=df
    )

    plt.title("Passenger Class Distribution")
    plt.xlabel("Passenger Class")
    plt.ylabel("Number of Passengers")
    plt.show()


# ------------------------------------------------------------
# 37. AGE VS SURVIVAL
# ------------------------------------------------------------

if "Age" in df.columns and "Survived" in df.columns:

    plt.figure(figsize=(8, 5))

    sns.boxplot(
        x="Survived",
        y="Age",
        data=df
    )

    plt.title("Age Distribution by Survival")
    plt.xlabel("Survived: 0 = No, 1 = Yes")
    plt.ylabel("Age")
    plt.show()


# ------------------------------------------------------------
# 38. FARE VS PASSENGER CLASS
# ------------------------------------------------------------

if "Fare" in df.columns and "Pclass" in df.columns:

    plt.figure(figsize=(8, 5))

    sns.boxplot(
        x="Pclass",
        y="Fare",
        data=df
    )

    plt.title("Fare Distribution by Passenger Class")
    plt.xlabel("Passenger Class")
    plt.ylabel("Fare")
    plt.show()


# ------------------------------------------------------------
# 39. FINAL DATASET INFORMATION
# ------------------------------------------------------------

print("\nFinal dataset information:")
print("Final shape:", df.shape)

print("\nFinal missing values:")
print(df.isnull().sum())

print("\nFinal duplicate count:")
print(df.duplicated().sum())


# ------------------------------------------------------------
# 40. FINAL OBSERVATIONS
# ------------------------------------------------------------

print("\n================ FINAL OBSERVATIONS ================\n")

if "Survived" in df.columns:
    print("1. The dataset contains information about Titanic passengers and their survival status.")

if "Age" in df.columns:
    print("2. Missing Age values were filled using the median age.")

if "Embarked" in df.columns:
    print("3. Missing Embarked values were filled using the most frequent value.")

if "Sex" in df.columns and "Survived" in df.columns:
    print("4. Survival rates were calculated separately for male and female passengers.")

if "Pclass" in df.columns and "Survived" in df.columns:
    print("5. Survival rates were calculated for each passenger class.")

if "Fare" in df.columns:
    print("6. Fare distribution was analyzed using a histogram and boxplot.")

print("7. Correlation between numerical columns was analyzed using a heatmap.")
print("8. The five highest-paying passengers were identified.")
print("9. Graphs were created to understand important patterns in the dataset.")
print("10. Exploratory Data Analysis helped identify missing values, distributions, and relationships.")