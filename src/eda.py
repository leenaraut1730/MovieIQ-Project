import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import plotly.express as px


# ==========================
# Load Dataset
# ==========================

df = pd.read_csv(
    "data/cleaned_movies.csv"
)


# Create folder

os.makedirs(
    "data/assets",
    exist_ok=True
)


# ==========================
# Style
# ==========================

sns.set_theme(
    style="darkgrid"
)



# ==========================
# 1. Budget vs Revenue
# ==========================


plt.figure(figsize=(10,6))


sns.scatterplot(

    data=df,

    x="budget",

    y="revenue",

    hue="success",

    palette="viridis",

    size="popularity",

    sizes=(20,200)

)


plt.title(
    "Budget vs Revenue Analysis",
    fontsize=18
)


plt.xlabel(
    "Movie Budget"
)


plt.ylabel(
    "Revenue"
)


plt.savefig(
"data/assets/budget_vs_revenue.png",
bbox_inches="tight"
)


plt.close()





# ==========================
# 2. Runtime Distribution
# ==========================


plt.figure(figsize=(10,6))


sns.histplot(

data=df,

x="runtime",

kde=True,

color="purple"

)


plt.title(
"Movie Runtime Distribution",
fontsize=18
)


plt.savefig(
"data/assets/runtime_distribution.png",
bbox_inches="tight"
)


plt.close()





# ==========================
# 3. Popularity Distribution
# ==========================


plt.figure(figsize=(10,6))


sns.histplot(

data=df,

x="popularity",

bins=40,

color="orange"

)


plt.title(
"Movie Popularity Distribution",
fontsize=18
)


plt.savefig(
"data/assets/popularity_distribution.png",
bbox_inches="tight"
)


plt.close()





# ==========================
# 4. Vote Average
# ==========================


plt.figure(figsize=(10,6))


sns.histplot(

data=df,

x="vote_average",

kde=True,

color="green"

)


plt.title(
"Movie Rating Distribution",
fontsize=18
)


plt.savefig(
"data/assets/vote_average_distribution.png",
bbox_inches="tight"
)


plt.close()





# ==========================
# 5. Success Distribution
# ==========================


plt.figure(figsize=(8,5))


sns.countplot(

data=df,

x="success",

palette="cool"

)


plt.title(
"Successful vs Failed Movies",
fontsize=18
)


plt.savefig(
"data/assets/success_distribution.png",
bbox_inches="tight"
)


plt.close()





# ==========================
# 6. Genre Analysis
# ==========================


if "genre" in df.columns:


    genre_count=(

        df["genre"]

        .value_counts()

        .head(10)

        .reset_index()

    )


    genre_count.columns=[
        "Genre",
        "Count"
    ]


    fig=px.bar(

        genre_count,

        x="Genre",

        y="Count",

        color="Count",

        title="Top 10 Movie Genres",

        color_continuous_scale="Turbo"

    )


    fig.write_image(

    "data/assets/genre_analysis.png"

    )



print(
"EDA Charts Created Successfully 🎨"
)