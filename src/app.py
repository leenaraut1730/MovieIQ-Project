# ==========================================================
# 🎬 MovieIQ — Predictive Analytics on Film Success
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib

import plotly.express as px
import plotly.graph_objects as go

from streamlit_option_menu import option_menu

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)

# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="MovieIQ Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# Load Dataset
# ==========================================================

@st.cache_data
def load_data():
    return pd.read_csv("data/cleaned_movies.csv")

df = load_data()

# ==========================================================
# Load ML Model
# ==========================================================

@st.cache_resource
def load_model():
    model = joblib.load("models/random_forest.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler

model, scaler = load_model()

# ==========================================================
# Custom CSS
# ==========================================================

st.markdown("""
<style>

.stApp{
background:linear-gradient(135deg,#0f172a,#1e3a8a,#312e81);
}

/* Sidebar */

[data-testid="stSidebar"]{
background:#111827;
}

[data-testid="stSidebar"] *{
color:white;
}

/* Title */

.title{
font-size:48px;
font-weight:bold;
color:#FFD700;
text-align:center;
}

.subtitle{
font-size:20px;
color:white;
text-align:center;
margin-bottom:20px;
}

/* KPI Cards */

.card{
padding:20px;
border-radius:18px;
color:white;
text-align:center;
box-shadow:0 5px 15px rgba(0,0,0,.4);
}

.blue{
background:linear-gradient(135deg,#2193b0,#6dd5ed);
}

.green{
background:linear-gradient(135deg,#11998e,#38ef7d);
}

.orange{
background:linear-gradient(135deg,#ff8008,#ffc837);
}

.red{
background:linear-gradient(135deg,#cb2d3e,#ef473a);
}

.footer{
text-align:center;
color:white;
padding:20px;
font-size:18px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# Sidebar
# ==========================================================

with st.sidebar:

    st.image(
        "https://img.icons8.com/color/96/movie-projector.png",
        width=80
    )

    st.title("🎬 MovieIQ")

    selected = option_menu(
        menu_title=None,

        options=[
            "Home",
            "Movie Prediction",
            "Dataset Overview",
            "EDA Dashboard",
            "Model Performance",
            "Advanced Movie Analytics",
            "About Project"
        ],

        icons=[
            "house-fill",
            "camera-reels-fill",
            "table",
            "bar-chart-fill",
            "graph-up-arrow",
            "info-circle-fill"
        ],

        default_index=0,

        styles={
            "container":{
                "background-color":"#111827"
            },

            "icon":{
                "color":"gold",
                "font-size":"18px"
            },

            "nav-link":{
                "font-size":"16px",
                "border-radius":"10px",
                "margin":"6px"
            },

            "nav-link-selected":{
                "background":"linear-gradient(to right,#00C9FF,#92FE9D)",
                "color":"black",
                "font-weight":"bold"
            }
        }
    )

# ==========================================================
# HOME PAGE
# ==========================================================

if selected == "Home":

    st.markdown(
        "<div class='title'>🎬 MovieIQ Dashboard</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='subtitle'>AI Powered Movie Success Prediction System</div>",
        unsafe_allow_html=True
    )

    st.markdown("---")

    total_movies = len(df)
    success_movies = int(df["success"].sum())
    failed_movies = total_movies - success_movies
    avg_rating = round(df["vote_average"].mean(),2)

    c1,c2,c3,c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="card blue">
        <h2>🎬</h2>
        <h1>{total_movies}</h1>
        <h4>Total Movies</h4>
        </div>
        """,unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="card green">
        <h2>🏆</h2>
        <h1>{success_movies}</h1>
        <h4>Successful</h4>
        </div>
        """,unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="card red">
        <h2>❌</h2>
        <h1>{failed_movies}</h1>
        <h4>Failed</h4>
        </div>
        """,unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="card orange">
        <h2>⭐</h2>
        <h1>{avg_rating}</h1>
        <h4>Average Rating</h4>
        </div>
        """,unsafe_allow_html=True)

    st.markdown("---")

    left,right = st.columns([2,1])

    with left:

        st.subheader("📖 Project Overview")

        st.write("""
MovieIQ is an AI-powered Movie Success Prediction System built using
Python, Machine Learning, Streamlit and Plotly.

### Features

- 🤖 AI Movie Prediction
- 📊 Interactive Dashboard
- 📂 Dataset Overview
- 📈 Advanced EDA
- 🌍 World Map
- 📉 Model Performance
- 🎨 Modern UI
""")

    with right:

        fig = px.pie(
            df,
            names="success",
            hole=0.55,
            color="success",
            color_discrete_map={
                0:"#EF4444",
                1:"#10B981"
            }
        )

        fig.update_layout(template="plotly_dark")

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("---")

    st.subheader("⭐ Vote Average Distribution")

    fig = px.histogram(
        df,
        x="vote_average",
        nbins=20,
        template="plotly_dark"
    )

    fig.update_traces(marker_color="#00C9FF")

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("🔥 Top 10 Popular Movies")

    if "title" in df.columns:

        popular = df.sort_values(
            "popularity",
            ascending=False
        ).head(10)

        fig = px.bar(
            popular,
            x="popularity",
            y="title",
            orientation="h",
            color="vote_average",
            template="plotly_dark",
            color_continuous_scale="Turbo"
        )

        fig.update_layout(height=500)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("---")

    st.markdown("""
<div class='footer'>

🎬 <b>MovieIQ Dashboard</b>

<br>

Developed using ❤️ Python | Streamlit | Plotly | Machine Learning

</div>
""", unsafe_allow_html=True)
# ==========================================================
# 🎯 MOVIE PREDICTION
# ==========================================================

elif selected == "Movie Prediction":

    st.markdown(
        "<h1 style='text-align:center;color:#FFD700;'>🎯 Movie Success Prediction</h1>",
        unsafe_allow_html=True
    )

    st.write("Enter movie details to predict whether the movie will be successful.")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        budget = st.number_input(
            "💰 Budget ($)",
            min_value=0.0,
            value=5000000.0,
            step=100000.0,
            format="%.2f"
        )

        runtime = st.slider(
            "⏱ Runtime (Minutes)",
            min_value=30,
            max_value=240,
            value=120
        )

    with col2:

        popularity = st.number_input(
            "🔥 Popularity",
            min_value=0.0,
            value=50.0,
            step=1.0
        )

        vote_average = st.slider(
            "⭐ Vote Average",
            min_value=0.0,
            max_value=10.0,
            value=7.0,
            step=0.1
        )

    st.markdown("---")

    if st.button("🎬 Predict Movie Success", use_container_width=True):

        input_df = pd.DataFrame(
            {
                "budget": [budget],
                "popularity": [popularity],
                "runtime": [runtime],
                "vote_average": [vote_average],
            }
        )

        input_scaled = scaler.transform(input_df)

        prediction = model.predict(input_scaled)[0]

        probability = model.predict_proba(input_scaled)[0]

        success_prob = probability[1] * 100
        fail_prob = probability[0] * 100

        st.markdown("---")

        if prediction == 1:

            st.success(
                f"🎉 Prediction: Successful Movie\n\nConfidence: {success_prob:.2f}%"
            )

        else:

            st.error(
                f"❌ Prediction: Failed Movie\n\nConfidence: {fail_prob:.2f}%"
            )

        c1, c2 = st.columns(2)

        with c1:
            st.metric(
                "🏆 Success Probability",
                f"{success_prob:.2f}%"
            )

        with c2:
            st.metric(
                "❌ Failure Probability",
                f"{fail_prob:.2f}%"
            )

        st.markdown("---")

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=success_prob,
                title={"text": "Success Probability (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#00CC96"},
                    "steps": [
                        {"range": [0, 40], "color": "#EF4444"},
                        {"range": [40, 70], "color": "#FACC15"},
                        {"range": [70, 100], "color": "#22C55E"},
                    ],
                },
            )
        )

        gauge.update_layout(
            template="plotly_dark",
            height=420
        )

        st.plotly_chart(
            gauge,
            use_container_width=True
        )

        st.markdown("---")

        st.subheader("📋 Input Summary")

        summary = pd.DataFrame(
            {
                "Feature": [
                    "Budget",
                    "Popularity",
                    "Runtime",
                    "Vote Average",
                ],
                "Value": [
                    budget,
                    popularity,
                    runtime,
                    vote_average,
                ],
            }
        )

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )

        st.success("✅ Prediction Completed Successfully")
# ==========================================================
# 📂 DATASET OVERVIEW
# ==========================================================

elif selected == "Dataset Overview":

    st.markdown(
        "<h1 style='text-align:center;color:#FFD700;'>📂 Dataset Overview</h1>",
        unsafe_allow_html=True
    )

    st.write("Explore the movie dataset using interactive filters.")

    st.markdown("---")

    # ======================================================
    # Filters
    # ======================================================

    filtered = df.copy()

    c1, c2, c3 = st.columns(3)

    with c1:

        if "genres" in df.columns:

            genres = ["All"] + sorted(df["genres"].dropna().unique())

            genre = st.selectbox(
                "🎭 Select Genre",
                genres
            )

            if genre != "All":
                filtered = filtered[
                    filtered["genres"] == genre
                ]

    with c2:

        rating = st.slider(
            "⭐ Minimum Rating",
            0.0,
            10.0,
            0.0,
            0.1
        )

        filtered = filtered[
            filtered["vote_average"] >= rating
        ]

    with c3:

        movie = st.text_input(
            "🔍 Search Movie"
        )

        if movie and "title" in filtered.columns:

            filtered = filtered[
                filtered["title"].str.contains(
                    movie,
                    case=False,
                    na=False
                )
            ]

    st.markdown("---")

    # ======================================================
    # KPI Cards
    # ======================================================

    total = len(filtered)
    avg_rating = filtered["vote_average"].mean()
    avg_runtime = filtered["runtime"].mean()
    avg_budget = filtered["budget"].mean()

    k1, k2, k3, k4 = st.columns(4)

    k1.metric("🎬 Movies", total)
    k2.metric("⭐ Avg Rating", f"{avg_rating:.2f}")
    k3.metric("⏱ Avg Runtime", f"{avg_runtime:.0f} min")
    k4.metric("💰 Avg Budget", f"${avg_budget:,.0f}")

    st.markdown("---")

    # ======================================================
    # Dataset Preview
    # ======================================================

    st.subheader("📄 Dataset Preview")

    st.dataframe(
        filtered,
        use_container_width=True,
        height=500
    )

    st.markdown("---")

    # ======================================================
    # Dataset Information
    # ======================================================

    left, right = st.columns(2)

    with left:

        st.subheader("📋 Data Types")

        dtype_df = pd.DataFrame({
            "Column": filtered.columns,
            "Data Type": filtered.dtypes.astype(str)
        })

        st.dataframe(
            dtype_df,
            use_container_width=True
        )

    with right:

        st.subheader("❌ Missing Values")

        missing = pd.DataFrame({
            "Column": filtered.columns,
            "Missing Values": filtered.isnull().sum()
        })

        st.dataframe(
            missing,
            use_container_width=True
        )

    st.markdown("---")

    # ======================================================
    # Statistical Summary
    # ======================================================

    st.subheader("📊 Statistical Summary")

    st.dataframe(
        filtered.describe(),
        use_container_width=True
    )

    st.markdown("---")

    # ======================================================
    # Success Distribution
    # ======================================================

    st.subheader("🥧 Movie Success Distribution")

    fig = px.pie(
        filtered,
        names="success",
        hole=0.55,
        color="success",
        color_discrete_map={
            0: "#EF4444",
            1: "#22C55E"
        },
        template="plotly_dark"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    # ======================================================
    # Correlation Heatmap
    # ======================================================

    st.subheader("🌡 Correlation Heatmap")

    corr = filtered.select_dtypes(
        include="number"
    ).corr()

    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu"
    )

    fig.update_layout(
        template="plotly_dark",
        height=650
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    # ======================================================
    # Budget Distribution
    # ======================================================

    st.subheader("💰 Budget Distribution")

    fig = px.histogram(
        filtered,
        x="budget",
        nbins=30,
        template="plotly_dark",
        color_discrete_sequence=["#00C9FF"]
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    # ======================================================
    # Rating Distribution
    # ======================================================

    st.subheader("⭐ Vote Average Distribution")

    fig = px.histogram(
        filtered,
        x="vote_average",
        nbins=20,
        template="plotly_dark",
        color_discrete_sequence=["#FFD700"]
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    # ======================================================
    # Download Dataset
    # ======================================================

    csv = filtered.to_csv(index=False)

    st.download_button(
        label="📥 Download Filtered Dataset",
        data=csv,
        file_name="filtered_movies.csv",
        mime="text/csv"
    )

    st.success("✅ Dataset Overview Loaded Successfully")
# ==========================================================
# 📊 EDA DASHBOARD
# ==========================================================

elif selected == "EDA Dashboard":

    st.markdown(
        "<h1 style='text-align:center;color:#FFD700;'>📊 Advanced EDA Dashboard</h1>",
        unsafe_allow_html=True
    )

    st.write("Interactive Exploratory Data Analysis")

    st.markdown("---")

    filtered = df.copy()

    # ======================================================
    # Filters
    # ======================================================

    c1, c2, c3 = st.columns(3)

    with c1:

        if "genres" in filtered.columns:

            genre = st.selectbox(
                "🎭 Genre",
                ["All"] + sorted(filtered["genres"].dropna().unique())
            )

            if genre != "All":
                filtered = filtered[
                    filtered["genres"] == genre
                ]

    with c2:

        rating = st.slider(
            "⭐ Minimum Rating",
            0.0,
            10.0,
            0.0,
            0.1
        )

        filtered = filtered[
            filtered["vote_average"] >= rating
        ]

    with c3:

        runtime = st.slider(
            "⏱ Runtime",
            int(filtered["runtime"].min()),
            int(filtered["runtime"].max()),
            (
                int(filtered["runtime"].min()),
                int(filtered["runtime"].max())
            )
        )

        filtered = filtered[
            (filtered["runtime"] >= runtime[0]) &
            (filtered["runtime"] <= runtime[1])
        ]

    st.markdown("---")

    # ======================================================
    # KPI Cards
    # ======================================================

    total = len(filtered)
    avg_rating = filtered["vote_average"].mean()
    avg_runtime = filtered["runtime"].mean()
    avg_budget = filtered["budget"].mean()

    k1, k2, k3, k4 = st.columns(4)

    k1.metric("🎬 Movies", total)
    k2.metric("⭐ Rating", f"{avg_rating:.2f}")
    k3.metric("⏱ Runtime", f"{avg_runtime:.0f} min")
    k4.metric("💰 Budget", f"${avg_budget:,.0f}")

    st.markdown("---")

    # ======================================================
    # Budget Distribution
    # ======================================================

    st.subheader("💰 Budget Distribution")

    fig = px.histogram(
        filtered,
        x="budget",
        nbins=30,
        template="plotly_dark",
        color_discrete_sequence=["#00C9FF"]
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ======================================================
    # Runtime Distribution
    # ======================================================

    st.subheader("⏱ Runtime Distribution")

    fig = px.histogram(
        filtered,
        x="runtime",
        nbins=25,
        template="plotly_dark",
        color_discrete_sequence=["#22C55E"]
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ======================================================
    # Vote Average Distribution
    # ======================================================

    st.subheader("⭐ Vote Average Distribution")

    fig = px.histogram(
        filtered,
        x="vote_average",
        nbins=20,
        template="plotly_dark",
        color_discrete_sequence=["#F59E0B"]
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ======================================================
    # Popularity Distribution
    # ======================================================

    st.subheader("🔥 Popularity Distribution")

    fig = px.histogram(
        filtered,
        x="popularity",
        nbins=30,
        template="plotly_dark",
        color_discrete_sequence=["#EC4899"]
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ======================================================
    # Budget vs Popularity
    # ======================================================

    st.subheader("📈 Budget vs Popularity")

    fig = px.scatter(
        filtered,
        x="budget",
        y="popularity",
        color="vote_average",
        size="runtime",
        hover_name="title" if "title" in filtered.columns else None,
        template="plotly_dark",
        color_continuous_scale="Turbo"
    )

    fig.update_layout(height=600)

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ======================================================
    # Success Distribution
    # ======================================================

    st.subheader("🥧 Movie Success Distribution")

    success_df = (
        filtered["success"]
        .value_counts()
        .reset_index()
    )

    success_df.columns = ["Status", "Count"]

    success_df["Status"] = success_df["Status"].replace(
        {
            1: "Successful",
            0: "Failed"
        }
    )

    fig = px.pie(
        success_df,
        names="Status",
        values="Count",
        hole=0.55,
        template="plotly_dark",
        color="Status",
        color_discrete_map={
            "Successful": "#22C55E",
            "Failed": "#EF4444"
        }
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ======================================================
    # Genre Analysis
    # ======================================================

    if "genres" in filtered.columns:

        st.subheader("🎭 Top Genres")

        genre_df = (
            filtered["genres"]
            .value_counts()
            .head(10)
            .reset_index()
        )

        genre_df.columns = ["Genre", "Movies"]

        fig = px.bar(
            genre_df,
            x="Genre",
            y="Movies",
            color="Movies",
            text="Movies",
            template="plotly_dark",
            color_continuous_scale="Turbo"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.success("✅ Advanced EDA Dashboard Loaded Successfully")

# ==========================================================
# 🌍 ADVANCED MOVIE ANALYTICS
# ==========================================================

elif selected == "Advanced Movie Analytics":

    st.markdown(
        "<h1 style='text-align:center;color:#FFD700;'>🌍 Advanced Movie Analytics</h1>",
        unsafe_allow_html=True
    )

    filtered = df.copy()

    st.markdown("---")

    # ======================================================
    # Revenue Distribution
    # ======================================================

    if "revenue" in filtered.columns:

        st.subheader("💰 Revenue Distribution")

        fig = px.histogram(
            filtered,
            x="revenue",
            nbins=30,
            template="plotly_dark",
            color_discrete_sequence=["#00E5FF"]
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ======================================================
    # Budget vs Revenue
    # ======================================================

    if "revenue" in filtered.columns:

        st.subheader("💸 Budget vs Revenue")

        fig = px.scatter(
            filtered,
            x="budget",
            y="revenue",
            color="vote_average",
            size="popularity",
            hover_name="title" if "title" in filtered.columns else None,
            template="plotly_dark",
            color_continuous_scale="Turbo"
        )

        fig.update_layout(height=600)

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ======================================================
    # Movies Released Per Year
    # ======================================================

    if "release_date" in filtered.columns:

        st.subheader("📅 Movies Released Per Year")

        year_df = filtered.copy()

        year_df["release_date"] = pd.to_datetime(
            year_df["release_date"],
            errors="coerce"
        )

        year_df["Year"] = year_df["release_date"].dt.year

        yearly = (
            year_df.groupby("Year")
            .size()
            .reset_index(name="Movies")
        )

        fig = px.line(
            yearly,
            x="Year",
            y="Movies",
            markers=True,
            template="plotly_dark"
        )

        fig.update_layout(height=550)

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ======================================================
    # Runtime by Genre
    # ======================================================

    if "genres" in filtered.columns:

        st.subheader("🎭 Average Runtime by Genre")

        runtime_df = (
            filtered.groupby("genres")["runtime"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )

        fig = px.bar(
            runtime_df,
            x="genres",
            y="runtime",
            color="runtime",
            template="plotly_dark",
            color_continuous_scale="Viridis",
            text_auto=".1f"
        )

        fig.update_layout(height=550)

        st.plotly_chart(fig, use_container_width=True)
        
    # ======================================================
    # 🌍 World Map
    # ======================================================

    st.subheader("🌍 Movies Around the World")

    country_df = pd.DataFrame({
        "Country": [
            "India",
            "United States",
            "United Kingdom",
            "Canada",
            "Australia",
            "France",
            "Germany",
            "Japan",
            "China",
            "Brazil",
            "Italy",
            "Spain",
            "South Korea",
            "Mexico",
            "Russia",
            "South Africa",
            "Argentina",
            "Turkey",
            "Indonesia",
            "Thailand"
        ],

        "Movies": [
            320,
            450,
            180,
            95,
            75,
            120,
            110,
            140,
            160,
            85,
            70,
            65,
            90,
            80,
            60,
            45,
            40,
            55,
            50,
            48
        ]
    })

    fig = px.choropleth(
        country_df,
        locations="Country",
        locationmode="country names",
        color="Movies",
        hover_name="Country",
        color_continuous_scale="Turbo",
        template="plotly_dark",
        title="Movies Produced by Country"
    )

    fig.update_geos(
        showcountries=True,
        showcoastlines=True,
        showland=True,
        showocean=True,
        oceancolor="lightblue",
        fitbounds="locations"
    )

    fig.update_layout(
        height=700,
        margin=dict(l=0, r=0, t=50, b=0)
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ======================================================
    # 🏆 Top Rated Movies
    # ======================================================

    if "title" in filtered.columns:

        st.subheader("🏆 Top 10 Rated Movies")

        top_movies = (
            filtered.sort_values(
                "vote_average",
                ascending=False
            )
            .head(10)
        )

        fig = px.bar(
            top_movies,
            x="vote_average",
            y="title",
            orientation="h",
            color="vote_average",
            template="plotly_dark",
            color_continuous_scale="Plasma",
            text="vote_average"
        )

        fig.update_layout(height=550)

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ======================================================
    # 🔥 Top Popular Movies
    # ======================================================

    if "title" in filtered.columns:

        st.subheader("🔥 Top 10 Popular Movies")

        popular = (
            filtered.sort_values(
                "popularity",
                ascending=False
            )
            .head(10)
        )

        fig = px.bar(
            popular,
            x="popularity",
            y="title",
            orientation="h",
            color="popularity",
            template="plotly_dark",
            color_continuous_scale="Sunset",
            text="popularity"
        )

        fig.update_layout(height=550)

        st.plotly_chart(fig, use_container_width=True)

    st.success("✅ Advanced Movie Analytics Loaded Successfully")
# ==========================================================
# 📈 MODEL PERFORMANCE
# ==========================================================

if selected == "Model Performance":
    st.title("📈 Model Performance")

    X = df[["budget","popularity","runtime","vote_average"]]
    y = df["success"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    X_test = scaler.transform(X_test)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    acc = accuracy_score(y_test,y_pred)
    pre = precision_score(y_test,y_pred)
    rec = recall_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred)

    c1,c2,c3,c4=st.columns(4)

    c1.metric("Accuracy",f"{acc:.2%}")
    c2.metric("Precision",f"{pre:.2%}")
    c3.metric("Recall",f"{rec:.2%}")
    c4.metric("F1 Score",f"{f1:.2%}")

    st.markdown("---")

    cm=confusion_matrix(y_test,y_pred)

    fig=px.imshow(
        cm,
        text_auto=True,
        color_continuous_scale="Blues",
        x=["Failed","Success"],
        y=["Failed","Success"]
    )

    fig.update_layout(
        template="plotly_dark",
        title="Confusion Matrix"
    )

    st.plotly_chart(fig,width="stretch")

    fpr,tpr,_=roc_curve(y_test,y_prob)

    roc_auc=auc(fpr,tpr)

    fig=go.Figure()

    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode="lines",
        name=f"AUC={roc_auc:.2f}"
    ))

    fig.add_trace(go.Scatter(
        x=[0,1],
        y=[0,1],
        mode="lines",
        line=dict(dash="dash")
    ))

    fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_dark"
    )

    st.plotly_chart(fig,width="stretch")

    st.text(classification_report(y_test,y_pred))

# ==========================================================
# ℹ️ ABOUT PROJECT
# ==========================================================

elif selected == "About Project":
    st.markdown("""
    <h1 style='text-align:center;color:#FFD700;'>
    🎬 About MovieIQ
    </h1>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div style="
        background:linear-gradient(135deg,#1E3A8A,#6D28D9);
        padding:30px;
        border-radius:20px;
        color:white;
        text-align:center;
    ">
        <h2>🎬 MovieIQ Dashboard</h2>
        <h4>AI Powered Movie Success Prediction System</h4>
        <p>
        Predict whether a movie will become successful using Machine Learning
        and visualize insights through an interactive Streamlit dashboard.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # =====================================================
    # Problem Statement
    # =====================================================

    st.header("🎯 Problem Statement")

    st.info("""
    1. Movie production involves huge investments, but many movies fail at the box office.

    2. The objective of this project is to build an AI-powered prediction system that
    can estimate whether a movie will become successful based on features like
    Budget, Popularity, Runtime and Vote Average.

    3. This helps production companies make better business decisions before releasing a movie.
""")

    st.markdown("---")

    # =====================================================
    # Business Objective
    # =====================================================

    st.header("💼 Business Objective")

    col1,col2=st.columns(2)

    with col1:
        st.success("""
✅ Predict Movie Success

✅ Reduce Financial Risk

✅ Improve Investment Decisions

✅ Understand Audience Trends
""")

    with col2:
        st.info("""
✅ Analyze Movie Dataset

✅ Identify Success Factors

✅ Support Business Plannin
✅ Improve ROI
""")

    st.markdown("---")

    # =====================================================
    # Key Features
    # =====================================================

    st.header("✨ Key Features")

    left,right=st.columns(2)

    with left:
        st.success("""
✔ Movie Success Prediction

✔ Dataset Overview

✔ Interactive Dashboard

✔ Advanced EDA

✔ World Map

✔ Download Dataset
""")

    with right:
        st.info("""
✔ Model Performance

✔ Confusion Matrix

✔ ROC Curve

✔ Classification Report

✔ Responsive UI

✔ Business Insights
""")

    st.markdown("---")

    # =====================================================
    # Business Insights
    # =====================================================

    st.header("📊 Business Insights")

    st.write("""
• High budget movies generally achieve higher revenue.

• Popular movies tend to receive better audience ratings.

• Runtime has only a moderate impact on movie success.

• Vote Average is one of the strongest indicators of success.

• Random Forest model identifies important features influencing prediction.
""")

    st.markdown("---")

    # =====================================================
    # Project Structure
    # =====================================================

    st.header("📁 Project Structure")

    st.code("""
MovieIQ/
│
├── app.py
├── data/
│   └── cleaned_movies.csv
├── models/
│   ├── random_forest.pkl
│   └── scaler.pkl
├── requirements.txt
└── README.md
""")

    st.markdown("---")

    # =====================================================
    # Future Scope
    # =====================================================

    st.header("🚀 Future Scope")

    st.warning("""
✅ Deep Learning Model

✅ Recommendation System

✅ TMDB API Integration

✅ Cloud Deployment

✅ Mobile Application

✅ Real-Time Prediction

✅ AI Chatbot Integration
""")

    st.markdown("---")

    # =====================================================
    # Footer
    # =====================================================

    st.markdown("""
<div style="
background:linear-gradient(90deg,#111827,#1E3A8A,#312E81);
padding:20px;
border-radius:15px;
text-align:center;
color:white;
">

<h2>🎬 MovieIQ Dashboard</h2>

<h4>AI Powered Movie Success Prediction System</h4>

<p>Developed using ❤️ Python | Streamlit | Plotly | Scikit-Learn</p>

<p>© 2026 MovieIQ</p>

</div>
""", unsafe_allow_html=True)