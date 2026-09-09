"""
Mobile Reviews Explorer
------------------------
Streamlit app covering the full pipeline: EDA, price segmentation (K-Means),
a content-based recommender, and a dedicated Insights & Reporting view.

Run with:
    streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Mobile Reviews Explorer", layout="wide")

DEFAULT_DATA_PATH = r"E:\SQL\Mobile Reviews\Mobile Reviews_cleaned.csv"

REQUIRED_COLS = [
    "brand", "model", "rating", "sentiment", "review_date", "age",
    "verified_purchase", "price_usd", "battery_life_rating",
    "camera_rating", "performance_rating", "design_rating",
    "display_rating", "review_id", "country",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_data(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df = df.drop_duplicates().reset_index(drop=True)
    if "review_date" in df.columns:
        df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
    return df


@st.cache_data
def build_product_table(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["positive_sentiment"] = d["sentiment"] == "Positive"
    product_mean = d.groupby(["brand", "model"]).agg(
        {
            "price_usd": "median",
            "rating": "mean",
            "battery_life_rating": "mean",
            "camera_rating": "mean",
            "performance_rating": "mean",
            "design_rating": "mean",
            "display_rating": "mean",
            "verified_purchase": "mean",
            "review_id": "count",
            "positive_sentiment": "mean",
        }
    ).reset_index()
    product_mean = product_mean.rename(columns={"review_id": "review_counts"})
    return product_mean


@st.cache_data
def segment_by_price(product_mean: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    pm = product_mean.copy()
    n_samples = len(pm)
    if n_samples == 0:
        st.warning("No products available for segmentation.")
        return pm
    k = min(k, n_samples)
    scaler = StandardScaler()
    price_scaled = scaler.fit_transform(pm[["price_usd"]])

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    pm["cluster"] = kmeans.fit_predict(price_scaled)

    # Cluster IDs from KMeans are arbitrary - sort by mean price before labeling
    # so "cluster 0" isn't assumed to be the cheapest.
    ordered_clusters = pm.groupby("cluster")["price_usd"].mean().sort_values().index
    if k == 3:
        names = ["Budget", "Mid-range", "Premium"]
    else:
        names = [f"Segment {i+1}" for i in range(k)]
    mapping = {cid: name for cid, name in zip(ordered_clusters, names)}
    pm["segment"] = pm["cluster"].map(mapping)
    return pm


@st.cache_data
def build_similarity(product_mean: pd.DataFrame):
    brand_enc = pd.get_dummies(product_mean["brand"], prefix="brand").astype(int)

    num_cols = [
        "price_usd", "rating", "battery_life_rating", "camera_rating",
        "performance_rating", "design_rating", "display_rating",
        "positive_sentiment", "verified_purchase",
    ]
    num_features = product_mean[num_cols]
    scaler = StandardScaler()
    num_scaled = pd.DataFrame(
        scaler.fit_transform(num_features),
        columns=num_features.columns,
        index=product_mean.index,
    )

    rec_features = pd.concat([num_scaled, brand_enc], axis=1)
    sim_matrix = cosine_similarity(rec_features)
    sim_df = pd.DataFrame(
        sim_matrix, index=product_mean["model"], columns=product_mean["model"]
    )
    return sim_df


def recommend_similar(sim_df: pd.DataFrame, model_name: str, top_n: int = 5) -> pd.Series:
    scores = sim_df.loc[model_name].sort_values(ascending=False)
    scores = scores.drop(model_name, errors="ignore")
    scores = scores[~scores.index.duplicated(keep="first")]
    return scores.head(top_n)


# ---------------------------------------------------------------------------
# Sidebar - data input
# ---------------------------------------------------------------------------
st.sidebar.title("Mobile Reviews Explorer")

uploaded = st.sidebar.file_uploader(
    "Upload a different Mobile Reviews CSV (optional)", type=["csv"]
)

if uploaded is not None:
    df = load_data(uploaded)
elif os.path.exists(DEFAULT_DATA_PATH):
    df = load_data(DEFAULT_DATA_PATH)
    st.sidebar.caption(f"Using bundled dataset: {DEFAULT_DATA_PATH}")
else:
    st.title("Mobile Reviews Explorer")
    st.info(
        "No bundled dataset found. Upload a CSV in the sidebar to get started. "
        f"Expected columns include: {', '.join(REQUIRED_COLS)}."
    )
    st.stop()

missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
if missing_cols:
    st.error(f"The dataset is missing expected columns: {missing_cols}")
    st.stop()

brands = sorted(df["brand"].dropna().unique().tolist())
selected_brands = st.sidebar.multiselect("Filter by brand", brands, default=brands)
df = df[df["brand"].isin(selected_brands)]

if df.empty:
    st.warning("Please select at least one brand from the sidebar.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.caption(f"{len(df):,} reviews loaded across {df['model'].nunique()} models")

tabs = st.tabs([
    "Overview",
    "Ratings & Sentiment",
    "Price Segmentation & Clusters",
    "Recommender",
    "Insights & Reporting",
])

# ---------------------------------------------------------------------------
# Tab 1: Overview
# ---------------------------------------------------------------------------
with tabs[0]:
    st.header("Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reviews", f"{len(df):,}")
    c2.metric("Models", df["model"].nunique())
    c3.metric("Brands", df["brand"].nunique())
    c4.metric("Avg. rating", f"{df['rating'].mean():.2f}")

    st.subheader("Sample data")
    st.dataframe(df.head(20), use_container_width=True)

    with st.expander("Missing values"):
        missing = df.isna().sum()
        missing = missing[missing > 0]
        if missing.empty:
            st.success("No missing values.")
        else:
            st.dataframe(missing.rename("missing_count"))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Reviews by country")
        country_counts = df["country"].value_counts().reset_index()
        country_counts.columns = ["country", "reviews"]
        fig = px.bar(country_counts, x="country", y="reviews", title="Reviews by Country")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        if df["review_date"].notna().any():
            st.subheader("Reviews over time")
            trend = df.set_index("review_date").resample("ME").size().reset_index(name="reviews")
            fig = px.line(trend, x="review_date", y="reviews", title="Reviews Over Time")
            st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 2: Ratings & Sentiment
# ---------------------------------------------------------------------------
with tabs[1]:
    st.header("Ratings & Sentiment")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Rating distribution")
        fig = px.histogram(df, x="rating", nbins=5, title="Rating Distribution")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Sentiment distribution")
        sentiment_counts = df["sentiment"].value_counts().reset_index()
        sentiment_counts.columns = ["sentiment", "count"]
        fig = px.pie(sentiment_counts, names="sentiment", values="count", title="Sentiment Share")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Rating by brand")
    rating_brand = df.groupby(["brand", "rating"]).size().reset_index(name="count")
    fig = px.bar(
        rating_brand, x="brand", y="count", color="rating", title="Rating Distribution by Brand",
        barmode="stack",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sentiment (%) by brand")
    sent_pct = (
        pd.crosstab(df["brand"], df["sentiment"], normalize="index").mul(100).reset_index()
        .melt(id_vars="brand", var_name="sentiment", value_name="percent")
    )
    fig = px.bar(sent_pct, x="brand", y="percent", color="sentiment", title="Sentiment (%) by Brand", barmode="stack")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Rating vs. sentiment (%)")
    st.dataframe(
        pd.crosstab(df["rating"], df["sentiment"], normalize="index").mul(100).round(1)
    )

    st.subheader("Correlation between ratings & price")
    corr_cols = [
        "price_usd", "rating", "battery_life_rating", "camera_rating",
        "performance_rating", "design_rating", "display_rating", "helpful_votes",
    ]
    corr_cols = [c for c in corr_cols if c in df.columns]
    corr = df[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(corr, annot=True, cmap="YlGnBu", ax=ax)
    st.pyplot(fig)

# ---------------------------------------------------------------------------
# Tab 3: Price segmentation & clusters
# ---------------------------------------------------------------------------
with tabs[2]:
    st.header("Price Segmentation & Cluster Insights")

    product_mean = build_product_table(df)
    k = st.slider("Number of price segments (k)", min_value=2, max_value=6, value=3)
    product_seg = segment_by_price(product_mean, k=k)

    st.subheader("Cluster / segment summary")
    segment_profile = product_seg.groupby("segment").agg(
        models=("model", "count"),
        avg_price=("price_usd", "mean"),
        avg_rating=("rating", "mean"),
        avg_positive_sentiment=("positive_sentiment", "mean"),
    ).round(2)
    st.dataframe(segment_profile, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Products by segment (price)")
        fig = px.strip(
            product_seg, x="price_usd", y="segment", color="segment",
            title="Product Segments by Price", stripmode="overlay",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Average rating by segment")
        fig = px.bar(
            segment_profile.reset_index(), x="segment", y="avg_rating",
            color="segment", title="Average Rating per Segment",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("All products with segment labels")
    st.dataframe(
        product_seg[["brand", "model", "price_usd", "rating", "segment"]].sort_values("price_usd"),
        use_container_width=True,
    )

# ---------------------------------------------------------------------------
# Tab 4: Recommender
# ---------------------------------------------------------------------------
with tabs[3]:
    st.header("Phone Recommender")
    st.caption("Content-based recommendations using cosine similarity over ratings, price, sentiment, and brand.")

    product_mean = build_product_table(df)
    sim_df = build_similarity(product_mean)

    model_choice = st.selectbox("Pick a phone model", sorted(product_mean["model"].unique()))
    top_n = st.slider("Number of recommendations", 1, 10, 5)

    recs = recommend_similar(sim_df, model_choice, top_n=top_n)
    rec_table = product_mean.set_index("model").loc[recs.index, ["brand", "price_usd", "rating"]]
    rec_table["similarity"] = recs.values.round(3)
    st.subheader(f"Phones similar to {model_choice}")
    st.dataframe(rec_table.reset_index(), use_container_width=True)

    fig = px.bar(
        rec_table.reset_index(), x="model", y="similarity", color="brand",
        title=f"Similarity Score vs. {model_choice}",
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 5: Insights & Reporting
# ---------------------------------------------------------------------------
with tabs[4]:
    st.header("Insights & Reporting")
    st.caption("Data-driven summary covering segmentation, top/bottom performers, price-vs-performance, and customer preference.")

    product_mean = build_product_table(df)
    product_seg = segment_by_price(product_mean, k=3)

    # --- 1. Product segmentation (cluster-wise analysis) ---
    st.subheader("1. Product Segmentation (Cluster-wise)")
    segment_profile = product_seg.groupby("segment").agg(
        models=("model", "count"),
        avg_price=("price_usd", "mean"),
        avg_rating=("rating", "mean"),
    ).round(2)
    st.dataframe(segment_profile, use_container_width=True)
    best_segment = segment_profile["avg_rating"].idxmax()
    st.markdown(f"**Takeaway:** the **{best_segment}** segment has the highest average rating "
                f"({segment_profile.loc[best_segment, 'avg_rating']:.2f}) among the price tiers.")

    # --- 2. High/low performing products ---
    st.subheader("2. High-Performing and Low-Performing Products")
    col1, col2 = st.columns(2)
    model_rating = df.groupby("model")["rating"].mean().sort_values(ascending=False)
    with col1:
        st.markdown("**Top 10 rated models**")
        st.dataframe(model_rating.head(10).round(2).rename("avg_rating"))
    with col2:
        st.markdown("**Bottom 10 rated models**")
        st.dataframe(model_rating.tail(10).sort_values().round(2).rename("avg_rating"))

    # --- 3. Price vs performance trends ---
    st.subheader("3. Price vs. Performance Trends")
    fig = px.scatter(
        product_mean, x="price_usd", y="rating", color="brand", hover_name="model",
        title="Price vs. Average Rating (per model)",
    )
    # simple linear trend line via numpy, no extra dependency required
    x = product_mean["price_usd"].values
    y = product_mean["rating"].values
    if len(x) > 1:
        coeffs = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = np.polyval(coeffs, x_line)
        fig.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines", name="Trend", line=dict(dash="dash")))
    st.plotly_chart(fig, use_container_width=True)

    price_rating_corr = product_mean["price_usd"].corr(product_mean["rating"])
    st.markdown(f"**Takeaway:** price and average rating have a correlation of **{price_rating_corr:.2f}** "
                "across models — " + (
                    "a meaningfully positive relationship, i.e. pricier phones tend to rate higher."
                    if price_rating_corr > 0.3 else
                    "a meaningfully negative relationship, i.e. pricier phones tend to rate lower."
                    if price_rating_corr < -0.3 else
                    "a weak relationship — price alone doesn't explain rating differences well."
                ))

    # --- 4. Customer preference patterns ---
    st.subheader("4. Customer Preference Patterns")
    col1, col2, col3 = st.columns(3)
    with col1:
        fig = px.pie(df, names="sentiment", title="Overall Sentiment Share")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        verified_pct = df["verified_purchase"].mean() * 100
        st.metric("Verified purchase rate", f"{verified_pct:.1f}%")
        fig = px.histogram(df, x="age", nbins=20, title="Reviewer Age Distribution")
        st.plotly_chart(fig, use_container_width=True)
    with col3:
        if "source" in df.columns:
            source_counts = df["source"].value_counts().reset_index()
            source_counts.columns = ["source", "reviews"]
            fig = px.bar(source_counts, x="source", y="reviews", title="Reviews by Source")
            st.plotly_chart(fig, use_container_width=True)

    # --- 5. Data-driven summary ---
    st.subheader("5. Summary for Decision-Making")
    top_model = model_rating.idxmax()
    bottom_model = model_rating.idxmin()
    top_country = df["country"].value_counts().idxmax()
    positive_pct = (df["sentiment"] == "Positive").mean() * 100

    st.markdown(f"""
- **Best-reviewed model:** `{top_model}` ({model_rating.max():.2f} avg rating) — candidate to feature or restock.
- **Weakest-reviewed model:** `{bottom_model}` ({model_rating.min():.2f} avg rating) — candidate for quality review or phase-out.
- **Strongest price segment:** `{best_segment}` — highest average satisfaction for its price tier.
- **Price-performance relationship:** r = {price_rating_corr:.2f} — {"higher price tends to track with higher satisfaction" if price_rating_corr > 0.15 else "price is not a reliable proxy for satisfaction" if abs(price_rating_corr) <= 0.15 else "higher price tends to track with lower satisfaction"}.
- **Customer sentiment:** {positive_pct:.1f}% of reviews are positive; {verified_pct:.1f}% come from verified purchases.
- **Largest market by volume:** `{top_country}` — {df['country'].value_counts().max():,} reviews.
""")
