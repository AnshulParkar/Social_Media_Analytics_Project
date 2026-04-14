"""
War Prediction — Real-Time Content Analysis Dashboard
Streamlit app that reads CSV data collected by the notebook and auto-refreshes.
Run with: streamlit run exp_04/app.py
"""

import os
import re
import time
import sys
from datetime import datetime
from collections import Counter
from pathlib import Path

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wordcloud import WordCloud

import streamlit as st
from streamlit_autorefresh import st_autorefresh

import spacy
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

import folium
from streamlit_folium import st_folium

from sklearn.feature_extraction.text import TfidfVectorizer

import feedparser
import requests
from dotenv import load_dotenv

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="War Content Analysis", page_icon="🌍", layout="wide")

# Auto-refresh every 20 seconds
count = st_autorefresh(interval=20_000, limit=None, key="war_dashboard_refresh")

# ─── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

YOUTUBE_CSV = os.path.join(SCRIPT_DIR, "youtube_data.csv")
NEWS_CSV = os.path.join(SCRIPT_DIR, "news_data.csv")
RSS_CSV = os.path.join(SCRIPT_DIR, "rss_data.csv")
LOCATION_CSV = os.path.join(SCRIPT_DIR, "location_data.csv")

# Load env
load_dotenv(os.path.join(ROOT_DIR, ".env"))
YOUTUBE_API = os.getenv("YOUTUBE_API")
NEWS_API_KEY = os.getenv("NEWSAPI")

# Load spaCy (cached)
@st.cache_resource
def load_nlp():
    return spacy.load("en_core_web_sm")

nlp = load_nlp()

WAR_KEYWORDS = [
    "Ukraine war", "Gaza conflict", "Sudan war",
    "Israel Hamas", "Russia Ukraine", "Yemen Houthi",
    "Syria conflict", "Myanmar civil war",
]

# ─── Data Loading ────────────────────────────────────────────────────────────

@st.cache_data(ttl=15)
def load_csv(path):
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def load_all_data():
    return load_csv(YOUTUBE_CSV), load_csv(NEWS_CSV), load_csv(RSS_CSV), load_csv(LOCATION_CSV)


# ─── Data Collection Functions (for live refresh) ───────────────────────────

def collect_youtube_live(keywords, max_results=5):
    """Collect fresh YouTube data."""
    if not YOUTUBE_API:
        return pd.DataFrame()
    try:
        from googleapiclient.discovery import build
        youtube = build("youtube", "v3", developerKey=YOUTUBE_API)
        all_videos = []
        for keyword in keywords:
            try:
                search_resp = youtube.search().list(
                    q=keyword, part="id,snippet", type="video",
                    maxResults=max_results, order="date", relevanceLanguage="en"
                ).execute()
                video_ids = [item["id"]["videoId"] for item in search_resp.get("items", [])]
                if not video_ids:
                    continue
                stats_resp = youtube.videos().list(
                    part="snippet,statistics", id=",".join(video_ids)
                ).execute()
                for item in stats_resp.get("items", []):
                    snippet = item["snippet"]
                    stats = item.get("statistics", {})
                    all_videos.append({
                        "video_id": item["id"],
                        "title": snippet.get("title", ""),
                        "description": snippet.get("description", ""),
                        "channel": snippet.get("channelTitle", ""),
                        "published_at": snippet.get("publishedAt", ""),
                        "tags": "|".join(snippet.get("tags", [])),
                        "thumbnail_url": snippet.get("thumbnails", {}).get("high", {}).get("url", ""),
                        "view_count": int(stats.get("viewCount", 0)),
                        "like_count": int(stats.get("likeCount", 0)),
                        "comment_count": int(stats.get("commentCount", 0)),
                        "search_keyword": keyword,
                        "collected_at": datetime.now().isoformat(),
                    })
            except Exception:
                continue
        return pd.DataFrame(all_videos)
    except Exception:
        return pd.DataFrame()


def collect_news_live(keywords):
    """Collect fresh news data."""
    if not NEWS_API_KEY:
        return pd.DataFrame()
    try:
        from newsapi import NewsApiClient
        newsapi = NewsApiClient(api_key=NEWS_API_KEY)
        all_articles = []
        for keyword in keywords:
            try:
                resp = newsapi.get_everything(q=keyword, language="en", sort_by="publishedAt", page_size=10)
                for article in resp.get("articles", []):
                    all_articles.append({
                        "title": article.get("title", ""),
                        "description": article.get("description", ""),
                        "content": article.get("content", ""),
                        "source": article.get("source", {}).get("name", ""),
                        "author": article.get("author", ""),
                        "url": article.get("url", ""),
                        "published_at": article.get("publishedAt", ""),
                        "image_url": article.get("urlToImage", ""),
                        "search_keyword": keyword,
                        "collected_at": datetime.now().isoformat(),
                    })
            except Exception:
                continue
        df = pd.DataFrame(all_articles)
        if not df.empty:
            df = df.drop_duplicates(subset="url", keep="first")
        return df
    except Exception:
        return pd.DataFrame()


RSS_FEEDS = {
    "BBC World": "http://feeds.bbci.co.uk/news/world/rss.xml",
    "Al Jazeera": "https://www.aljazeera.com/xml/rss/all.xml",
    "Reuters World": "https://www.reutersagency.com/feed/?taxonomy=best-sectors&post_type=best",
}

def collect_rss_live():
    """Collect fresh RSS data."""
    war_terms = ["war", "conflict", "military", "attack", "troops", "bomb",
                 "missile", "ceasefire", "invasion", "strike", "combat",
                 "ukraine", "gaza", "israel", "hamas", "russia", "sudan",
                 "yemen", "houthi", "syria", "myanmar"]
    all_entries = []
    for source_name, url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                title = entry.get("title", "")
                summary = entry.get("summary", "")
                text = (title + " " + summary).lower()
                if any(term in text for term in war_terms):
                    all_entries.append({
                        "title": title,
                        "summary": summary,
                        "link": entry.get("link", ""),
                        "published": entry.get("published", ""),
                        "source": source_name,
                        "collected_at": datetime.now().isoformat(),
                    })
        except Exception:
            continue
    return pd.DataFrame(all_entries)


def append_to_csv(df, filepath):
    if df.empty:
        return
    if os.path.exists(filepath):
        df.to_csv(filepath, mode="a", header=False, index=False)
    else:
        df.to_csv(filepath, index=False)


# ─── Helper Functions ────────────────────────────────────────────────────────

def build_corpus(yt_df, news_df, rss_df):
    texts = []
    if not yt_df.empty:
        texts.extend(yt_df["title"].fillna("").tolist())
        texts.extend(yt_df["description"].fillna("").tolist())
    if not news_df.empty:
        texts.extend(news_df["title"].fillna("").tolist())
        texts.extend(news_df["description"].fillna("").tolist())
    if not rss_df.empty:
        texts.extend(rss_df["title"].fillna("").tolist())
        if "summary" in rss_df.columns:
            texts.extend(rss_df["summary"].fillna("").tolist())
    return [t for t in texts if isinstance(t, str) and t.strip()]


def extract_hashtags(texts):
    pattern = re.compile(r"#(\w+)")
    tags = []
    for t in texts:
        if t:
            tags.extend(pattern.findall(str(t)))
    return tags


# ─── Main Dashboard ──────────────────────────────────────────────────────────

st.title("War Prediction — Real-Time Content Analysis Dashboard")
st.caption(f"Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  Auto-refresh every 20s")

# Load existing data
yt_df, news_df, rss_df, geo_df = load_all_data()

# Show data status in sidebar
with st.sidebar:
    st.header("Data Status")
    st.metric("YouTube Videos", len(yt_df))
    st.metric("News Articles", len(news_df))
    st.metric("RSS Entries", len(rss_df))
    st.metric("Geocoded Locations", len(geo_df))
    st.divider()
    st.caption("Data is collected by the Jupyter notebook and refreshed here automatically.")

    if st.button("Collect Fresh Data Now"):
        with st.spinner("Collecting data..."):
            new_yt = collect_youtube_live(WAR_KEYWORDS, max_results=3)
            append_to_csv(new_yt, YOUTUBE_CSV)
            new_news = collect_news_live(WAR_KEYWORDS)
            append_to_csv(new_news, NEWS_CSV)
            new_rss = collect_rss_live()
            append_to_csv(new_rss, RSS_CSV)
            st.success(f"Collected: {len(new_yt)} videos, {len(new_news)} articles, {len(new_rss)} RSS entries")
            st.rerun()

corpus = build_corpus(yt_df, news_df, rss_df)

# ─── Tabs ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Live Map", "Trending Topics", "Engagement",
    "Hashtag Cloud", "Content Feed", "Issue Trends"
])

# ── Tab 1: Live Map ──────────────────────────────────────────────────────────
with tab1:
    st.subheader("Conflict Hotspot Map")

    if not geo_df.empty and "latitude" in geo_df.columns:
        m = folium.Map(location=[20, 30], zoom_start=3, tiles="CartoDB dark_matter")
        max_count = geo_df["count"].max() if "count" in geo_df.columns else 1
        for _, row in geo_df.iterrows():
            cnt = row.get("count", 1)
            radius = max(5, (cnt / max(max_count, 1)) * 40)
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=radius,
                popup=f"{row.get('location', 'Unknown')}: {cnt} mentions",
                color="red", fill=True, fill_color="red",
                fill_opacity=0.6, weight=1
            ).add_to(m)
        st_folium(m, width=1100, height=550)
    else:
        st.info("No geocoded location data yet. Run Section 4 in the notebook to generate location_data.csv.")

        # Fallback: try NER on corpus for a quick map
        if corpus:
            with st.expander("Generate a quick map from current text data"):
                if st.button("Run NER + Geocoding"):
                    with st.spinner("Extracting locations..."):
                        loc_counts = Counter()
                        for text in corpus[:200]:
                            doc = nlp(text[:3000])
                            for ent in doc.ents:
                                if ent.label_ == "GPE":
                                    loc_counts[ent.text] += 1

                        geolocator = Nominatim(user_agent="war_dashboard")
                        geocode_fn = RateLimiter(geolocator.geocode, min_delay_seconds=1)
                        geo_rows = []
                        for loc, cnt in loc_counts.most_common(30):
                            try:
                                result = geocode_fn(loc)
                                if result:
                                    geo_rows.append({
                                        "location": loc, "count": cnt,
                                        "latitude": result.latitude, "longitude": result.longitude
                                    })
                            except Exception:
                                pass

                        if geo_rows:
                            new_geo = pd.DataFrame(geo_rows)
                            new_geo.to_csv(LOCATION_CSV, index=False)
                            st.success(f"Geocoded {len(new_geo)} locations. Refresh to see the map.")
                            st.rerun()

# ── Tab 2: Trending Topics ───────────────────────────────────────────────────
with tab2:
    st.subheader("Trending Keywords & Topics")

    if corpus:
        col1, col2 = st.columns(2)

        with col1:
            # Keyword frequency
            full_text = " ".join(corpus).lower()
            kw_freq = {kw: full_text.count(kw.lower()) for kw in WAR_KEYWORDS}
            kw_freq = dict(sorted(kw_freq.items(), key=lambda x: x[1], reverse=True))
            fig = px.bar(x=list(kw_freq.keys()), y=list(kw_freq.values()),
                         title="War Keyword Mentions",
                         labels={"x": "Keyword", "y": "Count"},
                         color=list(kw_freq.values()), color_continuous_scale="Reds")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # TF-IDF top keywords
            try:
                tfidf = TfidfVectorizer(max_features=500, stop_words="english",
                                        ngram_range=(1, 2), max_df=0.95)
                matrix = tfidf.fit_transform(corpus)
                names = tfidf.get_feature_names_out()
                scores = matrix.mean(axis=0).A1
                top_idx = scores.argsort()[-20:][::-1]
                trend_df = pd.DataFrame({
                    "keyword": [names[i] for i in top_idx],
                    "score": [scores[i] for i in top_idx]
                })
                fig2 = px.bar(trend_df, x="keyword", y="score",
                              title="TF-IDF Trending Keywords",
                              color="score", color_continuous_scale="Viridis")
                fig2.update_layout(xaxis_tickangle=-45, showlegend=False)
                st.plotly_chart(fig2, use_container_width=True)
            except Exception as e:
                st.warning(f"TF-IDF analysis requires more data: {e}")
    else:
        st.info("No text data available yet. Run the notebook to collect data.")

# ── Tab 3: Engagement ────────────────────────────────────────────────────────
with tab3:
    st.subheader("Content Engagement Metrics")

    if not yt_df.empty and "view_count" in yt_df.columns:
        # Ensure numeric
        for col in ["view_count", "like_count", "comment_count"]:
            if col in yt_df.columns:
                yt_df[col] = pd.to_numeric(yt_df[col], errors="coerce").fillna(0)

        engagement = yt_df.groupby("search_keyword").agg({
            "view_count": "sum", "like_count": "sum", "comment_count": "sum",
            "video_id": "count"
        }).rename(columns={"video_id": "videos"}).sort_values("view_count", ascending=False)

        fig = px.bar(engagement, x=engagement.index,
                     y=["view_count", "like_count", "comment_count"],
                     title="YouTube Engagement by Conflict", barmode="group")
        st.plotly_chart(fig, use_container_width=True)

        # Top videos table
        st.markdown("**Top 20 Most Viewed Videos**")
        top = yt_df.nlargest(20, "view_count")[
            ["title", "channel", "search_keyword", "view_count", "like_count", "comment_count"]
        ]
        st.dataframe(top, use_container_width=True)
    else:
        st.info("No YouTube engagement data available yet.")

    if not news_df.empty and "source" in news_df.columns:
        src_counts = news_df["source"].value_counts().head(15)
        fig = px.bar(x=src_counts.index, y=src_counts.values,
                     title="Top News Sources Covering War Content",
                     labels={"x": "Source", "y": "Articles"})
        st.plotly_chart(fig, use_container_width=True)

# ── Tab 4: Hashtag Cloud ─────────────────────────────────────────────────────
with tab4:
    st.subheader("Hashtag Cloud")

    hashtag_texts = []
    if not yt_df.empty:
        hashtag_texts.extend(yt_df["description"].fillna("").tolist())
        if "tags" in yt_df.columns:
            hashtag_texts.extend(yt_df["tags"].fillna("").tolist())
    if not news_df.empty:
        hashtag_texts.extend(news_df["title"].fillna("").tolist())

    all_hashtags = extract_hashtags(hashtag_texts)

    if all_hashtags:
        col1, col2 = st.columns([2, 1])
        with col1:
            wc = WordCloud(width=900, height=450, background_color="black",
                           colormap="YlOrRd", max_words=80).generate(" ".join(all_hashtags))
            fig_wc, ax = plt.subplots(figsize=(12, 6))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig_wc)
            plt.close()

        with col2:
            tag_counts = Counter(all_hashtags).most_common(25)
            tag_df = pd.DataFrame(tag_counts, columns=["Hashtag", "Count"])
            st.dataframe(tag_df, use_container_width=True)
    else:
        st.info("No hashtags found in collected data yet.")

# ── Tab 5: Content Feed ──────────────────────────────────────────────────────
with tab5:
    st.subheader("Live Content Feed")

    feed_items = []
    if not news_df.empty:
        for _, row in news_df.head(30).iterrows():
            feed_items.append({
                "source": f"NEWS — {row.get('source', '')}",
                "title": row.get("title", ""),
                "url": row.get("url", ""),
                "time": row.get("published_at", ""),
            })
    if not rss_df.empty:
        for _, row in rss_df.head(20).iterrows():
            feed_items.append({
                "source": f"RSS — {row.get('source', '')}",
                "title": row.get("title", ""),
                "url": row.get("link", ""),
                "time": row.get("published", ""),
            })
    if not yt_df.empty:
        for _, row in yt_df.head(20).iterrows():
            feed_items.append({
                "source": f"YT — {row.get('channel', '')}",
                "title": row.get("title", ""),
                "url": f"https://youtube.com/watch?v={row.get('video_id', '')}",
                "time": row.get("published_at", ""),
            })

    if feed_items:
        for item in feed_items[:50]:
            with st.container():
                st.markdown(f"**[{item['source']}]** [{item['title']}]({item['url']})  ")
                st.caption(item["time"])
                st.divider()
    else:
        st.info("No content collected yet. Run the notebook or click 'Collect Fresh Data Now' in the sidebar.")

# ── Tab 6: Issue Trends ──────────────────────────────────────────────────────
with tab6:
    st.subheader("Issue & Keyword Trends Over Time")

    # Check if we have collected_at timestamps in YouTube data
    if not yt_df.empty and "collected_at" in yt_df.columns:
        yt_copy = yt_df.copy()
        yt_copy["collected_at"] = pd.to_datetime(yt_copy["collected_at"], errors="coerce")
        yt_copy = yt_copy.dropna(subset=["collected_at"])

        if not yt_copy.empty:
            yt_copy["hour"] = yt_copy["collected_at"].dt.floor("h")
            trend_data = yt_copy.groupby(["hour", "search_keyword"]).size().reset_index(name="count")

            fig = px.line(trend_data, x="hour", y="count", color="search_keyword",
                          title="Content Volume Over Time by Conflict",
                          labels={"hour": "Time", "count": "Items Collected"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Timestamp data not parseable. Collect more data over time to see trends.")
    else:
        st.info("Collect data over multiple polling cycles to see time-series trends.")

    # Keyword frequency comparison
    if corpus:
        st.markdown("**Current Keyword Landscape**")
        full_text = " ".join(corpus).lower()
        kw_data = []
        for kw in WAR_KEYWORDS:
            kw_data.append({"conflict": kw, "mentions": full_text.count(kw.lower())})
        kw_df = pd.DataFrame(kw_data).sort_values("mentions", ascending=False)

        fig = px.pie(kw_df, names="conflict", values="mentions",
                     title="Share of Mentions by Conflict")
        st.plotly_chart(fig, use_container_width=True)
