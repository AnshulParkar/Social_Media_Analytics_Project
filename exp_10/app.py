"""
Conflict Escalation Risk Dashboard — Real-Time Multimodal Analysis
Reads data from exp_04/ CSVs, scores escalation risk dynamically, and displays
an actionable intelligence dashboard.

Run with:  streamlit run exp_10/app.py
"""

import os
import re
import time
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

import folium
from streamlit_folium import st_folium

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Conflict Escalation Dashboard",
    page_icon="⚠️",
    layout="wide"
)

# Auto-refresh every 30 seconds
count = st_autorefresh(interval=30_000, limit=None, key="escalation_refresh")

# ─── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

YOUTUBE_CSV = os.path.join(ROOT_DIR, "exp_04", "youtube_data.csv")
NEWS_CSV = os.path.join(ROOT_DIR, "exp_04", "news_data.csv")
RSS_CSV = os.path.join(ROOT_DIR, "exp_04", "rss_data.csv")
LOCATION_CSV = os.path.join(ROOT_DIR, "exp_04", "location_data.csv")
COMMENTS_CSV = os.path.join(ROOT_DIR, "data", "comments_data.csv")
ESCALATION_CSV = os.path.join(SCRIPT_DIR, "escalation_data.csv")


# ─── Escalation Scoring Engine ──────────────────────────────────────────────

HIGH_THREAT_WORDS = [
    'kill', 'killed', 'bomb', 'bombing', 'missile', 'missiles', 'airstrike',
    'airstrikes', 'invasion', 'invade', 'massacre', 'genocide', 'casualties',
    'dead', 'death', 'deaths', 'destroy', 'destroyed', 'destruction',
    'execute', 'executed', 'explosion', 'terror', 'terrorist', 'assassination',
    'nuclear', 'chemical', 'biological', 'drone strike', 'shelling',
]

MEDIUM_THREAT_WORDS = [
    'war', 'conflict', 'attack', 'attacked', 'strike', 'strikes', 'troops',
    'military', 'combat', 'weapon', 'weapons', 'drone', 'drones', 'siege',
    'blockade', 'sanctions', 'ceasefire', 'violation', 'violations',
    'escalation', 'tension', 'tensions', 'threat', 'threats', 'hostile',
    'refugee', 'displaced', 'humanitarian', 'crisis', 'occupation',
]

CONFLICT_REGIONS = [
    'ukraine', 'russia', 'gaza', 'israel', 'iran', 'sudan', 'yemen',
    'houthi', 'syria', 'myanmar', 'lebanon', 'hezbollah', 'hamas',
    'palestine', 'kremlin', 'nato', 'taliban', 'darfur',
]


def count_keywords(text, keyword_list):
    if pd.isna(text):
        return 0
    text_lower = str(text).lower()
    return sum(1 for kw in keyword_list if kw in text_lower)


def compute_aggression_score(text):
    if pd.isna(text):
        return 0.0
    text_lower = str(text).lower()
    words = text_lower.split()
    if len(words) == 0:
        return 0.0
    high_count = count_keywords(text, HIGH_THREAT_WORDS)
    medium_count = count_keywords(text, MEDIUM_THREAT_WORDS)
    region_count = count_keywords(text, CONFLICT_REGIONS)
    raw_score = (high_count * 3.0) + (medium_count * 1.5) + (region_count * 0.5)
    return round(raw_score / max(len(words), 1) * 100, 2)


def detect_region(text):
    if pd.isna(text):
        return 'Unknown'
    text_lower = str(text).lower()
    region_map = {
        'Ukraine-Russia': ['ukraine', 'russia', 'kremlin', 'kyiv', 'moscow', 'donbas'],
        'Israel-Gaza': ['gaza', 'israel', 'hamas', 'palestine', 'idf', 'netanyahu'],
        'Iran-US': ['iran', 'hormuz', 'tehran', 'irgc'],
        'Sudan': ['sudan', 'darfur', 'khartoum'],
        'Yemen': ['yemen', 'houthi', 'aden', 'sanaa'],
        'Syria': ['syria', 'damascus', 'assad'],
        'Myanmar': ['myanmar', 'burma', 'rohingya'],
        'Lebanon': ['lebanon', 'hezbollah', 'beirut'],
    }
    for region, keywords in region_map.items():
        if any(kw in text_lower for kw in keywords):
            return region
    return 'Other'


def assign_escalation(row):
    score = 0
    score += row.get('high_threat_count', 0) * 3
    score += row.get('medium_threat_count', 0) * 1.5
    if row.get('view_count', 0) > 50000:
        score += 2
    elif row.get('view_count', 0) > 10000:
        score += 1
    if row.get('like_count', 0) > 500:
        score += 1
    if row.get('comment_count', 0) > 100:
        score += 1
    if row.get('aggression_score', 0) > 15:
        score += 3
    elif row.get('aggression_score', 0) > 8:
        score += 1.5
    if score >= 8:
        return 'High'
    elif score >= 3:
        return 'Medium'
    else:
        return 'Low'


# ─── Data Loading & Processing ──────────────────────────────────────────────

@st.cache_data(ttl=25)
def load_csv(path):
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


@st.cache_data(ttl=25)
def generate_escalation_data():
    """Load all sources and generate escalation scores dynamically."""
    youtube_df = load_csv(YOUTUBE_CSV)
    news_df = load_csv(NEWS_CSV)
    rss_df = load_csv(RSS_CSV)
    comments_df = load_csv(COMMENTS_CSV)

    all_records = []

    # YouTube videos
    for _, row in youtube_df.iterrows():
        text = f"{row.get('title', '')} {row.get('description', '')}"
        all_records.append({
            'source_type': 'YouTube Video',
            'text': str(row.get('title', ''))[:300],
            'full_text': text[:1000],
            'high_threat_count': count_keywords(text, HIGH_THREAT_WORDS),
            'medium_threat_count': count_keywords(text, MEDIUM_THREAT_WORDS),
            'threat_keyword_count': count_keywords(text, HIGH_THREAT_WORDS) + count_keywords(text, MEDIUM_THREAT_WORDS),
            'aggression_score': compute_aggression_score(text),
            'view_count': pd.to_numeric(row.get('view_count', 0), errors='coerce') or 0,
            'like_count': pd.to_numeric(row.get('like_count', 0), errors='coerce') or 0,
            'comment_count': pd.to_numeric(row.get('comment_count', 0), errors='coerce') or 0,
            'region': detect_region(text),
            'timestamp': row.get('published_at', row.get('collected_at', '')),
            'source_name': row.get('channel', ''),
            'url': f"https://youtube.com/watch?v={row.get('video_id', '')}",
            'search_keyword': row.get('search_keyword', ''),
        })

    # News articles
    for _, row in news_df.iterrows():
        text = f"{row.get('title', '')} {row.get('description', '')}"
        all_records.append({
            'source_type': 'News Article',
            'text': str(row.get('title', ''))[:300],
            'full_text': text[:1000],
            'high_threat_count': count_keywords(text, HIGH_THREAT_WORDS),
            'medium_threat_count': count_keywords(text, MEDIUM_THREAT_WORDS),
            'threat_keyword_count': count_keywords(text, HIGH_THREAT_WORDS) + count_keywords(text, MEDIUM_THREAT_WORDS),
            'aggression_score': compute_aggression_score(text),
            'view_count': 0,
            'like_count': 0,
            'comment_count': 0,
            'region': detect_region(text),
            'timestamp': row.get('published_at', row.get('collected_at', '')),
            'source_name': row.get('source', ''),
            'url': row.get('url', ''),
            'search_keyword': row.get('search_keyword', ''),
        })

    # RSS feeds
    for _, row in rss_df.iterrows():
        text = f"{row.get('title', '')} {row.get('summary', '')}"
        all_records.append({
            'source_type': 'RSS Feed',
            'text': str(row.get('title', ''))[:300],
            'full_text': text[:1000],
            'high_threat_count': count_keywords(text, HIGH_THREAT_WORDS),
            'medium_threat_count': count_keywords(text, MEDIUM_THREAT_WORDS),
            'threat_keyword_count': count_keywords(text, HIGH_THREAT_WORDS) + count_keywords(text, MEDIUM_THREAT_WORDS),
            'aggression_score': compute_aggression_score(text),
            'view_count': 0,
            'like_count': 0,
            'comment_count': 0,
            'region': detect_region(text),
            'timestamp': row.get('published', row.get('collected_at', '')),
            'source_name': row.get('source', ''),
            'url': row.get('link', ''),
            'search_keyword': '',
        })

    # YouTube comments
    for _, row in comments_df.iterrows():
        text = str(row.get('comment_text', ''))
        all_records.append({
            'source_type': 'YouTube Comment',
            'text': text[:300],
            'full_text': text[:1000],
            'high_threat_count': count_keywords(text, HIGH_THREAT_WORDS),
            'medium_threat_count': count_keywords(text, MEDIUM_THREAT_WORDS),
            'threat_keyword_count': count_keywords(text, HIGH_THREAT_WORDS) + count_keywords(text, MEDIUM_THREAT_WORDS),
            'aggression_score': compute_aggression_score(text),
            'view_count': 0,
            'like_count': pd.to_numeric(row.get('like_count', 0), errors='coerce') or 0,
            'comment_count': pd.to_numeric(row.get('reply_count', 0), errors='coerce') or 0,
            'region': detect_region(text),
            'timestamp': row.get('published_at', ''),
            'source_name': row.get('author', ''),
            'url': '',
            'search_keyword': '',
        })

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df['escalation_risk'] = df.apply(assign_escalation, axis=1)
    df['engagement_score'] = (
        df['view_count'].clip(upper=1e6) / 1e6 * 0.5 +
        df['like_count'].clip(upper=1e4) / 1e4 * 0.3 +
        df['comment_count'].clip(upper=1e3) / 1e3 * 0.2
    ) * 100

    # Save to CSV
    try:
        df.to_csv(ESCALATION_CSV, index=False)
    except Exception:
        pass

    return df


# ─── Main Dashboard ─────────────────────────────────────────────────────────

st.title("Conflict Escalation Risk Dashboard")
st.caption(
    f"Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  "
    "Auto-refresh every 30s  |  Data scored dynamically from exp_04 sources"
)

# Load data
df = generate_escalation_data()
location_df = load_csv(LOCATION_CSV)

if df.empty:
    st.error("No data available. Ensure exp_04/ CSVs exist.")
    st.stop()

# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Data Status")
    st.metric("Total Items", len(df))
    source_counts = df['source_type'].value_counts()
    for src, cnt in source_counts.items():
        st.metric(src, cnt)
    st.divider()

    st.subheader("Risk Summary")
    high_n = len(df[df['escalation_risk'] == 'High'])
    med_n = len(df[df['escalation_risk'] == 'Medium'])
    low_n = len(df[df['escalation_risk'] == 'Low'])
    st.metric("HIGH Risk", high_n, delta=f"{high_n/len(df)*100:.1f}%")
    st.metric("MEDIUM Risk", med_n, delta=f"{med_n/len(df)*100:.1f}%")
    st.metric("LOW Risk", low_n, delta=f"{low_n/len(df)*100:.1f}%")
    st.divider()

    # Filters
    st.subheader("Filters")
    selected_risk = st.multiselect(
        "Escalation Risk", ['High', 'Medium', 'Low'],
        default=['High', 'Medium', 'Low']
    )
    selected_sources = st.multiselect(
        "Data Source", df['source_type'].unique().tolist(),
        default=df['source_type'].unique().tolist()
    )
    selected_regions = st.multiselect(
        "Conflict Region",
        [r for r in df['region'].unique() if r not in ('Other', 'Unknown')],
        default=[r for r in df['region'].unique() if r not in ('Other', 'Unknown')]
    )

# Apply filters
mask = (
    df['escalation_risk'].isin(selected_risk) &
    df['source_type'].isin(selected_sources)
)
if selected_regions:
    mask = mask & (df['region'].isin(selected_regions) | df['region'].isin(['Other', 'Unknown']))
filtered_df = df[mask]


# ─── Tabs ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Escalation Overview", "Conflict Map", "Risk Timeline",
    "Keyword Analysis", "Source Explorer", "Actionable Report"
])


# ── Tab 1: Escalation Overview ──────────────────────────────────────────────
with tab1:
    st.subheader("Escalation Risk Overview")

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Analyzed", len(filtered_df))
    with col2:
        st.metric("HIGH Risk", len(filtered_df[filtered_df['escalation_risk'] == 'High']),
                   delta="Critical", delta_color="inverse")
    with col3:
        st.metric("MEDIUM Risk", len(filtered_df[filtered_df['escalation_risk'] == 'Medium']),
                   delta="Warning", delta_color="off")
    with col4:
        st.metric("LOW Risk", len(filtered_df[filtered_df['escalation_risk'] == 'Low']),
                   delta="Normal", delta_color="normal")

    col_left, col_right = st.columns(2)

    with col_left:
        # Pie chart
        risk_counts = filtered_df['escalation_risk'].value_counts()
        fig = px.pie(
            values=risk_counts.values, names=risk_counts.index,
            title="Escalation Risk Distribution",
            color=risk_counts.index,
            color_discrete_map={'High': '#e74c3c', 'Medium': '#f39c12', 'Low': '#2ecc71'}
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        # By source
        fig = px.histogram(
            filtered_df, x='source_type', color='escalation_risk',
            title="Risk Distribution by Data Source",
            color_discrete_map={'High': '#e74c3c', 'Medium': '#f39c12', 'Low': '#2ecc71'},
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)

    # Region risk heatmap
    region_data = filtered_df[~filtered_df['region'].isin(['Other', 'Unknown'])]
    if not region_data.empty:
        region_risk = region_data.groupby(['region', 'escalation_risk']).size().unstack(fill_value=0)
        for col in ['High', 'Medium', 'Low']:
            if col not in region_risk.columns:
                region_risk[col] = 0
        fig = px.imshow(
            region_risk[['High', 'Medium', 'Low']],
            title="Escalation Risk Heatmap by Conflict Region",
            color_continuous_scale='YlOrRd', aspect='auto',
            labels=dict(color="Count")
        )
        st.plotly_chart(fig, use_container_width=True)


# ── Tab 2: Conflict Map ────────────────────────────────────────────────────
with tab2:
    st.subheader("Conflict Hotspot Map")

    if not location_df.empty and 'latitude' in location_df.columns:
        # Merge location data with escalation region counts
        region_high = filtered_df[filtered_df['escalation_risk'] == 'High']['region'].value_counts()

        m = folium.Map(location=[25, 45], zoom_start=3, tiles="CartoDB dark_matter")
        max_count = location_df['count'].max() if 'count' in location_df.columns else 1

        for _, row in location_df.iterrows():
            loc_name = row.get('location', 'Unknown')
            cnt = row.get('count', 1)
            radius = max(5, (cnt / max(max_count, 1)) * 40)

            # Color by risk (simple mapping)
            loc_lower = str(loc_name).lower()
            color = 'green'
            for region, high_cnt in region_high.items():
                if any(kw in loc_lower for kw in str(region).lower().split('-')):
                    if high_cnt > 20:
                        color = 'red'
                    elif high_cnt > 5:
                        color = 'orange'
                    break

            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=radius,
                popup=f"{loc_name}: {cnt} mentions",
                color=color, fill=True, fill_color=color,
                fill_opacity=0.6, weight=1
            ).add_to(m)

        st_folium(m, width=1100, height=550)

        st.markdown("**Legend:** Red = High escalation, Orange = Medium, Green = Low")
    else:
        st.info("No location data available. Ensure exp_04/location_data.csv exists.")


# ── Tab 3: Risk Timeline ───────────────────────────────────────────────────
with tab3:
    st.subheader("Escalation Trends Over Time")

    time_df = filtered_df.copy()
    time_df['timestamp'] = pd.to_datetime(time_df['timestamp'], errors='coerce')
    time_df = time_df.dropna(subset=['timestamp'])

    if not time_df.empty:
        time_df['date'] = time_df['timestamp'].dt.date
        daily_risk = time_df.groupby(['date', 'escalation_risk']).size().reset_index(name='count')
        fig = px.line(
            daily_risk, x='date', y='count', color='escalation_risk',
            title="Daily Escalation Risk Volume",
            color_discrete_map={'High': '#e74c3c', 'Medium': '#f39c12', 'Low': '#2ecc71'},
            labels={'date': 'Date', 'count': 'Items'}
        )
        st.plotly_chart(fig, use_container_width=True)

        # By search keyword over time
        if 'search_keyword' in time_df.columns:
            kw_time = time_df[time_df['search_keyword'] != ''].groupby(
                ['date', 'search_keyword']).size().reset_index(name='count')
            if not kw_time.empty:
                fig2 = px.area(
                    kw_time, x='date', y='count', color='search_keyword',
                    title="Content Volume by Conflict Topic Over Time"
                )
                st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Insufficient timestamp data for timeline visualization.")


# ── Tab 4: Keyword Analysis ────────────────────────────────────────────────
with tab4:
    st.subheader("Threat Keyword Analysis")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Word cloud from HIGH risk text
        high_texts = filtered_df[filtered_df['escalation_risk'] == 'High']['full_text'].fillna('')
        all_high_text = ' '.join(high_texts.tolist())
        if all_high_text.strip():
            wc = WordCloud(
                width=900, height=450, background_color='black',
                colormap='YlOrRd', max_words=100,
                stopwords={'the', 'a', 'an', 'in', 'on', 'of', 'to', 'and', 'is',
                           'was', 'for', 'that', 'with', 'has', 'have', 'are', 'from',
                           'its', 'it', 'this', 'by', 'at', 'as', 'be', 'or', 'not',
                           'but', 'he', 'she', 'his', 'her', 'they', 'their', 'been',
                           'said', 'more', 'after', 'will', 'who', 'than', 'would'}
            ).generate(all_high_text)
            fig_wc, ax = plt.subplots(figsize=(12, 6))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            ax.set_title('Threat Keyword Cloud (HIGH Risk Content)', fontsize=14, fontweight='bold')
            st.pyplot(fig_wc)
            plt.close()
        else:
            st.info("No HIGH risk content for word cloud.")

    with col2:
        # Top keywords table
        all_kws = HIGH_THREAT_WORDS + MEDIUM_THREAT_WORDS
        kw_counts = {}
        all_text = ' '.join(filtered_df['full_text'].fillna('').tolist()).lower()
        for kw in all_kws:
            c = all_text.count(kw)
            if c > 0:
                kw_counts[kw] = c
        if kw_counts:
            kw_df = pd.DataFrame(
                sorted(kw_counts.items(), key=lambda x: x[1], reverse=True)[:25],
                columns=['Keyword', 'Count']
            )
            st.dataframe(kw_df, use_container_width=True, height=500)

    # Keyword bar chart
    if kw_counts:
        top_kw = dict(sorted(kw_counts.items(), key=lambda x: x[1], reverse=True)[:20])
        fig = px.bar(
            x=list(top_kw.keys()), y=list(top_kw.values()),
            title="Top 20 Threat Keywords Across All Content",
            labels={'x': 'Keyword', 'y': 'Frequency'},
            color=list(top_kw.values()), color_continuous_scale='Reds'
        )
        fig.update_layout(xaxis_tickangle=-45, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


# ── Tab 5: Source Explorer ──────────────────────────────────────────────────
with tab5:
    st.subheader("Source Explorer — All Scored Items")

    # Color-coded table
    display_cols = ['escalation_risk', 'source_type', 'region', 'text',
                    'threat_keyword_count', 'aggression_score', 'view_count',
                    'like_count', 'source_name', 'timestamp']
    display_df = filtered_df[display_cols].sort_values(
        'aggression_score', ascending=False
    ).head(200)

    def color_risk(val):
        color_map = {'High': '#ffcccc', 'Medium': '#fff3cd', 'Low': '#d4edda'}
        return f'background-color: {color_map.get(val, "")}'

    styled = display_df.style.map(color_risk, subset=['escalation_risk'])
    st.dataframe(styled, use_container_width=True, height=600)

    # Download button
    csv_data = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Full Escalation Data (CSV)",
        data=csv_data,
        file_name=f"escalation_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime='text/csv'
    )


# ── Tab 6: Actionable Report ───────────────────────────────────────────────
with tab6:
    st.subheader("Actionable Intelligence Report")

    total = len(filtered_df)
    high_c = len(filtered_df[filtered_df['escalation_risk'] == 'High'])
    med_c = len(filtered_df[filtered_df['escalation_risk'] == 'Medium'])
    low_c = len(filtered_df[filtered_df['escalation_risk'] == 'Low'])

    # Top regions
    high_regions = filtered_df[
        (filtered_df['escalation_risk'] == 'High') &
        (~filtered_df['region'].isin(['Other', 'Unknown']))
    ]['region'].value_counts().head(5)

    # Top keywords in HIGH
    high_text = ' '.join(
        filtered_df[filtered_df['escalation_risk'] == 'High']['full_text'].fillna('').tolist()
    ).lower()
    top_kws = sorted(
        [(kw, high_text.count(kw)) for kw in HIGH_THREAT_WORDS if high_text.count(kw) > 0],
        key=lambda x: x[1], reverse=True
    )[:10]

    report = f"""
## CONFLICT ESCALATION INTELLIGENCE REPORT
**Generated:** {datetime.now().strftime('%B %d, %Y at %H:%M')}

---

### Data Sources Analyzed
| Source | Count |
|--------|-------|
"""
    for src, cnt in filtered_df['source_type'].value_counts().items():
        report += f"| {src} | {cnt} |\n"

    report += f"""| **TOTAL** | **{total}** |

---

### Escalation Risk Assessment

| Risk Level | Count | Percentage | Status |
|------------|-------|------------|--------|
| **HIGH** | {high_c} | {high_c/max(total,1)*100:.1f}% | Aggressive language + high engagement + multiple threat indicators |
| **MEDIUM** | {med_c} | {med_c/max(total,1)*100:.1f}% | Some conflict indicators, moderate engagement |
| **LOW** | {low_c} | {low_c/max(total,1)*100:.1f}% | General discussion, minimal threat language |

---

### Highest-Risk Conflict Regions
"""
    for region, cnt in high_regions.items():
        report += f"- **{region}**: {cnt} high-risk items detected\n"

    report += "\n### Top Flagged Keywords (in HIGH-risk content)\n"
    for kw, cnt in top_kws:
        report += f"- `{kw}`: {cnt} mentions\n"

    report += f"""
---

### Actionable Recommendations

1. **Monitor {high_regions.index[0] if len(high_regions) > 0 else 'top region'} closely** — highest concentration of escalation signals detected across multiple sources
2. **Cross-reference HIGH-risk YouTube content with news articles** for independent verification of conflict escalation
3. **Track engagement velocity** on flagged videos — sudden spikes in views/comments often precede major conflict events by 24-48 hours
4. **Deploy humanitarian monitoring** for regions showing sustained HIGH risk patterns
5. **Use this dashboard for daily situation awareness briefings** — auto-refreshes every 30 seconds with latest data

---

### Who Should Use This Report

- **Journalists**: Identify which conflict zones show online escalation before traditional media reports
- **NGOs**: Prioritize humanitarian resources based on regional risk scores
- **Defense Analysts**: Monitor engagement velocity as early warning signals
- **Policy Makers**: Track discourse escalation to inform diplomatic decisions
"""

    st.markdown(report)

    # Download report
    st.download_button(
        label="Download Report (Markdown)",
        data=report.encode('utf-8'),
        file_name=f"escalation_report_{datetime.now().strftime('%Y%m%d')}.md",
        mime='text/markdown'
    )
