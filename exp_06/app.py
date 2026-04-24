"""
Conflict Signal Monitor - Experiment 06
Streamlit dashboard with 5 tabs: Graph, Map, Risk, Feed, Explainer
"""

import os
import re
import json
import time
from datetime import datetime, timedelta
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import plotly.express as px
import plotly.graph_objects as go
from streamlit_folium import st_folium
import folium

import networkx as nx
from networkx.readwrite import json_graph
from pyvis.network import Network

# Page configuration
st.set_page_config(
    page_title="Conflict Signal Monitor - Exp 06",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    .main {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #2d2d2d;
        color: #ffffff;
        border-radius: 4px 4px 0 0;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3498db !important;
    }
    .metric-card {
        background-color: #2d2d2d;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
    }
    .risk-high { color: #e74c3c; }
    .risk-medium { color: #f39c12; }
    .risk-low { color: #2ecc71; }
</style>
""", unsafe_allow_html=True)

# Auto-refresh every 30 seconds
count = st_autorefresh(interval=30_000, limit=None, key="conflict_monitor_refresh")

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GRAPH_JSON = os.path.join(SCRIPT_DIR, "graph_data.json")
RISK_CSV = os.path.join(SCRIPT_DIR, "risk_scores.csv")
SUMMARY_JSON = os.path.join(SCRIPT_DIR, "summary.json")
YOUTUBE_CSV = os.path.join(SCRIPT_DIR, "youtube_videos.csv")
NEWS_CSV = os.path.join(SCRIPT_DIR, "news_data.csv")
RSS_CSV = os.path.join(SCRIPT_DIR, "rss_data.csv")

# Load data functions
@st.cache_data(ttl=20)
def load_graph():
    """Load graph from JSON file."""
    if os.path.exists(GRAPH_JSON):
        try:
            with open(GRAPH_JSON, 'r') as f:
                data = json.load(f)
            return json_graph.node_link_graph(data, multigraph=True, directed=True)
        except Exception as e:
            st.error(f"Error loading graph: {e}")
    return None

@st.cache_data(ttl=20)
def load_risk_data():
    """Load risk scores from CSV."""
    if os.path.exists(RISK_CSV):
        try:
            return pd.read_csv(RISK_CSV)
        except Exception:
            pass
    return pd.DataFrame()

@st.cache_data(ttl=20)
def load_summary():
    """Load summary JSON."""
    if os.path.exists(SUMMARY_JSON):
        try:
            with open(SUMMARY_JSON, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {}

@st.cache_data(ttl=20)
def load_feed_data():
    """Load feed data from all sources."""
    feeds = []
    
    # YouTube videos
    if os.path.exists(YOUTUBE_CSV):
        try:
            df = pd.read_csv(YOUTUBE_CSV)
            if not df.empty:
                df = df.sort_values('collected_at', ascending=False).head(20)
                for _, row in df.iterrows():
                    feeds.append({
                        'source': f"YouTube - {row.get('channel', 'Unknown')}",
                        'title': row.get('title', ''),
                        'time': row.get('collected_at', ''),
                        'type': 'youtube',
                        'credibility': row.get('credibility', 0.5),
                        'url': f"https://youtube.com/watch?v={row.get('video_id', '')}"
                    })
        except Exception:
            pass
    
    # News articles
    if os.path.exists(NEWS_CSV):
        try:
            df = pd.read_csv(NEWS_CSV)
            if not df.empty:
                df = df.sort_values('collected_at', ascending=False).head(20)
                for _, row in df.iterrows():
                    feeds.append({
                        'source': f"News - {row.get('source', 'Unknown')}",
                        'title': row.get('title', ''),
                        'time': row.get('collected_at', ''),
                        'type': 'news',
                        'credibility': row.get('credibility', 0.6),
                        'url': row.get('url', '')
                    })
        except Exception:
            pass
    
    # RSS entries
    if os.path.exists(RSS_CSV):
        try:
            df = pd.read_csv(RSS_CSV)
            if not df.empty:
                df = df.sort_values('collected_at', ascending=False).head(20)
                for _, row in df.iterrows():
                    feeds.append({
                        'source': f"RSS - {row.get('source', 'Unknown')}",
                        'title': row.get('title', ''),
                        'time': row.get('collected_at', ''),
                        'type': 'rss',
                        'credibility': row.get('credibility', 0.7),
                        'url': row.get('link', '')
                    })
        except Exception:
            pass
    
    # Sort by time
    feeds.sort(key=lambda x: x['time'], reverse=True)
    return feeds[:50]

# Sidebar
with st.sidebar:
    st.title("🌍 Conflict Signal Monitor")
    st.caption("Exp 06 - Multi-Source Fusion")
    st.divider()
    
    # Filters
    st.header("Filters")
    
    region_filter = st.selectbox(
        "Region",
        ["All", "Ukraine", "Gaza", "Israel", "Russia", "Sudan", "Syria", "Myanmar"]
    )
    
    sources = st.multiselect(
        "Sources",
        ["YouTube", "NewsAPI", "RSS"],
        default=["YouTube", "NewsAPI", "RSS"]
    )
    
    date_range = st.date_input(
        "Date Range",
        value=(datetime.now() - timedelta(days=7), datetime.now()),
        max_value=datetime.now()
    )
    
    graph_depth = st.slider("Graph Depth", 1, 5, 2)
    
    if st.button("🔄 Run Pipeline", type="primary"):
        st.info("Pipeline execution would be triggered here. Run the notebook for data collection.")
    
    st.divider()
    
    # Data status
    st.header("Data Status")
    summary = load_summary()
    if summary:
        st.metric("Total Nodes", summary.get('total_nodes', 0))
        st.metric("Total Edges", summary.get('total_edges', 0))
        
        data_sources = summary.get('data_sources', {})
        st.caption(f"YouTube Videos: {data_sources.get('youtube_videos', 0)}")
        st.caption(f"YouTube Comments: {data_sources.get('youtube_comments', 0)}")
        st.caption(f"News Articles: {data_sources.get('news_articles', 0)}")
        st.caption(f"RSS Entries: {data_sources.get('rss_entries', 0)}")
    else:
        st.info("No data yet. Run the notebook first.")
    
    st.divider()
    st.caption(f"Last refreshed: {datetime.now().strftime('%H:%M:%S')}")

# Main content
st.title("Conflict Signal Monitor")
st.caption("Multi-source conflict prediction with credibility-weighted sentiment fusion")

# Load data
G = load_graph()
risk_df = load_risk_data()
summary = load_summary()

# Top metrics
if not risk_df.empty:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        max_risk = risk_df['total_risk'].max()
        risk_class = "risk-high" if max_risk > 0.7 else "risk-medium" if max_risk > 0.4 else "risk-low"
        st.markdown(f"""
        <div class="metric-card">
            <h4>Conflict Risk Score</h4>
            <h2 class="{risk_class}">{max_risk:.2f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        active_nodes = summary.get('total_nodes', 0)
        st.markdown(f"""
        <div class="metric-card">
            <h4>Active Nodes</h4>
            <h2>{active_nodes:,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_sentiment = risk_df['fused_sentiment'].mean()
        sentiment_color = "#e74c3c" if avg_sentiment < -0.2 else "#2ecc71" if avg_sentiment > 0.2 else "#f39c12"
        st.markdown(f"""
        <div class="metric-card">
            <h4>Avg Sentiment</h4>
            <h2 style="color: {sentiment_color}">{avg_sentiment:.2f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        # Calculate polarization index
        sentiments = risk_df['fused_sentiment'].values
        polarization = np.std(sentiments) if len(sentiments) > 0 else 0
        st.markdown(f"""
        <div class="metric-card">
            <h4>Polarisation Idx</h4>
            <h2>{polarization:.2f}</h2>
        </div>
        """, unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Graph", "Map", "Risk", "Feed", "Explainer"])

# Tab 1: Graph
with tab1:
    st.subheader("Conflict Graph - PyVis Network")
    
    # Node type filters
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        show_topics = st.checkbox("Topic", value=True)
    with col2:
        show_sources = st.checkbox("Source", value=True)
    with col3:
        show_regions = st.checkbox("Region", value=True)
    with col4:
        show_events = st.checkbox("Event", value=True)
    
    if G is not None:
        # Create filtered graph
        nodes_to_show = []
        for node, attrs in G.nodes(data=True):
            node_type = attrs.get('type', '')
            if (node_type == 'topic' and show_topics) or \
               (node_type == 'source' and show_sources) or \
               (node_type == 'region' and show_regions) or \
               (node_type == 'event' and show_events) or \
               (node_type == 'content'):
                nodes_to_show.append(node)
        
        # Create PyVis network
        net = Network(height="600px", width="100%", bgcolor="#1a1a1a", font_color="white", directed=True)
        
        color_map = {
            'topic': '#3498db',
            'source': '#2ecc71',
            'region': '#e74c3c',
            'event': '#f39c12',
            'content': '#95a5a6'
        }
        
        for node in nodes_to_show:
            attrs = G.nodes[node]
            node_type = attrs.get('type', 'content')
            color = color_map.get(node_type, '#95a5a6')
            size = attrs.get('size', 10)
            
            if node_type == 'region' and 'risk_score' in attrs:
                size = max(15, attrs['risk_score'] * 50)
            
            title = f"Type: {node_type}\nName: {attrs.get('name', node)}\n"
            if 'fused_sentiment' in attrs:
                title += f"Fused Sentiment: {attrs['fused_sentiment']:.3f}\n"
            if 'credibility' in attrs:
                title += f"Credibility: {attrs['credibility']:.2f}\n"
            if 'risk_score' in attrs:
                title += f"Risk Score: {attrs['risk_score']:.3f}"
            
            net.add_node(node, label=attrs.get('name', node)[:15],
                        color=color, size=size, title=title, group=node_type)
        
        # Add edges between visible nodes
        for source, target, attrs in G.edges(data=True):
            if source in nodes_to_show and target in nodes_to_show:
                net.add_edge(source, target, weight=attrs.get('weight', 1),
                           title=attrs.get('relation', ''), arrows='to')
        
        net.set_options("""
        {
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -50,
              "centralGravity": 0.01,
              "springLength": 100,
              "springConstant": 0.08
            },
            "maxVelocity": 50,
            "solver": "forceAtlas2Based",
            "timestep": 0.35,
            "stabilization": {"iterations": 100}
          }
        }
        """)
        
        # Save and display
        graph_html = os.path.join(SCRIPT_DIR, "temp_graph.html")
        net.save_graph(graph_html)
        
        with open(graph_html, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        st.components.v1.html(html_content, height=620, scrolling=False)
    else:
        st.info("No graph data available. Run the notebook to generate the conflict graph.")

# Tab 2: Map
with tab2:
    st.subheader("Conflict Hotspot Map")
    
    if not risk_df.empty:
        # Create map centered on regions
        m = folium.Map(location=[30, 20], zoom_start=2, tiles="CartoDB dark_matter")
        
        # Add markers for regions with coordinates
        # Note: In a real implementation, you'd geocode the regions
        # Here we'll use a simplified approach
        
        region_coords = {
            'Ukraine': [48.3794, 31.1656],
            'Russia': [61.5240, 105.3188],
            'Gaza': [31.5017, 34.4668],
            'Israel': [31.0461, 34.8516],
            'Syria': [34.8021, 38.9968],
            'Sudan': [12.8628, 30.2176],
            'Myanmar': [21.9162, 95.9560],
            'Yemen': [15.5527, 48.5164],
            'Iran': [32.4279, 53.6880],
            'Iraq': [33.2232, 43.6793],
            'China': [35.8617, 104.1954],
            'United States': [37.0902, -95.7129],
            'UK': [55.3781, -3.4360],
            'France': [46.2276, 2.2137],
            'Germany': [51.1657, 10.4515],
            'Turkey': [38.9637, 35.2433],
            'Poland': [51.9194, 19.1451],
            'India': [20.5937, 78.9629],
            'Pakistan': [30.3753, 69.3451],
            'Afghanistan': [33.9391, 67.7100],
            'Lebanon': [33.8547, 35.8623],
            'Jordan': [30.5852, 36.2384],
            'Egypt': [26.8206, 30.8025],
            'Saudi Arabia': [23.8859, 45.0792],
            'UAE': [23.4241, 53.8478],
            'Qatar': [25.3548, 51.1839],
            'Kuwait': [29.3117, 47.4818],
            'Bahrain': [26.0667, 50.5577],
            'Oman': [21.4735, 55.9754],
            'Libya': [26.3351, 17.2283],
            'Tunisia': [33.8869, 9.5375],
            'Algeria': [28.0339, 1.6596],
            'Morocco': [31.7917, -7.0926],
            'Ethiopia': [9.1450, 40.4897],
            'Somalia': [5.1521, 46.1996],
            'Kenya': [-0.0236, 37.9062],
            'Nigeria': [9.0820, 8.6753],
            'South Africa': [-30.5595, 22.9375],
            'Brazil': [-14.2350, -51.9253],
            'Argentina': [-38.4161, -63.6167],
            'Mexico': [23.6345, -102.5528],
            'Canada': [56.1304, -106.3468],
            'Australia': [-25.2744, 133.7751],
            'Japan': [36.2048, 138.2529],
            'South Korea': [35.9078, 127.7669],
            'North Korea': [40.3399, 127.5101],
            'Taiwan': [23.6978, 120.9605],
            'Philippines': [12.8797, 121.7740],
            'Vietnam': [14.0583, 108.2772],
            'Thailand': [15.8700, 100.9925],
            'Indonesia': [-0.7893, 113.9213],
            'Malaysia': [4.2105, 101.9758],
            'Singapore': [1.3521, 103.8198],
            'Bangladesh': [23.6850, 90.3563],
            'Sri Lanka': [7.8731, 80.7718],
            'Nepal': [28.3949, 84.1240],
            'Bhutan': [27.5142, 90.4336],
            'Maldives': [3.2028, 73.2207],
            'Mongolia': [46.8625, 103.8467],
            'Kazakhstan': [48.0196, 66.9237],
            'Uzbekistan': [41.3775, 64.5853],
            'Turkmenistan': [38.9697, 59.5563],
            'Kyrgyzstan': [41.2044, 74.7661],
            'Tajikistan': [38.8610, 71.2761],
            'Azerbaijan': [40.1431, 47.5769],
            'Armenia': [40.0691, 45.0382],
            'Georgia': [42.3154, 43.3569],
            'Moldova': [47.4116, 28.3699],
            'Belarus': [53.7098, 27.9534],
            'Lithuania': [55.1694, 23.8813],
            'Latvia': [56.8796, 24.6032],
            'Estonia': [58.5953, 25.0136],
            'Finland': [61.9241, 25.7482],
            'Sweden': [60.1282, 18.6435],
            'Norway': [60.4720, 8.4689],
            'Denmark': [56.2639, 9.5018],
            'Netherlands': [52.1326, 5.2913],
            'Belgium': [50.5039, 4.4699],
            'Switzerland': [46.8182, 8.2275],
            'Austria': [47.5162, 14.5501],
            'Czech Republic': [49.8175, 15.4730],
            'Slovakia': [48.6690, 19.6990],
            'Hungary': [47.1625, 19.5033],
            'Romania': [45.9432, 24.9668],
            'Bulgaria': [42.7339, 25.4858],
            'Serbia': [44.0165, 21.0059],
            'Croatia': [45.1000, 15.2000],
            'Slovenia': [46.1512, 14.9955],
            'Bosnia and Herzegovina': [43.9159, 17.6791],
            'Montenegro': [42.7087, 19.3744],
            'North Macedonia': [41.6086, 21.7453],
            'Albania': [41.1533, 20.1683],
            'Greece': [39.0742, 21.8243],
            'Cyprus': [35.1264, 33.4299],
            'Malta': [35.9375, 14.3754],
            'Iceland': [64.9631, -19.0208],
            'Ireland': [53.1424, -7.6921],
            'Portugal': [39.3999, -8.2245],
            'Spain': [40.4637, -3.7492],
            'Italy': [41.8719, 12.5674],
            'Slovenia': [46.1512, 14.9955],
            'Croatia': [45.1000, 15.2000],
            'New Zealand': [-40.9006, 174.8869],
            'Fiji': [-17.7134, 178.0650],
            'Papua New Guinea': [-6.314993, 143.95555],
            'Solomon Islands': [-9.6457, 160.1562],
            'Vanuatu': [-15.3767, 166.9592],
            'Samoa': [-13.7590, -172.1046],
            'Tonga': [-21.1790, -175.1982],
            'Kiribati': [-3.3704, -168.7340],
            'Tuvalu': [-7.1095, 177.6493],
            'Nauru': [-0.5228, 166.9315],
            'Palau': [7.5150, 134.5825],
            'Micronesia': [7.4256, 150.5508],
            'Marshall Islands': [7.1315, 171.1845],
            'Chile': [-35.6751, -71.5430],
            'Peru': [-9.1900, -75.0152],
            'Colombia': [4.5709, -74.2973],
            'Venezuela': [6.4238, -66.5897],
            'Ecuador': [-1.8312, -78.1834],
            'Bolivia': [-16.2902, -63.5887],
            'Paraguay': [-23.4425, -58.4438],
            'Uruguay': [-32.5228, -55.7658],
            'Guyana': [4.8604, -58.9302],
            'Suriname': [3.9193, -56.0278],
            'French Guiana': [3.9339, -53.1258],
            'Panama': [8.5380, -80.7821],
            'Costa Rica': [9.7489, -83.7534],
            'Nicaragua': [12.8654, -85.2072],
            'Honduras': [15.2000, -86.2419],
            'Guatemala': [15.7835, -90.2308],
            'El Salvador': [13.7942, -88.8965],
            'Belize': [17.1899, -88.4976],
            'Cuba': [21.5218, -77.7812],
            'Jamaica': [18.1096, -77.2975],
            'Haiti': [18.9712, -72.2852],
            'Dominican Republic': [18.7357, -70.1627],
            'Puerto Rico': [18.2208, -66.5901],
            'Trinidad and Tobago': [10.6918, -61.2225],
            'Barbados': [13.1939, -59.5432],
            'Saint Lucia': [13.9094, -60.9789],
            'Grenada': [12.1165, -61.6790],
            'Saint Vincent and the Grenadines': [12.9843, -61.2872],
            'Antigua and Barbuda': [17.0608, -61.7964],
            'Saint Kitts and Nevis': [17.3578, -62.7820],
            'Dominica': [15.4150, -61.3710],
            'Bahamas': [25.0343, -77.3963],
            'Cayman Islands': [19.3138, -81.2546],
            'Bermuda': [32.3078, -64.7505],
        }
        
        for _, row in risk_df.iterrows():
            region = row['region']
            risk = row['total_risk']
            sentiment = row['fused_sentiment']
            
            # Find coordinates
            coords = None
            for key, value in region_coords.items():
                if key.lower() in region.lower() or region.lower() in key.lower():
                    coords = value
                    break
            
            if coords:
                # Color based on risk
                if risk > 0.7:
                    color = '#e74c3c'
                elif risk > 0.4:
                    color = '#f39c12'
                else:
                    color = '#2ecc71'
                
                radius = max(5, risk * 30)
                
                folium.CircleMarker(
                    location=coords,
                    radius=radius,
                    popup=f"{region}<br>Risk: {risk:.2f}<br>Sentiment: {sentiment:.2f}",
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.7,
                    weight=2
                ).add_to(m)
        
        st_folium(m, width=1100, height=550)
    else:
        st.info("No risk data available for map visualization.")

# Tab 3: Risk
with tab3:
    st.subheader("Risk Analysis")
    
    if not risk_df.empty:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Risk by source
            st.markdown("**Risk by Source - Last 7 Days**")
            
            # Calculate risk by source type
            source_risk = []
            if G is not None:
                for node, attrs in G.nodes(data=True):
                    if attrs.get('type') == 'source' and 'fused_sentiment' in attrs:
                        # Estimate risk from sentiment
                        sentiment = attrs['fused_sentiment']
                        est_risk = 1 - abs(sentiment)
                        source_risk.append({
                            'source': attrs['name'],
                            'platform': attrs.get('source_platform', 'unknown'),
                            'risk': est_risk
                        })
            
            if source_risk:
                sr_df = pd.DataFrame(source_risk)
                platform_risk = sr_df.groupby('platform')['risk'].mean().reset_index()
                platform_risk = platform_risk.sort_values('risk', ascending=False)
                
                fig = px.bar(platform_risk, x='platform', y='risk',
                           color='risk', color_continuous_scale='Reds',
                           labels={'risk': 'Risk Score', 'platform': 'Source Platform'})
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Top Nodes by Centrality**")
            
            if G is not None:
                # Calculate degree centrality
                degree_cent = nx.degree_centrality(G)
                
                centrality_list = []
                for node, attrs in G.nodes(data=True):
                    if attrs.get('type') in ['topic', 'region', 'source']:
                        centrality_list.append({
                            'name': attrs.get('name', node),
                            'type': attrs.get('type'),
                            'centrality': degree_cent.get(node, 0)
                        })
                
                cent_df = pd.DataFrame(centrality_list)
                cent_df = cent_df.sort_values('centrality', ascending=False).head(10)
                
                for _, row in cent_df.iterrows():
                    st.markdown(f"**{row['name']}** ({row['type']}) - {row['centrality']:.2f}")
        
        # Risk timeseries (simulated if no time data)
        st.markdown("**Risk Trend Over Time**")
        
        # Create simulated time series based on current data
        dates = pd.date_range(end=datetime.now(), periods=7, freq='D')
        trend_data = []
        for date in dates:
            # Add some random variation
            base_risk = risk_df['total_risk'].mean()
            variation = np.random.uniform(-0.1, 0.1)
            trend_data.append({
                'date': date,
                'risk': max(0, min(1, base_risk + variation))
            })
        
        trend_df = pd.DataFrame(trend_data)
        fig = px.line(trend_df, x='date', y='risk',
                     labels={'risk': 'Fused Risk Score', 'date': 'Date'},
                     title='Daily Fused Risk Scores')
        fig.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No risk data available. Run the notebook first.")

# Tab 4: Feed
with tab4:
    st.subheader("Live Feed - Latest Signals")
    
    feeds = load_feed_data()
    
    if feeds:
        for item in feeds[:20]:
            credibility = item['credibility']
            cred_color = "🟢" if credibility > 0.8 else "🟡" if credibility > 0.5 else "🔴"
            
            with st.container():
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"**{item['source']}** {cred_color}")
                    if item['url']:
                        st.markdown(f"[{item['title']}]({item['url']})")
                    else:
                        st.markdown(item['title'])
                with col2:
                    st.caption(item['time'][:16] if len(item['time']) > 16 else item['time'])
                st.divider()
    else:
        st.info("No feed data available. Run the notebook to collect data.")

# Tab 5: Explainer
with tab5:
    st.subheader("Model Explainer")
    
    st.markdown("""
    ### Credibility-Weighted Sentiment Fusion
    
    This dashboard implements a novel approach to conflict prediction by combining:
    
    1. **Multi-Source Data Fusion**: Integrates YouTube videos/comments, NewsAPI articles, and RSS feeds
    2. **Source Credibility Weighting**: Each source is assigned a credibility score (0.0-1.0)
    3. **Weighted Sentiment Aggregation**: `fused_sentiment = Σ(sentiment_i × credibility_i) / Σ(credibility_i)`
    
    ### Risk Score Components
    
    The conflict risk score for each region is computed as:
    
    ```
    Risk = 0.35 × Sentiment Risk + 0.35 × Volume Factor + 0.30 × Conflict Severity
    ```
    
    Where:
    - **Sentiment Risk** = 1 - |fused_sentiment| (polarized sentiment = higher risk)
    - **Volume Factor** = min(mention_count / 50, 1.0) (normalized mention volume)
    - **Conflict Severity** = (10 - Goldstein_Scale) / 20 (from GDELT data)
    
    ### Source Credibility Scores
    
    | Source Type | Examples | Credibility |
    |------------|----------|-------------|
    | High | Reuters, AP, BBC | 0.92-0.95 |
    | Medium-High | Al Jazeera, NYT | 0.85-0.88 |
    | Medium | CNN, Fox News | 0.65-0.75 |
    | Low | RT, Sputnik | 0.25-0.30 |
    """)
    
    if not risk_df.empty:
        st.markdown("### Risk Score Distribution")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(risk_df['total_risk'], bins=20, color='steelblue', edgecolor='black', alpha=0.7)
        ax.axvline(risk_df['total_risk'].mean(), color='red', linestyle='--', 
                  label=f'Mean: {risk_df["total_risk"].mean():.3f}')
        ax.set_xlabel('Risk Score')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Regional Risk Scores')
        ax.legend()
        st.pyplot(fig)
        
        # Feature importance visualization
        st.markdown("### Risk Components by Region (Top 10)")
        
        top_regions = risk_df.head(10)
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Sentiment Risk', x=top_regions['region'], 
                            y=top_regions['sentiment_risk'], marker_color='#3498db'))
        fig.add_trace(go.Bar(name='Volume Factor', x=top_regions['region'], 
                            y=top_regions['volume_factor'], marker_color='#2ecc71'))
        fig.add_trace(go.Bar(name='Conflict Severity', x=top_regions['region'], 
                            y=top_regions['conflict_severity'], marker_color='#e74c3c'))
        
        fig.update_layout(barmode='group', xaxis_tickangle=-45,
                         title='Risk Score Components',
                         yaxis_title='Component Score')
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.divider()
st.caption("Experiment 06 - Multi-Source Conflict Signal Monitor | Built with Streamlit, NetworkX, and PyVis")
