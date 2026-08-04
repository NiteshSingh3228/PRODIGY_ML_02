import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if "theme" not in st.session_state:
    st.session_state.theme = "light" 

def toggle_theme():
    st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"

IS_DARK = st.session_state.theme == "dark"

# CSS styling
css = f"""
<style>
:root {{
    --bg: {'#09090b' if IS_DARK else '#ffffff'};
    --bg-subtle: {'#0c0c0f' if IS_DARK else '#f9fafb'};
    --card: {'#0c0c0f' if IS_DARK else '#ffffff'};
    --card-hover: {'#131316' if IS_DARK else '#f4f4f5'};
    --border: {'#1e1e24' if IS_DARK else '#e4e4e7'};
    --border-subtle: {'#16161a' if IS_DARK else '#f0f0f2'};
    --text: {'#fafafa' if IS_DARK else '#09090b'};
    --text-muted: #71717a;
    --text-dim: {'#52525b' if IS_DARK else '#a1a1aa'};
    --accent: #2563eb;
    --accent-muted: #1d4ed8;
    --green: {'#22c55e' if IS_DARK else '#16a34a'};
    --green-muted: {'rgba(34,197,94,0.12)' if IS_DARK else 'rgba(22,163,74,0.08)'};
    --red: {'#ef4444' if IS_DARK else '#dc2626'};
    --red-muted: {'rgba(239,68,68,0.12)' if IS_DARK else 'rgba(220,38,38,0.08)'};
    --amber: {'#f59e0b' if IS_DARK else '#d97706'};
    --amber-muted: {'rgba(245,158,11,0.12)' if IS_DARK else 'rgba(217,119,6,0.08)'};
    --shadow: {'none' if IS_DARK else '0 1px 3px rgba(0,0,0,0.04), 0 1px 2px rgba(0,0,0,0.03)'};
    --radius: 10px;
}}

header[data-testid="stHeader"], #MainMenu, footer, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"], .stDeployButton,
div[data-testid="stSidebarCollapsedControl"] {{
    display: none !important;
}}

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"], .main, .block-container, section[data-testid="stMain"] {{
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', -apple-system, sans-serif !important;
}}

.block-container {{
    padding: 2rem 2.5rem 3rem !important;
    max-width: 1360px !important;
}}

button[data-baseweb="tab"] {{
    background: transparent !important;
    color: var(--text-muted) !important;
    font-size: 0.835rem !important;
    font-weight: 500 !important;
    padding: 0.55rem 1rem !important;
    border: 1px solid transparent !important;
    border-radius: 7px !important;
}}

button[data-baseweb="tab"][aria-selected="true"] {{
    color: var(--text) !important;
    background: var(--card) !important;
    border-color: var(--border) !important;
}}

[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] {{
    display: none !important;
}}

[data-baseweb="tab-list"] {{
    gap: 4px !important;
    background: var(--bg-subtle) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 3px;
}}

.metric-card {{ background: var(--card); border: 1px solid var(--border); border-radius: var(--radius); padding: 1.25rem 1.4rem; box-shadow: var(--shadow); }}
.metric-label {{ font-size: 0.78rem; color: var(--text-muted); font-weight: 500; }}
.metric-value {{ font-size: 1.75rem; font-weight: 700; color: var(--text); letter-spacing: -0.03em; }}
.metric-delta {{ font-size: 0.75rem; font-weight: 500; margin-top: 0.4rem; padding: 2px 8px; border-radius: 6px; display: inline-flex; align-items: center; gap: 3px; }}
.delta-up {{ color: var(--green); background: var(--green-muted); }}
.delta-down {{ color: var(--red); background: var(--red-muted); }}
.delta-warn {{ color: var(--amber); background: var(--amber-muted); }}

.chart-wrap {{ background: var(--card); border: 1px solid var(--border); border-radius: var(--radius); padding: 1.2rem 1.2rem 0.6rem; box-shadow: var(--shadow); }}
.chart-title {{ font-size: 0.82rem; font-weight: 600; color: var(--text); }}
.chart-subtitle {{ font-size: 0.72rem; color: var(--text-dim); margin-bottom: 0.8rem; }}

.brand {{ font-size: 1.4rem; font-weight: 700; color: var(--text); display: flex; align-items: center; gap: 10px; margin-bottom: 1rem; }}
.brand-icon {{ color: var(--accent); }}

.badge {{ display: inline-block; padding: 2px 9px; border-radius: 6px; font-size: 0.72rem; font-weight: 500; }}
.badge-green {{ color: var(--green); background: var(--green-muted); }}
.badge-red {{ color: var(--red); background: var(--red-muted); }}
.badge-amber {{ color: var(--amber); background: var(--amber-muted); }}
.badge-blue {{ color: var(--accent); background: rgba(37,99,235,0.1); }}
.badge-purple {{ color: #9333ea; background: rgba(147, 51, 234, 0.1); }}

.data-table {{ width: 100%; border-collapse: separate; border-spacing: 0; font-size: 0.8rem; }}
.data-table th {{ text-align: left; padding: 0.6rem 0.8rem; color: var(--text-muted); font-weight: 500; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.04em; border-bottom: 1px solid var(--border); }}
.data-table td {{ padding: 0.65rem 0.8rem; color: var(--text); border-bottom: 1px solid var(--border-subtle); }}
.data-table tr:last-child td {{ border-bottom: none; }}

[data-testid="stHorizontalBlock"] {{ gap: 1.25rem !important; }}
[data-testid="stVerticalBlock"] > div:has(> [data-testid="stHorizontalBlock"]) {{
    margin-bottom: 0.5rem !important;
}}

.predict-result {{
    padding: 1.5rem;
    border-radius: var(--radius);
    background: var(--bg-subtle);
    border: 1px solid var(--accent);
    margin-top: 1rem;
    text-align: center;
}}
.predict-result-title {{ font-size: 0.9rem; color: var(--text-muted); font-weight: 500; }}
.predict-result-value {{ font-size: 2rem; font-weight: 700; color: var(--accent); margin-top: 0.5rem; }}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color="#71717a" if not IS_DARK else "#a1a1aa", size=11),
    margin=dict(l=0, r=0, t=8, b=0),
    xaxis=dict(
        gridcolor="rgba(0,0,0,0.04)" if not IS_DARK else "rgba(255,255,255,0.04)",
        zerolinecolor="rgba(0,0,0,0.04)" if not IS_DARK else "rgba(255,255,255,0.04)",
        tickfont=dict(size=10, color="#71717a"),
    ),
    yaxis=dict(
        gridcolor="rgba(0,0,0,0.04)" if not IS_DARK else "rgba(255,255,255,0.04)",
        zerolinecolor="rgba(0,0,0,0.04)" if not IS_DARK else "rgba(255,255,255,0.04)",
        tickfont=dict(size=10, color="#71717a"),
    ),
)

def metric_card(label, value, delta=None, delta_type="up"):
    cls = f"delta-{delta_type}"
    arrow = "↑" if delta_type == "up" else ("↓" if delta_type == "down" else "→")
    delta_html = f'<div class="metric-delta {cls}">{arrow} {delta}</div>' if delta else ""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

# Data loading and Model training
@st.cache_data
def load_data_and_train_model():
    file_path = "Mall_Customers.csv"
    if not os.path.exists(file_path):
        return None, None, None, None
        
    df = pd.read_csv(file_path)
    features = ['Annual Income (k$)', 'Spending Score (1-100)']
    X = df[features].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    OPTIMAL_K = 5
    kmeans = KMeans(n_clusters=OPTIMAL_K, init='k-means++', random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    df['Cluster'] = clusters
    
    # Calculate mapping from cluster ID to label based on centroids
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    centers_df = pd.DataFrame(centroids, columns=['Annual Income (k$)', 'Spending Score (1-100)'])
    income_mean = centers_df['Annual Income (k$)'].mean()
    spend_mean = centers_df['Spending Score (1-100)'].mean()
    
    cluster_labels = {}
    for i, row in centers_df.iterrows():
        inc = row['Annual Income (k$)']
        spend = row['Spending Score (1-100)']
        if abs(inc - income_mean) < 12 and abs(spend - spend_mean) < 12:
            cluster_labels[i] = "Standard"
        elif inc >= income_mean and spend >= spend_mean:
            cluster_labels[i] = "Target"
        elif inc >= income_mean and spend < spend_mean:
            cluster_labels[i] = "Careful"
        elif inc < income_mean and spend >= spend_mean:
            cluster_labels[i] = "Careless"
        else:
            cluster_labels[i] = "Sensible"
            
    df['Cluster_Label'] = df['Cluster'].map(cluster_labels)
    df['Cluster_Str'] = df['Cluster_Label'] + " (Segment " + df['Cluster'].astype(str) + ")"
    
    return df, scaler, kmeans, OPTIMAL_K, cluster_labels

df, scaler, kmeans, k, cluster_labels = load_data_and_train_model()

# Header
head_left, head_right = st.columns([8, 1])
with head_left:
    st.markdown("""
    <div class="brand">
        <span class="brand-icon">◆</span>
        <span class="brand-name">Customer Segmentation</span>
    </div>
    """, unsafe_allow_html=True)
with head_right:
    theme_label = "☀️ Light" if IS_DARK else "🌙 Dark"
    st.button(theme_label, on_click=toggle_theme, use_container_width=True)

if df is None:
    st.error("Dataset not found. Please ensure Mall_Customers.csv is in the directory.")
    st.stop()

# Layout
tab1, tab2 = st.tabs(["Dashboard", "Predict Segment"])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    with c1: 
        metric_card("Total Customers", len(df), delta=None)
    with c2: 
        metric_card("Avg Annual Income", f"${df['Annual Income (k$)'].mean():.1f}k", delta=None)
    with c3: 
        metric_card("Avg Spending Score", f"{df['Spending Score (1-100)'].mean():.1f}", delta=None)
    with c4: 
        metric_card("Number of Segments", k, delta=None)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Scatter Plot
    st.markdown("""
    <div class="chart-wrap">
        <div class="chart-title">Customer Clusters (K-Means)</div>
        <div class="chart-subtitle">Annual Income vs Spending Score</div>
    """, unsafe_allow_html=True)
    
    fig = px.scatter(
        df, 
        x="Annual Income (k$)", 
        y="Spending Score (1-100)", 
        color="Cluster_Str",
        hover_data=["Age", "Gender"],
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    
    # Add centroids
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    fig.add_trace(go.Scatter(
        x=centroids[:, 0],
        y=centroids[:, 1],
        mode='markers',
        marker=dict(size=12, color='black', symbol='x', line=dict(width=2, color='white')),
        name='Centroids'
    ))
    
    fig.update_layout(**PLOT_LAYOUT)
    fig.update_layout(legend_title_text='')
    
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Data Table
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="chart-title" style="margin-bottom: 0.5rem;">Sample Customer Data</div>', unsafe_allow_html=True)
    
    sample_df = df.head(10)
    rows = ""
    for _, row in sample_df.iterrows():
        if row['Cluster'] == 0: cluster_class = "badge-amber"
        elif row['Cluster'] == 1: cluster_class = "badge-green"
        elif row['Cluster'] == 2: cluster_class = "badge-red"
        elif row['Cluster'] == 3: cluster_class = "badge-purple"
        elif row['Cluster'] == 4: cluster_class = "badge-blue"
        
        rows += f"""<tr>
    <td>{row['CustomerID']}</td>
    <td>{row['Gender']}</td>
    <td>{row['Age']}</td>
    <td>${row['Annual Income (k$)']}k</td>
    <td>{row['Spending Score (1-100)']}</td>
    <td><span class="badge {cluster_class}">{row['Cluster_Label']} (Segment {row['Cluster']})</span></td>
</tr>"""
        
    st.markdown(f"""
<div class="chart-wrap">
<table class="data-table">
    <thead>
        <tr>
            <th>Customer ID</th>
            <th>Gender</th>
            <th>Age</th>
            <th>Annual Income</th>
            <th>Spending Score</th>
            <th>Assigned Segment</th>
        </tr>
    </thead>
    <tbody>{rows}</tbody>
</table>
</div>
""", unsafe_allow_html=True)

with tab2:
    st.markdown("""
    <div style="max-width: 600px; margin: 0 auto; padding-top: 2rem;">
        <h3 style="text-align: center; margin-bottom: 1.5rem;">Determine Customer Segment</h3>
    """, unsafe_allow_html=True)
    
    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            income = st.number_input("Annual Income (k$)", min_value=1, max_value=200, value=60)
        with col2:
            spending = st.number_input("Spending Score (1-100)", min_value=1, max_value=100, value=50)
            
        submitted = st.form_submit_button("Predict Segment", use_container_width=True)
        
    if submitted:
        features = ['Annual Income (k$)', 'Spending Score (1-100)']
        input_data = pd.DataFrame([[income, spending]], columns=features)
        scaled_input = scaler.transform(input_data)
        predicted_cluster = kmeans.predict(scaled_input)[0]
        predicted_label = cluster_labels[predicted_cluster]
        
        st.markdown(f"""
        <div class="predict-result">
            <div class="predict-result-title">This customer belongs to</div>
            <div class="predict-result-value">{predicted_label} (Segment {predicted_cluster})</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
