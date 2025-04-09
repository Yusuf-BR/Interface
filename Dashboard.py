# ✅ Final Clean Enterprise Dashboard — Region + Industry Filters

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

OUTPUT_FOLDER = Path("outputs")
UPLOADED_CSV_PATH = OUTPUT_FOLDER / "uploaded_dataset.csv"

# === Utility Functions ===
def get_summary(df):
    conversion_threshold = df['Conversion Probability'].mean()
    successful_leads = df[df['Conversion Probability'] > conversion_threshold].shape[0]

    return {
        "total_leads": df.shape[0],
        "avg_conversion": round(df['Conversion Probability'].mean() * 100, 2),
        "hot_leads": df[df['Conversion Probability'] > df['Conversion Probability'].mean() * 1.2].shape[0],
        "success_rate": round(successful_leads / df.shape[0] * 100, 2) if df.shape[0] > 0 else 0,
        "max_conversion": round(df['Conversion Probability'].max() * 100, 2),
        "min_conversion": round(df['Conversion Probability'].min() * 100, 2),
        "region_summary": df['Region'].value_counts().idxmax() if 'Region' in df.columns else 'N/A',
        "industry_summary": df['Industry'].value_counts().idxmax() if 'Industry' in df.columns else 'N/A'
    }

def apply_transparent_template(fig):
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#333'),
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(bgcolor='rgba(0,0,0,0)')
    )
    return fig

# === Main Dashboard Function ===
def show_dashboard():
    st.markdown("""
        <h1 style='text-align: center; color: #4A90E2; margin-bottom: 40px;'>
            📊 Executive Dashboard
        </h1>
    """, unsafe_allow_html=True)

    if not UPLOADED_CSV_PATH.exists():
        st.warning("⚠️ Please upload a dataset first.")
        return

    df = pd.read_csv(UPLOADED_CSV_PATH)

    # === Region Selector ===
    if 'Region' in df.columns:
        region_options = ['All Regions'] + sorted(df['Region'].dropna().unique().tolist())
        selected_region = st.selectbox("🌍 Select Region:", region_options)
        if selected_region != 'All Regions':
            df = df[df['Region'] == selected_region]

    # === Industry Selector ===
    if 'Industry' in df.columns:
        industry_options = ['All Industries'] + sorted(df['Industry'].dropna().unique().tolist())
        selected_industry = st.selectbox("🏭 Select Industry:", industry_options)
        if selected_industry != 'All Industries':
            df = df[df['Industry'] == selected_industry]

    summary = get_summary(df)

    # === KPI CARDS (Balanced Layout) ===
    kpis = [
        ("📈 Total Leads", summary['total_leads']),
        ("🎯 Avg. Conversion %", f"{summary['avg_conversion']}%"),
        ("🔥 Hot Leads", summary['hot_leads']),
        ("✅ Success Rate", f"{summary['success_rate']}%")
    ]

    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    cols = [col1, col2, col3, col4]

    for col, (label, value) in zip(cols, kpis):
        col.markdown(f"""
            <div style='background-color:#f0f2f6; padding:30px; border-radius:15px; text-align:center; 
                        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); height: 150px; display: flex; 
                        flex-direction: column; justify-content: center;'>
                <div style='font-size: 18px; margin-bottom: 8px; color: #222;'>{label}</div>
                <div style='font-size: 28px; font-weight: bold; color: #4A90E2;'>{value}</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # === Global Summary Text ===
    st.subheader("🌍 Global Summary")
    st.markdown(f"- **Max Conversion Probability:** {summary['max_conversion']}%")
    st.markdown(f"- **Min Conversion Probability:** {summary['min_conversion']}%")
    st.markdown(f"- **Top Performing Region:** {summary['region_summary']}")
    st.markdown(f"- **Top Performing Industry:** {summary['industry_summary']}")

    st.markdown("---")

    # === Visualizations ===
    st.subheader("Conversion Probability Distribution")
    fig = apply_transparent_template(px.histogram(df, x='Conversion Probability', nbins=20))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("Hot / Warm / Cold Leads Breakdown")
    df['Lead Category'] = df['Conversion Probability'].apply(
        lambda prob: 'Hot' if prob > df['Conversion Probability'].mean() * 1.2 else
        ('Warm' if prob > df['Conversion Probability'].mean() * 0.95 else 'Cold')
    )
    lead_counts = df['Lead Category'].value_counts().reset_index()
    lead_counts.columns = ['Lead Category', 'count']
    fig2 = apply_transparent_template(px.pie(lead_counts, names='Lead Category', values='count', hole=0.4))
    fig2.update_traces(textinfo='percent+label')
    st.plotly_chart(fig2, use_container_width=True)

    if 'Status' in df.columns:
        st.markdown("---")
        st.subheader("Success / Failure Breakdown")
        status_counts = df['Status'].value_counts().reset_index()
        status_counts.columns = ['Status', 'count']
        fig3 = apply_transparent_template(px.bar(status_counts, x='Status', y='count', color='Status', text='count'))
        st.plotly_chart(fig3, use_container_width=True)

    if 'Region' in df.columns:
        st.markdown("---")
        st.subheader("Top Performing Regions")
        region_counts = df['Region'].value_counts().reset_index()
        region_counts.columns = ['Region', 'count']
        fig4 = apply_transparent_template(px.bar(region_counts, x='Region', y='count', text='count', color='Region'))
        st.plotly_chart(fig4, use_container_width=True)

    if 'Industry' in df.columns:
        st.markdown("---")
        st.subheader("Top Performing Industries")
        industry_counts = df['Industry'].value_counts().reset_index()
        industry_counts.columns = ['Industry', 'count']
        fig5 = apply_transparent_template(px.bar(industry_counts, x='Industry', y='count', text='count', color='Industry'))
        st.plotly_chart(fig5, use_container_width=True)

    if 'Job title' in df.columns:
        st.markdown("---")
        st.subheader("Top Performing Job Titles")
        job_counts = df['Job title'].value_counts().reset_index()
        job_counts.columns = ['Job title', 'count']
        fig6 = apply_transparent_template(px.bar(job_counts, x='Job title', y='count', text='count', color='Job title'))
        st.plotly_chart(fig6, use_container_width=True)

    st.success("✅ Enterprise Dashboard with Region + Industry filters generated successfully!")
