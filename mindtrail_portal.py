import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# --------------------------- PAGE CONFIG ---------------------------
st.set_page_config(page_title="MindTrail", page_icon="🧭", layout="wide")

# --------------------------- CUSTOM STYLE ---------------------------
st.markdown("""
<style>
    body {
        background-color: #0e1117;
        color: #FFFFFF;
    }
    h1, h2, h3, h4, h5, h6, p, label, div {
        color: #FFFFFF !important;
    }
    .stButton>button {
        background-color: #2E4374;
        color: white;
        border-radius: 10px;
        padding: 0.6rem 1rem;
        font-weight: 600;
    }
    .career-box {
        background-color: #000000;
        color: white;
        padding: 1rem;
        border-radius: 12px;
        margin: 0.7rem 0;
        border: 1px solid #444;
        box-shadow: 0px 4px 8px rgba(255,255,255,0.1);
    }
    .career-box h4 {
        color: #FFD700;
    }
    .career-box p {
        color: #EAEAEA;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------- LOAD DATA ---------------------------
@st.cache_data
def load_data():
    # Load your internal dataset
    df = pd.read_csv(r"C:\Users\sonyn\OneDrive\Documents\GitHub\MindTrail\data\cleaned_dataset_week3.csv")
    return df

# --------------------------- APP LAYOUT ---------------------------
st.title("🧭 MindTrail Portal")
st.write("Select your role below to continue:")

portal = st.radio("Login as:", ["🎓 Student", "💻 Developer"], horizontal=True)

# =====================================================
# STUDENT PORTAL
# =====================================================
if portal == "🎓 Student":
    st.header("🎓 Student Career Recommendation Portal")
    st.markdown("Fill in your personality and aptitude details to find your best career match.")

    col1, col2 = st.columns(2)
    with col1:
        O = st.slider("Openness (O_score)", 0.0, 10.0, 5.0)
        C = st.slider("Conscientiousness (C_score)", 0.0, 10.0, 5.0)
        E = st.slider("Extraversion (E_score)", 0.0, 10.0, 5.0)
        A = st.slider("Agreeableness (A_score)", 0.0, 10.0, 5.0)
        N = st.slider("Neuroticism (N_score)", 0.0, 10.0, 5.0)
    with col2:
        NA = st.slider("Numerical Aptitude", 0.0, 10.0, 5.0)
        SA = st.slider("Spatial Aptitude", 0.0, 10.0, 5.0)
        PA = st.slider("Perceptual Aptitude", 0.0, 10.0, 5.0)
        AR = st.slider("Abstract Reasoning", 0.0, 10.0, 5.0)
        VR = st.slider("Verbal Reasoning", 0.0, 10.0, 5.0)

    if st.button("🔍 Get Top 5 Career Suggestions"):
        df = load_data()

        # Prepare user input
        user = pd.DataFrame({
            "O_score": [O], "C_score": [C], "E_score": [E],
            "A_score": [A], "N_score": [N],
            "Numerical Aptitude": [NA], "Spatial Aptitude": [SA],
            "Perceptual Aptitude": [PA], "Abstract Reasoning": [AR],
            "Verbal Reasoning": [VR]
        })

        # Compute Euclidean distance between user and dataset
        df['distance'] = ((df.iloc[:, :-1] - user.iloc[0])**2).sum(axis=1)**0.5

        # Normalize distances to similarity %
        df['similarity'] = (1 - df['distance'] / df['distance'].max()) * 100

        top5 = df.sort_values(by='similarity', ascending=False).head(5)[['Career', 'similarity']]

        st.markdown("### 🏆 Your Top 5 Career Matches")
        for idx, row in top5.iterrows():
            st.markdown(f"""
                <div class='career-box'>
                    <h4>{row['Career']}</h4>
                    <p>Match Confidence: {row['similarity']:.2f}%</p>
                </div>
            """, unsafe_allow_html=True)

# =====================================================
# DEVELOPER PORTAL
# =====================================================
elif portal == "💻 Developer":
    st.header("💻 Developer Dashboard (Team Only)")

    # Simple login system
    username = st.text_input("Enter username:")
    password = st.text_input("Enter password:", type="password")

    team_members = {
        "tanmeet": "design123",
        "syna": "data123",
        "vardaan": "ml123"
    }

    if st.button("🔓 Login"):
        if username.lower() in team_members and password == team_members[username.lower()]:
            st.success(f"Welcome, {username.capitalize()}!")
            df = load_data()

            st.subheader("📁 Dataset Overview")
            st.dataframe(df.head())

            st.subheader("📊 Career Distribution")
            fig = px.histogram(df, x="Career", title="Career Frequency", color_discrete_sequence=["#FFD700"])
            fig.update_layout(xaxis={'categoryorder':'total descending'}, height=500, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("📈 Correlation Heatmap")
            corr = df.iloc[:, :-1].corr()
            st.dataframe(corr.style.background_gradient(cmap="Greys"))
        else:
            st.error("Invalid credentials. Access denied.")
