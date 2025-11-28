import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# ---------------- CONFIG ----------------
st.set_page_config(page_title="MindTrail", page_icon="🧭", layout="wide")

DATASET_PATH = "data/cleaned_dataset.csv"

# -------------- DARK THEME --------------
st.markdown("""
<style>
.stApp {
    background-color: #000000 !important;
    color: white !important;
}
.question {
    font-weight: 600;
    font-size: 18px;
    margin-bottom: 6px;
}
.career-card {
    background: #111111;
    padding: 16px;
    border-radius: 10px;
    border: 1px solid #333;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATASET ----------------
@st.cache_data
def load_dataset():
    try:
        df = pd.read_csv(DATASET_PATH)
        return df
    except:
        return None

def detect_career_column(df):
    for col in df.columns:
        if "career" in col.lower() or "profession" in col.lower():
            return col
    return df.columns[-1]

EXPECTED_ORDER = [
    "O_score","C_score","E_score","A_score","N_score",
    "Numerical Aptitude","Spatial Aptitude","Perceptual Aptitude",
    "Abstract Reasoning","Verbal Reasoning"
]

def compute_top5(profile, df):
    match_cols = [c for c in EXPECTED_ORDER if c in df.columns]
    numeric = df[match_cols].astype(float)

    profile = np.array(profile[:len(match_cols)])
    distances = np.linalg.norm(numeric - profile, axis=1)

    df = df.copy()
    df["distance"] = distances
    df["similarity_pct"] = (1 - distances / distances.max()) * 100

    return df.sort_values("distance").head(5)

# ---------------- 50 QUESTIONS ----------------
SECTIONS = {
    "Numerical Aptitude": [
        ("A number increases by 20% and then decreases by 25%. What is the net percentage change?",
         ["–10%","–5%","0%","+5%"]),
        ("If 3x − 2 = 7 + 4x, what is x?",
         ["–9","9","–5","5"]),
        ("The ratio of boys to girls is 5:7. If there are 84 students, how many boys?",
         ["35","36","30","40"]),
        ("Marked price = +40%, Discount = –25%. Profit %?",
         ["5%","10%","15%","20%"]),
        ("Sum of first 20 even numbers?",
         ["400","420","380","440"])
    ],
    "Spatial Aptitude": [
        ("Cube painted & cut into 64. How many have 1 face painted?",
         ["24","36","16","48"]),
        ("After a 180° rotation, which property may change?",
         ["Area","Orientation","Shape","Size"]),
        ("Which net CANNOT form a cube?",
         ["T-shape","Cross shape","Zig-zag","Plus shape"]),
        ("Which view changes most with overhangs?",
         ["Top","Front","Side","All equally"]),
        ("Symmetry both ways. Which keeps it same?",
         ["90° rotation","180° rotation","Reflection","Translation"])
    ],
    "Perceptual Aptitude": [
        ("Which figure has the fewest intersections?",
         ["Triangle+Square","Two circles","Star+Pentagon","Hexagon+Triangle"]),
        ("A tilted symbol among 5 identical. Error type?",
         ["Distortion","Orientation","Position","Symmetry"]),
        ("Pattern flashed briefly. Best recall method?",
         ["Chunking","Linear scan","Reverse order","Grouping"]),
        ("Which is NOT perceptually equivalent?",
         ["Rotated","Reflected","Scaled","Distorted"]),
        ("Figure has 12 triangles. Adding 1 diagonal adds?",
         ["1","2","3","4"])
    ],
    "Abstract Reasoning": [
        ("Pattern 3→4→6→9→?", ["12","13","15","18"]),
        ("Most complex rule?", ["Shading","Size ↑","Rotate+Count ↑","Flip"]),
        ("▲→■ add 1 side; ■→⬟ add 2 sides. Next?",
         ["Octagon","Pentagon","Heptagon","Nonagon"]),
        ("If first rotates 45°, 4th rotates:", ["90°","135°","180°","225°"]),
        ("Circle:Sphere :: Square: ?", ["Cube","Pyramid","Cylinder","Cone"])
    ],
    "Verbal Reasoning": [
        ("Synonym of meticulous:", ["Careless","Precise","Moderate","Quick"]),
        ("Correct use of 'sanction':",
         ["He sanctioned the ball","Committee sanctioned policy",
          "She sanctioned quickly","They sanctioned table"]),
        ("Correct analogy:",
         ["Author:Book","Doctor:Disease","Knife:Cut","Teacher:Student"]),
        ("Conclusion: All programmers are logical; Some logical are introverts.",
         ["All introverts are programmers",
          "Some programmers may be introverts",
          "No programmer is introvert",
          "Programmers are extroverts"]),
        ("Opposite of ambiguous:", ["Vague","Clear","Blunt","Minor"])
    ],
    "Openness": [
        ("High openness correlates with:", ["Routine","Creativity","Conformity","Rules"]),
        ("High openness activity:", ["Avoid new","Abstract art","Repeat","Comfort zone"]),
        ("High openness low in:", ["Imagination","Curiosity","Resistance","Creativity"]),
        ("Philosophical debates show:", ["Low","Medium","High","None"]),
        ("Trait with openness:", ["Narrow","Traditional","Curious","Rigid"])
    ],
    "Conscientiousness": [
        ("High conscientious predicts:", ["Poor plan","Impulsive","Reliable","Careless"]),
        ("Least likely:", ["Meet deadlines","Forget tasks","Organize","Check details"]),
        ("Reflects conscientious:", ["Sloppy","Schedules","Skipping","Luck"]),
        ("They usually:", ["Procrastinate","Systematic","Ignore","Chaos"]),
        ("Correlates with:", ["Instability","Accuracy","Lazy","Disorder"])
    ],
    "Extraversion": [
        ("Extraverts prefer:", ["Solitary","Large groups","Isolation","Minimal talk"]),
        ("Highly extraverted:", ["Drained","Avoid attention","Social energy","Silence"]),
        ("Linked to:", ["Assertive","Withdrawn","Low social","Quiet"]),
        ("NOT trait:", ["Enthusiasm","Talkative","Social","Reserved"]),
        ("Extraverts gain energy from:", ["Alone","Quiet","Interaction","Thinking"])
    ],
    "Agreeableness": [
        ("High agreeableness:", ["Aggression","Cooperation","Hostility","Conflict"]),
        ("Highly agreeable person:", ["Insults","Harmony","Fights","Manipulate"]),
        ("They usually:", ["Trust","Doubt","Dislike kindness","Arguments"]),
        ("Contradicts agreeableness:", ["Empathy","Altruism","Hostility","Warmth"]),
        ("Predicts:", ["Rudeness","Compromise","Harshness","Domination"])
    ],
    "Neuroticism": [
        ("High neuroticism:", ["Calm","Worry","Balance","Stable"]),
        ("Low emotional stability:", ["Calm","Panic","Handle well","Recover"]),
        ("Signals high neuroticism:", ["Recover fast","Rare worry","Mood swings","Resilience"]),
        ("Opposite of stability:", ["Openness","Extraversion","Neuroticism","Agreeableness"]),
        ("High stability predicts:", ["Tolerance","Anxiety","Fear","Overthinking"])
    ]
}

def score_answer(option, options):
    idx = ["A","B","C","D"][:len(options)].index(option)
    return ((len(options) - 1 - idx) / (len(options)-1)) * 4

# ---------------- HOME ----------------
st.title("🧭 MindTrail")

choice = st.radio("Choose Portal", ["Student Portal", "Developer Portal"], horizontal=True)

# ---------------- STUDENT PORTAL ----------------
if choice == "Student Portal":
    st.header("🎓 Student Portal")

    mode = st.radio("Choose Action", ["Take Aptitude Test", "Upload Scores", "Get Career Matches"])

    # TAKE TEST
    if mode == "Take Aptitude Test":
        if "answers" not in st.session_state:
            st.session_state["answers"] = {}

        st.subheader("Aptitude Test — 50 Questions")
        st.write("Answer all questions below:")

        for section, qs in SECTIONS.items():
            st.subheader(section)
            for i, (q, opts) in enumerate(qs):
                key = f"{section}_{i}"
                st.markdown(f"<div class='question'>{q}</div>", unsafe_allow_html=True)
                ans = st.radio("Choose:", ["A","B","C","D"][:len(opts)], key=key)
                st.session_state["answers"][key] = ans

        if st.button("Submit Test"):
            profile = []
            for section, qs in SECTIONS.items():
                total = 0
                for i, (q, opts) in enumerate(qs):
                    key = f"{section}_{i}"
                    ans = st.session_state["answers"][key]
                    total += score_answer(ans, opts)
                profile.append((total / (len(qs)*4)) * 10)

            df = load_dataset()
            if df is None:
                st.error("Dataset missing. Ask developer to upload.")
            else:
                profile_vector = profile[:10]
                top5 = compute_top5(profile_vector, df)
                career_col = detect_career_column(df)

                st.subheader("Top 5 Careers")
                for _, row in top5.iterrows():
                    st.markdown(f"""
                    <div class='career-card'>
                        <b>{row[career_col]}</b><br>
                        Match: {row['similarity_pct']:.2f}%
                    </div>
                    """, unsafe_allow_html=True)

    # UPLOAD SCORES
    if mode == "Upload Scores":
        st.subheader("Upload your CSV/XLSX with 10 columns:")
        st.write(", ".join(EXPECTED_ORDER))

        up = st.file_uploader("Upload", type=["csv", "xlsx"])
        if up:
            if up.name.endswith(".csv"):
                df_user = pd.read_csv(up)
            else:
                df_user = pd.read_excel(up)

            row = df_user.iloc[0]
            profile = [float(row[c]) for c in EXPECTED_ORDER]

            df = load_dataset()
            if df is None:
                st.error("Developer has not uploaded dataset yet.")
            else:
                top5 = compute_top5(profile, df)
                career_col = detect_career_column(df)

                st.subheader("Top Careers")
                for _, row in top5.iterrows():
                    st.markdown(f"""
                    <div class='career-card'>
                        <b>{row[career_col]}</b><br>
                        Match: {row['similarity_pct']:.2f}%
                    </div>
                    """, unsafe_allow_html=True)

    # GET CAREER MATCHES
    if mode == "Get Career Matches":
        st.info("Please upload scores OR take the test.")

# ---------------- DEVELOPER PORTAL ----------------
if choice == "Developer Portal":
    st.header("🛠 Developer Login")

    if "dev_login" not in st.session_state:
        st.session_state["dev_login"] = False

    if not st.session_state["dev_login"]:
        user = st.text_input("Username")
        pwd = st.text_input("Password", type="password")

        if st.button("Login"):
            if user == "mindtrail" and pwd == "123":
                st.session_state["dev_login"] = True
                st.success("Logged in successfully.")
                st.rerun()
            else:
                st.error("Incorrect login.")
    else:
        st.subheader("Upload Main Dataset (CSV only)")
        up = st.file_uploader("Upload dataset", type=["csv"])

        if up:
            df = pd.read_csv(up)
            Path("data").mkdir(exist_ok=True)
            df.to_csv(DATASET_PATH, index=False)
            st.success("Dataset updated successfully!")

        if st.button("Logout"):
            st.session_state["dev_login"] = False
            st.rerun()
