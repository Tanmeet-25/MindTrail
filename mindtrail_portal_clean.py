# mindtrail_portal.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import io

# ----------------------- CONFIG ------------------------
st.set_page_config(page_title="MindTrail", page_icon="🧭", layout="wide")
DATA_PATH = Path("data/cleaned_dataset_week3.csv")  # if available, app will use this

# ----------------------- CSS (dark default, editable) ------------------------
def inject_custom_css(bg="#000000", text="#FFFFFF", accent="#2E4374"):
    css = f"""
    <style>
    /* existing styles… */

    /* buttons */
    .stButton>button {{
        background-color: {accent} !important;
        color: white !important;
        border-radius: 8px !important;
    }}

    /* cards */
    .career-card {{
        color: {text} !important;
        background: #111111 !important;
        border: 1px solid #333;
    }}

    /* input placeholders */
    input::placeholder {{
        color: {text} !important;
        opacity: 1 !important;
    }}

    /* ===== ADD THIS BLOCK BELOW ===== */
    /* Landing page buttons (Student / Developer portal) */
    .stButton>button[style*="width: 100%"] {{
        background-color: #000000 !important;  /* dark box */
        color: red !important;                 /* red font */
        border: 1px solid #2E4374 !important;  /* optional border */
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# apply default css (developer can change)
if "bg_color" not in st.session_state:
    st.session_state.bg_color = "#000000"
if "text_color" not in st.session_state:
    st.session_state.text_color = "#FFFFFF"
if "accent_color" not in st.session_state:
    st.session_state.accent_color = "#2E4374"

inject_custom_css(st.session_state.bg_color, st.session_state.text_color, st.session_state.accent_color)

# ----------------------- DATA LOADING ------------------------
@st.cache_data
def load_dataset():
    # If user has provided dataset file in data/ folder, read it.
    if DATA_PATH.exists():
        try:
            df = pd.read_csv(DATA_PATH)
            return df
        except Exception as e:
            st.warning(f"Could not read {DATA_PATH}: {e}. Using embedded sample dataset.")
    # fallback: small embedded sample (if actual file not present)
    csv = """O_score,C_score,E_score,A_score,N_score,Numerical Aptitude,Spatial Aptitude,Perceptual Aptitude,Abstract Reasoning,Verbal Reasoning,Career
5.45,8.67,3.45,5.34,4.23,9.23,4.56,6.78,7.89,6.12,Accountant
8.78,5.67,4.56,6.45,4.23,5.12,8.45,7.89,6.34,6.01,Graphic Designer
6.12,6.78,9.34,7.56,5.01,6.23,4.23,6.45,6.67,8.45,Salesperson
9.12,8.78,4.23,5.67,4.56,7.89,5.34,6.45,9.34,7.67,Research Scientist
6.45,7.56,5.67,9.12,4.23,5.34,4.01,6.23,5.78,8.67,Teacher
8.45,7.01,5.34,5.45,4.56,6.45,9.12,7.67,8.89,4.23,Architect
4.01,8.23,6.67,9.34,5.12,4.56,4.45,7.45,5.78,7.78,Nurse
8.78,7.89,5.67,6.01,4.67,8.45,4.23,5.34,9.23,6.45,Software Developer
8.78,7.45,5.34,8.45,6.01,4.56,4.01,4.23,6.67,9.23,Psychologist
6.78,6.45,8.45,7.67,4.56,3.45,3.12,9.34,3.67,4.23,Chef
"""
    df = pd.read_csv(io.StringIO(csv))
    return df

df_master = load_dataset()

# The numeric column names we care about (expected)
NUMERIC_COLS = [
    "O_score","C_score","E_score","A_score","N_score",
    "Numerical Aptitude","Spatial Aptitude","Perceptual Aptitude","Abstract Reasoning","Verbal Reasoning"
]

# Ensure columns exist; if not, try to detect the career col and numeric columns
def detect_career_column(df):
    for col in df.columns:
        if "career" in col.lower() or "profession" in col.lower():
            return col
    # fallback: last column
    return df.columns[-1]

CAREER_COL = detect_career_column(df_master)

# if some numeric cols missing, attempt to map first 10 numeric-like columns
available_numeric = [c for c in NUMERIC_COLS if c in df_master.columns]
if len(available_numeric) < 10:
    # try to detect numeric columns automatically
    numeric_candidates = df_master.select_dtypes(include=[np.number]).columns.tolist()
    # pick up to 10 of them
    available_numeric = numeric_candidates[:10]

# ----------------------- UTILS: similarity & top5 ------------------------
def get_top5_from_vector(vec):
    # vec: dict or pandas Series with numeric features matching available_numeric
    X = df_master[available_numeric].fillna(df_master[available_numeric].mean())
    user_vec = np.array([vec[c] for c in available_numeric], dtype=float)
    X_arr = X.values.astype(float)
    # Euclidean distances
    dists = np.linalg.norm(X_arr - user_vec.reshape(1, -1), axis=1)
    df_temp = df_master.copy()
    df_temp["_dist"] = dists
    df_temp = df_temp.sort_values("_dist").head(5)
    # convert distance -> similarity percentage (rough)
    maxd = dists.max() if dists.max() > 0 else 1.0
    df_temp["_sim_pct"] = (1 - df_temp["_dist"] / (maxd + 1e-9)) * 100
    return df_temp[[CAREER_COL, "_sim_pct"]].rename(columns={CAREER_COL: "Career", "_sim_pct": "Similarity (%)"})

# ----------------------- APTITUDE TEST QUESTIONS (from your list) ------------------------
# We'll store questions as list of dicts: {"q":..., "options": [...], "positive": index_of_option_that_indicates_high_trait}
# NOTE: "positive" indexing is used in the demo scoring mapping; this is a minimal mapping that groups Qs by trait.
QUESTIONS = [
    {"q":"1. A number increases by 20% and then decreases by 25%. What is the net percentage change?",
     "options":["A. –10%","B. –5%","C. 0%","D. +5%"], "positive": 1},
    {"q":"2. If 3x − 2 = 7 + 4x, what is x?",
     "options":["A. –9","B. 9","C. –5","D. 5"], "positive": 3},
    {"q":"3. The ratio of boys to girls in a class is 5:7. If there are 84 students, how many boys are there?",
     "options":["A. 35","B. 36","C. 30","D. 40"], "positive": 0},
    {"q":"4. A shopkeeper marks a product 40% above cost price and gives a 25% discount. What is the profit percentage?",
     "options":["A. 5%","B. 10%","C. 15%","D. 20%"], "positive": 1},
    {"q":"5. What is the sum of the first 20 even numbers?",
     "options":["A. 400","B. 420","C. 380","D. 440"], "positive": 0},
    {"q":"6. A cube is painted black on all six faces and cut into 64 equal cubes (4×4×4). How many small cubes have exactly one face painted?",
     "options":["A. 24","B. 36","C. 16","D. 48"], "positive": 0},
    {"q":"7. A figure is rotated 180°. Which property is not guaranteed to remain the same?",
     "options":["A. Area","B. Orientation","C. Shape","D. Size"], "positive": 1},
    {"q":"8. Which net cannot form a cube?",
     "options":["A. A T-shape with 4 squares in a line and 1 on each side","B. A cross of 5 squares with one attached","C. A zig-zag of 6 squares","D. A plus (+) shape of 5 squares with 1 extra"], "positive": 2},
    {"q":"9. Which view (top, front, side) will change most if a shape has large overhangs?",
     "options":["A. Top view","B. Front view","C. Side view","D. All views equally"], "positive": 0},
    {"q":"10. A shape is symmetrical horizontally and vertically. Which transformation leaves it unchanged?",
     "options":["A. 90° rotation","B. 180° rotation","C. Reflection only","D. Translation only"], "positive": 1},
    # ... continue adding more questions in the same format up to 50 (for brevity I add a representative set)
    {"q":"11. Which figure has the fewest intersections?",
     "options":["A. A triangle over a square","B. Two overlapping circles","C. A star inside a pentagon","D. A hexagon intersecting a triangle"], "positive": 1},
    {"q":"12. In a set of nearly identical symbols, one has a small tilt. What type of error is this?",
     "options":["A. Distortion error","B. Orientation error","C. Position error","D. Symmetry error"], "positive": 1},
    {"q":"13. If a pattern flashes briefly and a dot was at the 4th position from the left, which method best helps recall?",
     "options":["A. Chunking","B. Linear scanning","C. Reverse ordering","D. Logical grouping"], "positive": 0},
    {"q":"14. Which image pair is not perceptually equivalent?",
     "options":["A. Rotated versions","B. Reflected versions","C. Scaled versions","D. Distorted versions"], "positive": 3},
    {"q":"15. A figure has 12 hidden triangles. If one extra diagonal is added, how many new triangles are minimum added?",
     "options":["A. 1","B. 2","C. 3","D. 4"], "positive": 1},
    # ... some personality questions (openness, conscientiousness etc)
    {"q":"21. Choose the best synonym for “meticulous”:",
     "options":["A. Careless","B. Precise","C. Moderate","D. Quick"], "positive": 1},
    {"q":"22. Which sentence uses “sanction” correctly?",
     "options":["A. He sanctioned the ball to the roof.","B. The committee sanctioned the new policy.","C. She sanctioned quickly to the door.","D. They sanctioned the table."], "positive": 1},
    {"q":"26. High openness correlates most with:",
     "options":["A. Routine preference","B. Creativity","C. Conformity","D. Rule-following"], "positive": 1},
    {"q":"31. High conscientiousness predicts:",
     "options":["A. Poor planning","B. Impulsiveness","C. Reliability","D. Carelessness"], "positive": 2},
    {"q":"36. Extraverts usually prefer:",
     "options":["A. Solitary work","B. Large social gatherings","C. Isolation","D. Minimal interaction"], "positive": 1},
    {"q":"41. High agreeableness is associated with:",
     "options":["A. Aggression","B. Cooperation","C. Hostility","D. Conflict-seeking"], "positive": 1},
    {"q":"46. High neuroticism indicates:",
     "options":["A. Calmness","B. Worry-proneness","C. Emotional balance","D. Stability"], "positive": 1},
    {"q":"50. High emotional stability predicts:",
     "options":["A. Stress tolerance","B. Anxiety","C. Fear","D. Overthinking"], "positive": 0},
]
# If your full 50 questions differ, you can paste them into the QUESTIONS list above.

# ----------------------- QUESTION -> TRAIT MAPPING ------------------------
# This mapping determines which trait each question affects (demo mapping).
# Traits keys must match the dataset columns (or high-level OCEAN + aptitudes). We'll use:
TRAITS = ["O_score","C_score","E_score","A_score","N_score",
          "Numerical Aptitude","Spatial Aptitude","Perceptual Aptitude","Abstract Reasoning","Verbal Reasoning"]

# Map question index to trait index (simple grouping)
# We'll map early numeric/logic Qs -> Numerical Aptitude / Abstract Reasoning, others to personality.
mapping = {}
for i in range(len(QUESTIONS)):
    qidx = i+1
    # make reasonable grouping:
    if qidx <= 6:           # math/logic
        mapping[i] = "Numerical Aptitude"
    elif qidx <= 15:        # spatial/perceptual
        mapping[i] = "Spatial Aptitude"
    elif 16 <= qidx <= 25:  # verbal/personality
        mapping[i] = "Verbal Reasoning"
    elif 26 <= qidx <= 35:  # openness/conscientiousness
        mapping[i] = "O_score"
    elif 36 <= qidx <= 40:  # extraversion
        mapping[i] = "E_score"
    elif 41 <= qidx <= 45:  # agreeableness
        mapping[i] = "A_score"
    elif 46 <= qidx <= 50:  # neuroticism/stability
        mapping[i] = "N_score"
    else:
        mapping[i] = "Abstract Reasoning"

# ----------------------- APP NAVIGATION ------------------------
if "page" not in st.session_state:
    st.session_state.page = "landing"
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

# Top bar: show login/logout and navigation
top_cols = st.columns([1,6,1])
with top_cols[0]:
    st.write("")  # spacer
with top_cols[1]:
    st.markdown("<h1 style='margin:0; padding:0;'>🧭 MindTrail Portal</h1>", unsafe_allow_html=True)
with top_cols[2]:
    if st.session_state.logged_in:
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.session_state.page = "landing"

# ----------------------- LANDING PAGE ------------------------
if st.session_state.page == "landing":
    st.markdown("## Welcome — choose a portal")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🧑 Student Portal", key="student_portal", help="Take aptitude test or upload your scores"):
            st.session_state.page = "student"
            st.rerun()
    with c2:
        if st.button("🧑‍💻 Developer Portal", key="dev_portal", help="Login required"):
            st.session_state.page = "dev_login"
            st.rerun()

# ----------------------- STUDENT PORTAL ------------------------
elif st.session_state.page == "student":
    st.markdown("## Student Portal — Aptitude Test & Upload")
    mode = st.radio("Choose action:", ["Take Aptitude Test", "Upload scores (CSV)", "Back to Landing"], index=0)
    if mode == "Back to Landing":
        st.session_state.page = "landing"
        st.rerun()

    if mode == "Upload scores (CSV)":
        st.markdown("Upload a CSV with the following columns (or similar):")
        st.write(", ".join(available_numeric))
        uploaded = st.file_uploader("Upload CSV file (single row with scores or multiple rows):", type=["csv"])
        if uploaded is not None:
            try:
                uploaded_df = pd.read_csv(uploaded)
                st.success("File loaded.")
                st.write("Preview (first 5 rows):")
                st.dataframe(uploaded_df.head())
                # if multiple rows, let user pick one row
                if len(uploaded_df) > 1:
                    idx = st.number_input("Select row index to use for match (0-based):", min_value=0, max_value=len(uploaded_df)-1, value=0)
                    row = uploaded_df.iloc[int(idx)]
                else:
                    row = uploaded_df.iloc[0]
                # build vector mapping
                user_vec = {}
                for c in available_numeric:
                    try:
                        user_vec[c] = float(row.get(c, df_master[c].mean() if c in df_master.columns else 5.0))
                    except:
                        user_vec[c] = df_master[c].mean() if c in df_master.columns else 5.0
                # show top5
                top5 = get_top5_from_vector(user_vec)
                st.markdown("### Top 5 career matches")
                for _, r in top5.iterrows():
                    st.markdown(f"<div class='career-card'><strong>{r['Career']}</strong> — {r['Similarity (%)']:.1f}%</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Could not parse uploaded file: {e}")

    elif mode == "Take Aptitude Test":
        st.markdown("### Take the test — answer honestly. When done, click Submit to see Top 5 career matches.")
        # display questions with radio options
        answers = {}
        for i, q in enumerate(QUESTIONS):
            key = f"q_{i}"
            with st.container():
                st.markdown(f"{q['q']}")
                answers[key] = st.radio("", q["options"], key=key, index=0, horizontal=False)
        if st.button("Submit Test"):
            # scoring: initialize trait scores to zero
            trait_scores = {t: 0.0 for t in TRAITS}
            # For each answer, if selected option index matches question's 'positive' index, increment trait
            for i, q in enumerate(QUESTIONS):
                key = f"q_{i}"
                sel = answers.get(key)
                if sel is None:
                    continue
                # find index of selected option
                try:
                    sel_idx = q["options"].index(sel)
                except ValueError:
                    sel_idx = 0
                # positive? if equals positive index -> +1
                if sel_idx == q.get("positive", 0):
                    trait_name = mapping.get(i, "Abstract Reasoning")
                    # add a small amount; will normalize later
                    trait_scores[trait_name] = trait_scores.get(trait_name, 0) + 1.0
            # Normalize trait scores to a 0-10 scale roughly
            max_possible = max(1, max(trait_scores.values()))
            normalized = {}
            for t in TRAITS:
                normalized[t] = float(trait_scores.get(t, 0)) / max_possible * 10.0
                # if trait absent in available_numeric, still keep
            # ensure all available_numeric exist in normalized
            for c in available_numeric:
                if c not in normalized:
                    normalized[c] = 5.0  # neutral default
            # compute top5
            top5 = get_top5_from_vector(normalized)
            st.markdown("### Top 5 career matches")
            for _, r in top5.iterrows():
                st.markdown(f"<div class='career-card'><strong>{r['Career']}</strong> — {r['Similarity (%)']:.1f}%</div>", unsafe_allow_html=True)

    # navigation
    st.divider()
    if st.button("Back to Landing"):
        st.session_state.page = "landing"
        st.rerun()

# ----------------------- DEVELOPER LOGIN ------------------------
elif st.session_state.page == "dev_login":
    st.markdown("## Developer Portal — Login")
    uname = st.text_input("Username")
    pwd = st.text_input("Password", type="password")
    if st.button("Login"):
        if uname == "mindtrail" and pwd == "123":
            st.session_state.logged_in = True
            st.session_state.username = uname
            st.session_state.page = "developer"
            st.rerun()
        else:
            st.error("Invalid credentials.")

    if st.button("Back to Landing"):
        st.session_state.page = "landing"
        st.rerun()

# ----------------------- DEVELOPER PORTAL ------------------------
elif st.session_state.page == "developer":
    if not st.session_state.logged_in:
        st.session_state.page = "dev_login"
        st.rerun()
    st.markdown("## Developer Portal")
    st.markdown("Note: This portal shows the dataset only (no charts) and a simple background editor for UI preview.")

    st.markdown("#### Dataset (master)")
    st.dataframe(df_master)

    st.markdown("---")
    st.markdown("#### UI Background / Text Editor (preview)")
    with st.form("bg_form"):
        bg = st.color_picker("Background color", value=st.session_state.bg_color)
        text = st.color_picker("Text color", value=st.session_state.text_color)
        accent = st.color_picker("Accent (buttons) color", value=st.session_state.accent_color)
        submitted = st.form_submit_button("Apply")
        if submitted:
            st.session_state.bg_color = bg
            st.session_state.text_color = text
            st.session_state.accent_color = accent
            inject_custom_css(bg, text, accent)
            st.success("Updated preview styles.")

    if st.button("Logout (dev)"):
        st.session_state.logged_in = False
        st.session_state.page = "landing"
        st.rerun()

# ----------------------- fallback ------------------------
else:
    st.session_state.page = "landing"
    st.rerun()