import warnings
warnings.filterwarnings("ignore")

import time
import datetime

import streamlit as st
import extra_streamlit_components as stx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder

SESSION_SECONDS = 600  # 10 minutes

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sports Injury Intelligence Dashboard",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Colour palette ─────────────────────────────────────────────────────────────
PRIMARY   = "#1f4e79"
SECONDARY = "#2e75b6"
ACCENT    = "#ed7d31"
SUCCESS   = "#70ad47"
DANGER    = "#c00000"
BG_CARD   = "#f0f4f8"

# ── NEISS Code Mappings (from official NEISS_MAPPING table) ───────────────────
SEX_MAP = {0: "Unknown", 1: "Male", 2: "Female"}

DISPOSITION_MAP = {
    0: "No Injury",
    1: "Treated & Released",
    2: "Treated & Transferred",
    4: "Hospitalized/Admitted",
    5: "Held for Observation",
    6: "Left Without Being Seen",
    8: "Fatality (incl. DOA)",
    9: "Unknown/Not Stated"
}

BODY_PART_MAP = {
    0:  "Internal",
    30: "Shoulder",
    31: "Upper Trunk",
    32: "Elbow",
    33: "Lower Arm",
    34: "Wrist",
    35: "Knee",
    36: "Lower Leg",
    37: "Ankle",
    38: "Pubic Region",
    75: "Head",
    76: "Face",
    77: "Eyeball",
    78: "Upper Trunk (old)",
    79: "Lower Trunk",
    80: "Upper Arm",
    81: "Upper Leg",
    82: "Hand",
    83: "Foot",
    84: "25–50% of Body",
    85: "All Parts of Body",
    86: "Other (old)",
    87: "Not Stated/Unknown",
    88: "Mouth",
    89: "Neck",
    90: "Lower Arm (old)",
    91: "Lower Leg (old)",
    92: "Finger",
    93: "Toe",
    94: "Ear",
}

DIAGNOSIS_MAP = {
    41: "Ingestion",
    42: "Aspiration",
    46: "Burn – Electrical",
    47: "Burn – Not Specified",
    48: "Burn – Scald",
    49: "Burn – Chemical",
    50: "Amputation",
    51: "Burns – Thermal",
    52: "Concussion",
    53: "Contusions / Abrasions",
    54: "Crushing",
    55: "Dislocation",
    56: "Foreign Body",
    57: "Fracture",
    58: "Hematoma",
    59: "Laceration",
    60: "Dental Injury",
    61: "Nerve Damage",
    62: "Internal Injury",
    63: "Puncture",
    64: "Strain / Sprain",
    65: "Anoxia",
    66: "Hemorrhage",
    67: "Electric Shock",
    68: "Poisoning",
    69: "Submersion / Drowning",
    70: "Other",
    71: "Other",
    72: "Avulsion",
    73: "Radiation",
    74: "Dermatitis / Conjunctivitis",
}

LOCATION_MAP = {
    0: "Unknown",
    1: "Home",
    2: "Farm",
    3: "Apartment",
    4: "Street / Highway",
    5: "Public Property",
    6: "Mobile Home / RV",
    7: "Industrial",
    8: "School",
    9: "Sports / Recreation",
}

MONTH_ORDER = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

DOW_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday",
             "Friday", "Saturday", "Sunday"]

# ── Password Gate ──────────────────────────────────────────────────────────────
def check_password():
    # Fast path: session already authenticated — skip CookieManager entirely
    if st.session_state.get("authenticated"):
        if time.time() - st.session_state.get("auth_time", 0) < SESSION_SECONDS:
            return True
        st.session_state.authenticated = False  # timed out mid-session

    # Only instantiate CookieManager when we actually need it
    cm = stx.CookieManager(key="auth_mgr")

    # Check browser cookie (survives page refresh)
    auth_ts = cm.get("sports_dash_auth")
    if auth_ts:
        try:
            if time.time() - float(auth_ts) < SESSION_SECONDS:
                st.session_state.authenticated = True
                st.session_state.auth_time = float(auth_ts)
                return True
        except (ValueError, TypeError):
            cm.delete("sports_dash_auth", key="del_bad")

    # ── Login form ─────────────────────────────────────────────────────────
    st.markdown("""
    <style>
    [data-testid="stSidebar"]        { display: none; }
    [data-testid="stCustomComponentV1"] { display: none !important; height: 0 !important; }
    .block-container { max-width: 460px; padding-top: 3rem !important; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center;padding:12px 0 16px 0;">
        <div style="font-size:2.8rem;">⚕️</div>
        <h1 style="color:#1f4e79;font-size:1.8rem;margin:6px 0 2px 0;">
            Sports Injury Intelligence
        </h1>
        <p style="color:#2e75b6;font-size:0.95rem;margin:0 0 8px 0;">
            Healthcare Analytics Dashboard
        </p>
        <p style="color:#666;font-size:0.82rem;line-height:1.5;margin:0;">
            A consultant-grade analysis of sports-related emergency room injuries
            based on NEISS data.
        </p>
    </div>
    <hr style="border:none;border-top:1px solid #e0e0e0;margin:16px 0;">
    """, unsafe_allow_html=True)

    with st.form("login_form"):
        pwd = st.text_input("Dashboard Password", type="password", placeholder="Enter password…")
        submitted = st.form_submit_button("Login", use_container_width=True, type="primary")
    if submitted:
        if pwd == "msba382":
            now = time.time()
            st.session_state.authenticated = True
            st.session_state.auth_time = now
            expires = datetime.datetime.now() + datetime.timedelta(seconds=SESSION_SECONDS)
            cm.set("sports_dash_auth", str(now), expires_at=expires, key="set_auth")
            st.rerun()
        else:
            st.error("Incorrect password. Please try again.")

    st.markdown("""
    <div style="text-align:center;margin-top:12px;color:#aaa;font-size:0.75rem;">
        MSBA382 – Healthcare Analytics | Individual Project | 2025
    </div>
    """, unsafe_allow_html=True)
    return False


# ── Data Loading ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_excel("neiss_time_series_classified.xlsx")

    df["Sex_Label"]         = df["Sex"].map(SEX_MAP).fillna("Unknown")
    df["Disposition_Label"] = df["Disposition"].map(DISPOSITION_MAP).fillna("Unknown")
    df["Body_Part_Label"]   = df["Body_Part"].map(BODY_PART_MAP).fillna(df["Body_Part"].astype(str))
    df["Diagnosis_Label"]   = df["Diagnosis"].map(DIAGNOSIS_MAP).fillna(df["Diagnosis"].astype(str))
    df["Location_Label"]    = df["Location"].map(LOCATION_MAP).fillna("Unknown")

    dt = pd.to_datetime(df["Treatment_Date"])
    df["Year"]       = dt.dt.year
    df["Month_Num"]  = dt.dt.month
    df["Month_Name"] = dt.dt.strftime("%b")
    df["Quarter"]    = dt.dt.quarter
    df["DayOfWeek"]  = dt.dt.day_name()

    bins   = [-1, 12, 17, 25, 35, 50, 65, 200]
    labels = ["Child (0–12)", "Teen (13–17)", "Young Adult (18–25)",
              "Adult (26–35)", "Middle-Aged (36–50)", "Senior (51–65)", "Elderly (65+)"]
    df["Age_Group"] = pd.cut(df["Age"], bins=bins, labels=labels)

    df["Hospitalized"] = (df["Disposition"] == 4).astype(int)
    df["Severe"]       = df["Disposition"].isin([2, 4, 8]).astype(int)

    sports = df[df["Sports_Related"] == "Y"].copy()
    sports["Sport_Type"] = sports["Sport_Type"].str.strip().str.title()

    return df, sports


# ── KPI card helper ────────────────────────────────────────────────────────────
def kpi_card(label, value, note="", color=PRIMARY):
    return f"""
    <div style="background:{BG_CARD};border-left:5px solid {color};
                border-radius:8px;padding:14px 18px;height:100%;">
        <p style="margin:0;font-size:0.75rem;color:#888;font-weight:700;
                  text-transform:uppercase;letter-spacing:0.06em;">{label}</p>
        <h2 style="margin:4px 0 2px 0;color:{color};font-size:1.9rem;line-height:1.1;">{value}</h2>
        <p style="margin:0;font-size:0.78rem;color:#999;">{note}</p>
    </div>"""


# ── PAGE 1 · Overview ──────────────────────────────────────────────────────────
def page_overview(df, sports):
    st.title("📊 Overview")
    st.caption("High-level summary of all ER visits and the sports injury burden.")
    st.markdown("---")

    year_options = ["All Years"] + sorted(df["Year"].unique().tolist())
    sel_year = st.selectbox("Year", year_options, index=0, key="overview_year")

    if sel_year == "All Years":
        df_v     = df
        sports_v = sports
        period   = "2019–2025"
    else:
        df_v     = df[df["Year"] == sel_year]
        sports_v = sports[sports["Year"] == sel_year]
        period   = str(sel_year)

    st.markdown("<br>", unsafe_allow_html=True)

    total     = len(df_v)
    s_n       = len(sports_v)
    s_pct     = s_n / total * 100
    top_sport = sports_v["Sport_Type"].value_counts().idxmax()
    hosp_rate = sports_v["Hospitalized"].mean() * 100
    avg_age   = sports_v["Age"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(kpi_card("Total ER Visits",     f"{total:,}",          period,                    PRIMARY),   unsafe_allow_html=True)
    c2.markdown(kpi_card("Sports-Related",       f"{s_n:,}",            f"{s_pct:.1f}% of visits", SECONDARY), unsafe_allow_html=True)
    c3.markdown(kpi_card("Most Injured Sport",   top_sport.title(),     "by case count",           ACCENT),    unsafe_allow_html=True)
    c4.markdown(kpi_card("Hospitalization Rate", f"{hosp_rate:.1f}%",  "sports injuries only",    DANGER),    unsafe_allow_html=True)
    c5.markdown(kpi_card("Avg Patient Age",      f"{avg_age:.1f} yrs", "sports-related cases",    SUCCESS),   unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)

    with col_l:
        fig = px.pie(
            names=["Sports-Related", "Non-Sports"],
            values=[s_n, total - s_n],
            hole=0.55,
            color_discrete_sequence=[ACCENT, "#d0d7de"],
            title=f"Proportion of Sports-Related ER Visits ({period})"
        )
        fig.update_traces(textposition="outside", textinfo="percent+label")
        fig.update_layout(showlegend=False, height=340, margin=dict(t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        disp = sports_v["Disposition_Label"].value_counts().reset_index()
        disp.columns = ["Outcome", "Count"]
        fig2 = px.bar(
            disp.sort_values("Count"), x="Count", y="Outcome", orientation="h",
            color="Count", color_continuous_scale=["#bdd7ee", PRIMARY],
            title=f"Sports Injury Outcomes – Disposition ({period})", text="Count"
        )
        fig2.update_traces(textposition="outside")
        fig2.update_layout(coloraxis_showscale=False, height=340, yaxis_title="",
                           margin=dict(t=50, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    if sel_year == "All Years":
        ann_spo = sports.groupby("Year").size().reset_index(name="Count")
        ann_spo["Type"] = "Sports-Related"
        ann_non = df[df["Sports_Related"] == "N"].groupby("Year").size().reset_index(name="Count")
        ann_non["Type"] = "Non-Sports"
        annual = pd.concat([ann_spo, ann_non])
        fig3 = px.line(
            annual, x=annual["Year"].astype(str), y="Count", color="Type",
            markers=True,
            color_discrete_map={"Sports-Related": ACCENT, "Non-Sports": SECONDARY},
            title="Annual ER Visits Trend (2019–2025)",
            labels={"x": "Year", "Count": "ER Visits", "Type": ""}
        )
        fig3.update_layout(
            height=360, xaxis_title="Year",
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
    else:
        def monthly_counts(frame, label):
            g = frame.groupby(["Month_Num", "Month_Name"]).size().reset_index(name="Count")
            g["Type"] = label
            return g.sort_values("Month_Num")

        monthly = pd.concat([
            monthly_counts(sports_v, "Sports-Related"),
            monthly_counts(df_v[df_v["Sports_Related"] == "N"], "Non-Sports")
        ])
        fig3 = px.line(
            monthly, x="Month_Name", y="Count", color="Type",
            markers=True,
            color_discrete_map={"Sports-Related": ACCENT, "Non-Sports": SECONDARY},
            title=f"Monthly ER Visits: Sports vs. Non-Sports ({sel_year})",
            labels={"Month_Name": "Month", "Count": "ER Visits", "Type": ""}
        )
        fig3.update_layout(
            height=360,
            xaxis=dict(categoryorder="array", categoryarray=MONTH_ORDER),
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )

    st.plotly_chart(fig3, use_container_width=True)


# ── PAGE 2 · Sports Danger Index ───────────────────────────────────────────────
def page_sports_danger(df, sports):
    st.title("⚽ Sports Danger Index")
    st.caption("Which sports send the most people to the ER — and how serious are those injuries?")
    st.markdown("---")

    st.sidebar.markdown("### Filters")
    sex_sel = st.sidebar.multiselect("Gender", ["Male", "Female"], default=["Male", "Female"])
    all_ages = sorted(sports["Age_Group"].dropna().astype(str).unique())
    age_sel  = st.sidebar.multiselect("Age Group", all_ages, default=all_ages)

    fil = sports[
        sports["Sex_Label"].isin(sex_sel) &
        sports["Age_Group"].astype(str).isin(age_sel)
    ]

    counts = fil["Sport_Type"].value_counts().reset_index()
    counts.columns = ["Sport", "Injuries"]
    hosp   = fil.groupby("Sport_Type")["Hospitalized"].mean().reset_index()
    hosp.columns = ["Sport", "Hosp_Rate"]
    stats  = counts.merge(hosp, on="Sport")
    stats["Hosp_Pct"] = (stats["Hosp_Rate"] * 100).round(1)

    col1, col2 = st.columns(2)

    with col1:
        top15 = stats.head(15).sort_values("Injuries")
        fig = px.bar(
            top15, x="Injuries", y="Sport", orientation="h",
            color="Injuries", color_continuous_scale=["#bdd7ee", PRIMARY],
            title="Top 15 Sports by Injury Count", text="Injuries"
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(coloraxis_showscale=False, height=520, yaxis_title="",
                          margin=dict(t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.scatter(
            stats, x="Injuries", y="Hosp_Pct",
            size="Injuries", color="Hosp_Pct",
            hover_name="Sport", text="Sport",
            color_continuous_scale=["#70ad47", "#ed7d31", "#c00000"],
            title="Injury Volume vs. Hospitalization Rate",
            labels={"Injuries": "# Injuries", "Hosp_Pct": "Hospitalization Rate (%)"}
        )
        fig2.update_traces(textposition="top center", textfont_size=9)
        fig2.update_layout(height=520, coloraxis_showscale=False, margin=dict(t=50, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    # Treemap
    st.subheader("Injury Distribution Treemap")
    fig3 = px.treemap(
        stats, path=["Sport"], values="Injuries",
        color="Hosp_Pct",
        color_continuous_scale=["#70ad47", "#ed7d31", "#c00000"],
        title="Size = injury count · Colour = hospitalization rate (%)",
        color_continuous_midpoint=stats["Hosp_Pct"].median()
    )
    fig3.update_layout(height=460, margin=dict(t=40, b=10))
    st.plotly_chart(fig3, use_container_width=True)

    # Severity heatmap
    st.subheader("Outcome Severity Heatmap (Top 12 Sports)")
    top12 = counts.head(12)["Sport"].tolist()
    heat  = fil[fil["Sport_Type"].isin(top12)]
    pivot = heat.groupby(["Sport_Type", "Disposition_Label"]).size().unstack(fill_value=0)
    fig4  = px.imshow(
        pivot, color_continuous_scale=["#f0f4f8", PRIMARY],
        title="Number of Cases by Sport and Outcome", aspect="auto"
    )
    fig4.update_layout(height=420, margin=dict(t=50, b=10))
    st.plotly_chart(fig4, use_container_width=True)


# ── PAGE 3 · Demographics ──────────────────────────────────────────────────────
def page_demographics(df, sports):
    st.title("👥 Demographics")
    st.caption("Who is getting hurt? Age, gender, and population breakdown of sports injuries.")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        # Non-Sports first so it renders as bottom layer; Sports-Related on top in orange
        age_df = pd.concat([
            df[df["Sports_Related"] == "N"][["Age"]].assign(Type="Non-Sports"),
            sports[["Age"]].assign(Type="Sports-Related")
        ])
        fig = px.histogram(
            age_df, x="Age", color="Type", nbins=20,
            barmode="overlay", opacity=0.78,
            color_discrete_map={"Sports-Related": ACCENT, "Non-Sports": SECONDARY},
            category_orders={"Type": ["Non-Sports", "Sports-Related"]},
            title="Age Distribution: Sports vs. Non-Sports Injuries",
            labels={"Age": "Age (years)", "count": "Count", "Type": ""}
        )
        fig.update_layout(
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, title="")
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        top10 = sports["Sport_Type"].value_counts().head(10).index
        gd = sports[sports["Sport_Type"].isin(top10) & sports["Sex_Label"].isin(["Male", "Female"])]
        gp = gd.groupby(["Sport_Type", "Sex_Label"]).size().reset_index(name="Count")
        fig2 = px.bar(
            gp, x="Sport_Type", y="Count", color="Sex_Label",
            barmode="group",
            color_discrete_map={"Male": SECONDARY, "Female": ACCENT},
            title="Gender Split Across Top 10 Sports",
            labels={"Sport_Type": "", "Count": "Injuries", "Sex_Label": "Gender"}
        )
        fig2.update_layout(height=400, xaxis_tickangle=-30, legend_title="")
        st.plotly_chart(fig2, use_container_width=True)

    # Age group × Sport heatmap
    st.subheader("Age Group × Sport Heatmap")
    top12 = sports["Sport_Type"].value_counts().head(12).index
    h = sports[sports["Sport_Type"].isin(top12)]
    pivot = h.groupby(["Age_Group", "Sport_Type"]).size().unstack(fill_value=0)
    fig3 = px.imshow(
        pivot, color_continuous_scale=["#f0f4f8", SECONDARY],
        title="Injury Count by Age Group and Sport", aspect="auto"
    )
    fig3.update_layout(height=420, margin=dict(t=50, b=10))
    st.plotly_chart(fig3, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        ag = sports["Age_Group"].value_counts().reset_index()
        ag.columns = ["Age Group", "Count"]
        fig4 = px.pie(
            ag, names="Age Group", values="Count", hole=0.45,
            title="Sports Injuries by Age Group",
            color_discrete_sequence=px.colors.sequential.Blues_r
        )
        fig4.update_traces(textposition="outside", textinfo="percent+label")
        fig4.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig4, use_container_width=True)

    with col4:
        sx = sports[sports["Sex_Label"].isin(["Male", "Female"])]["Sex_Label"].value_counts().reset_index()
        sx.columns = ["Gender", "Count"]
        fig5 = px.bar(
            sx, x="Gender", y="Count", color="Gender",
            color_discrete_map={"Male": SECONDARY, "Female": ACCENT},
            title="Sports Injuries by Gender", text="Count"
        )
        fig5.update_traces(textposition="outside")
        fig5.update_layout(showlegend=False, height=400,
                           xaxis_title="", yaxis_title="Injuries")
        st.plotly_chart(fig5, use_container_width=True)


# ── PAGE 4 · Injury Profile ────────────────────────────────────────────────────
def page_injury_profile(df, sports):
    st.title("🩺 Injury Profile")
    st.caption("What injuries occur, which body parts are affected, and how severe are outcomes?")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        bp = sports["Body_Part_Label"].value_counts().head(12).reset_index()
        bp.columns = ["Body Part", "Count"]
        fig = px.bar(
            bp.sort_values("Count"), x="Count", y="Body Part", orientation="h",
            color="Count", color_continuous_scale=["#bdd7ee", PRIMARY],
            title="Most Commonly Injured Body Parts", text="Count"
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(coloraxis_showscale=False, height=480, yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        dx = sports["Diagnosis_Label"].value_counts().head(12).reset_index()
        dx.columns = ["Diagnosis", "Count"]
        fig2 = px.bar(
            dx.sort_values("Count"), x="Count", y="Diagnosis", orientation="h",
            color="Count", color_continuous_scale=["#fce4d6", DANGER],
            title="Most Common Diagnoses", text="Count"
        )
        fig2.update_traces(textposition="outside")
        fig2.update_layout(coloraxis_showscale=False, height=480, yaxis_title="")
        st.plotly_chart(fig2, use_container_width=True)

    # Body Part × Sport heatmap
    st.subheader("Body Part by Sport (Top 10)")
    top_s  = sports["Sport_Type"].value_counts().head(10).index
    top_bp = sports["Body_Part_Label"].value_counts().head(10).index
    cross  = sports[sports["Sport_Type"].isin(top_s) & sports["Body_Part_Label"].isin(top_bp)]
    pivot  = cross.groupby(["Sport_Type", "Body_Part_Label"]).size().unstack(fill_value=0)
    fig3   = px.imshow(
        pivot, color_continuous_scale=["#f0f4f8", ACCENT],
        title="Cases by Sport × Body Part", aspect="auto"
    )
    fig3.update_layout(height=420, margin=dict(t=50, b=10))
    st.plotly_chart(fig3, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        hd = sports.groupby("Diagnosis_Label")["Hospitalized"].agg(["sum", "count", "mean"]).reset_index()
        hd.columns = ["Diagnosis", "Hosp", "Total", "Rate"]
        hd = hd[hd["Total"] >= 3].sort_values("Rate", ascending=False).head(10)
        hd["Rate_Pct"] = (hd["Rate"] * 100).round(1)
        fig4 = px.bar(
            hd.sort_values("Rate_Pct"), x="Rate_Pct", y="Diagnosis", orientation="h",
            color="Rate_Pct", color_continuous_scale=["#fce4d6", DANGER],
            title="Hospitalization Rate by Diagnosis (%)", text="Rate_Pct"
        )
        fig4.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig4.update_layout(coloraxis_showscale=False, height=420, yaxis_title="")
        st.plotly_chart(fig4, use_container_width=True)

    with col4:
        loc = sports["Location_Label"].value_counts().reset_index()
        loc.columns = ["Location", "Count"]
        fig5 = px.pie(
            loc, names="Location", values="Count", hole=0.4,
            title="Where Do Sports Injuries Happen?",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig5.update_traces(textposition="outside", textinfo="percent+label")
        fig5.update_layout(showlegend=False, height=420)
        st.plotly_chart(fig5, use_container_width=True)


# ── PAGE 5 · Trends ────────────────────────────────────────────────────────────
def page_trends(df, sports):
    st.title("📅 Temporal Trends")
    st.caption("Year-over-year patterns (2019–2025) plus monthly and seasonal drill-down for any selected year.")
    st.markdown("---")

    # ── Year-over-year overview ────────────────────────────────────────────────
    st.subheader("Year-Over-Year Analysis (2019–2025)")

    annual_all    = df.groupby("Year").size().reset_index(name="All ER Visits")
    annual_sports = sports.groupby("Year").size().reset_index(name="Sports Injuries")
    annual        = annual_all.merge(annual_sports, on="Year")
    annual["Sports Rate (%)"] = (annual["Sports Injuries"] / annual["All ER Visits"] * 100).round(1)

    col1, col2 = st.columns(2)

    with col1:
        fig1 = go.Figure()
        fig1.add_bar(x=annual["Year"].astype(str), y=annual["All ER Visits"],
                     name="All ER Visits", marker_color=SECONDARY)
        fig1.add_bar(x=annual["Year"].astype(str), y=annual["Sports Injuries"],
                     name="Sports Injuries", marker_color=ACCENT)
        fig1.update_layout(
            barmode="group",
            title="Annual ER Visits: All vs. Sports-Related",
            xaxis_title="Year", yaxis_title="Cases",
            height=380,
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = go.Figure(go.Scatter(
            x=annual["Year"].astype(str),
            y=annual["Sports Rate (%)"],
            mode="lines+markers+text",
            text=[f"{v:.1f}%" for v in annual["Sports Rate (%)"]],
            textposition="top center",
            line=dict(color=ACCENT, width=3),
            marker=dict(size=10),
        ))
        fig2.update_layout(
            title="Sports Injury Rate (% of All ER Visits)",
            xaxis_title="Year", yaxis_title="Rate (%)",
            height=380, showlegend=False
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Top 5 sports by year — grouped bar
    top5_sports = sports["Sport_Type"].value_counts().head(5).index.tolist()
    sy_grp = (
        sports[sports["Sport_Type"].isin(top5_sports)]
        .groupby(["Year", "Sport_Type"]).size().reset_index(name="Count")
    )
    fig3 = px.bar(
        sy_grp, x=sy_grp["Year"].astype(str), y="Count", color="Sport_Type",
        barmode="group",
        title="Annual Injury Count: Top 5 Sports (2019–2025)",
        labels={"x": "Year", "Count": "Injuries", "Sport_Type": "Sport"},
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig3.update_layout(
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis_title="Year"
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Hospitalization rate by year
    hosp_year = sports.groupby("Year")["Hospitalized"].mean().reset_index()
    hosp_year["Hosp Rate (%)"] = (hosp_year["Hospitalized"] * 100).round(1)
    fig4 = go.Figure(go.Scatter(
        x=hosp_year["Year"].astype(str),
        y=hosp_year["Hosp Rate (%)"],
        mode="lines+markers+text",
        text=[f"{v:.1f}%" for v in hosp_year["Hosp Rate (%)"]],
        textposition="top center",
        line=dict(color=DANGER, width=3),
        marker=dict(size=9),
    ))
    fig4.update_layout(
        title="Hospitalization Rate for Sports Injuries by Year (%)",
        xaxis_title="Year", yaxis_title="Hospitalization Rate (%)",
        height=320, showlegend=False
    )
    st.plotly_chart(fig4, use_container_width=True)

    # ── Monthly drill-down ─────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Monthly Drill-Down")

    years    = sorted(sports["Year"].unique())
    sel_year = st.selectbox("Select a year to explore monthly detail:", years, index=len(years) - 1)

    yr_sports = sports[sports["Year"] == sel_year]

    monthly = yr_sports.groupby(["Month_Num", "Month_Name"]).size().reset_index(name="Count")
    monthly = monthly.sort_values("Month_Num")
    fig5 = px.area(
        monthly, x="Month_Name", y="Count",
        color_discrete_sequence=[ACCENT],
        title=f"Monthly Sports Injury Volume ({sel_year})",
        labels={"Month_Name": "Month", "Count": "Injuries"}
    )
    fig5.update_layout(
        height=360, margin=dict(t=50, b=10),
        xaxis=dict(categoryorder="array", categoryarray=MONTH_ORDER)
    )
    st.plotly_chart(fig5, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        top6   = yr_sports["Sport_Type"].value_counts().head(6).index
        ms_grp = (
            yr_sports[yr_sports["Sport_Type"].isin(top6)]
            .groupby(["Month_Name", "Month_Num", "Sport_Type"]).size().reset_index(name="Count")
            .sort_values("Month_Num")
        )
        fig6 = px.line(
            ms_grp, x="Month_Name", y="Count", color="Sport_Type",
            markers=True,
            title=f"Monthly Trend: Top 6 Sports ({sel_year})",
            labels={"Month_Name": "Month", "Count": "Injuries", "Sport_Type": "Sport"}
        )
        fig6.update_layout(
            height=400,
            xaxis=dict(categoryorder="array", categoryarray=MONTH_ORDER),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=9))
        )
        st.plotly_chart(fig6, use_container_width=True)

    with col4:
        top5 = yr_sports["Sport_Type"].value_counts().head(5).index
        qg   = (
            yr_sports[yr_sports["Sport_Type"].isin(top5)]
            .groupby(["Quarter", "Sport_Type"]).size().reset_index(name="Count")
        )
        qg["Quarter_Label"] = "Q" + qg["Quarter"].astype(str)
        fig7 = px.bar(
            qg, x="Quarter_Label", y="Count", color="Sport_Type",
            barmode="stack",
            title=f"Quarterly Injuries: Top 5 Sports ({sel_year})",
            labels={"Quarter_Label": "Quarter", "Count": "Injuries", "Sport_Type": "Sport"}
        )
        fig7.update_layout(height=400, legend=dict(font=dict(size=9)))
        st.plotly_chart(fig7, use_container_width=True)

    dow = yr_sports.groupby("DayOfWeek").size().reset_index(name="Count")
    dow["DayOfWeek"] = pd.Categorical(dow["DayOfWeek"], categories=DOW_ORDER, ordered=True)
    dow = dow.sort_values("DayOfWeek")
    fig8 = px.bar(
        dow, x="DayOfWeek", y="Count",
        color="Count", color_continuous_scale=["#bdd7ee", PRIMARY],
        title=f"Sports Injuries by Day of Week ({sel_year})", text="Count"
    )
    fig8.update_traces(textposition="outside")
    fig8.update_layout(
        coloraxis_showscale=False, height=360,
        xaxis_title="", yaxis_title="Injuries"
    )
    st.plotly_chart(fig8, use_container_width=True)


# ── PAGE 6 · Predictive Analysis ──────────────────────────────────────────────
def page_predictive(df, sports):
    st.title("🤖 Predictive Analysis")
    st.caption("Using machine learning to predict hospitalization risk from a sports injury.")
    st.markdown("---")

    st.info(
        "**Model:** Random Forest Classifier  |  "
        "**Target:** Will the patient be hospitalized? (Disposition = Admitted)  |  "
        "**Features:** Age, Gender, Sport, Body Part, Diagnosis"
    )

    mdf = sports.dropna(subset=["Sport_Type", "Body_Part_Label", "Diagnosis_Label"]).copy()

    le_sport = LabelEncoder()
    le_bp    = LabelEncoder()
    le_dx    = LabelEncoder()
    le_sex   = LabelEncoder()

    mdf["Sport_Enc"] = le_sport.fit_transform(mdf["Sport_Type"])
    mdf["BP_Enc"]    = le_bp.fit_transform(mdf["Body_Part_Label"])
    mdf["Dx_Enc"]    = le_dx.fit_transform(mdf["Diagnosis_Label"])
    mdf["Sex_Enc"]   = le_sex.fit_transform(mdf["Sex_Label"])

    X = mdf[["Age", "Sex_Enc", "Sport_Enc", "BP_Enc", "Dx_Enc"]]
    y = mdf["Hospitalized"]

    if y.sum() < 5:
        st.warning("Not enough hospitalized cases in the sports subset for reliable modelling.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=200, max_depth=6,
                                 random_state=42, class_weight="balanced")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    accuracy = (y_pred == y_test).mean()
    try:
        auc = roc_auc_score(y_test, y_prob)
    except Exception:
        auc = 0.0
    cm = confusion_matrix(y_test, y_pred)
    hosp_prec = cm[1, 1] / max(cm[:, 1].sum(), 1)

    c1, c2, c3 = st.columns(3)
    c1.markdown(kpi_card("Model Accuracy",        f"{accuracy:.1%}", "on held-out test set", SUCCESS),  unsafe_allow_html=True)
    c2.markdown(kpi_card("AUC-ROC Score",         f"{auc:.3f}",      "discrimination ability", SECONDARY), unsafe_allow_html=True)
    c3.markdown(kpi_card("Hospitalization Precision", f"{hosp_prec:.1%}", "of flagged cases",  ACCENT), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col4, col5 = st.columns(2)

    with col4:
        fi = pd.DataFrame({
            "Feature":    ["Age", "Gender", "Sport", "Body Part", "Diagnosis"],
            "Importance": clf.feature_importances_
        }).sort_values("Importance")
        fig = px.bar(
            fi, x="Importance", y="Feature", orientation="h",
            color="Importance", color_continuous_scale=["#bdd7ee", PRIMARY],
            title="Feature Importance (Random Forest)"
        )
        fig.update_layout(coloraxis_showscale=False, height=360, yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with col5:
        cm_df = pd.DataFrame(
            cm,
            index=["Actual: Released", "Actual: Hospitalized"],
            columns=["Pred: Released", "Pred: Hospitalized"]
        )
        fig2 = px.imshow(
            cm_df, text_auto=True,
            color_continuous_scale=["#f0f4f8", PRIMARY],
            title="Confusion Matrix"
        )
        fig2.update_layout(height=360)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("Try a Prediction")
    st.caption("Enter patient details to estimate hospitalization probability.")

    p1, p2, p3 = st.columns(3)
    with p1:
        pred_age  = st.slider("Age", 0, 100, 22)
        pred_sex  = st.selectbox("Gender", ["Male", "Female"])
    with p2:
        pred_sport = st.selectbox("Sport", sorted(sports["Sport_Type"].dropna().unique()))
        pred_bp    = st.selectbox("Body Part", sorted(sports["Body_Part_Label"].dropna().unique()))
    with p3:
        pred_dx = st.selectbox("Diagnosis", sorted(sports["Diagnosis_Label"].dropna().unique()))

    if st.button("Predict Risk", type="primary", use_container_width=False):
        try:
            row = np.array([[
                pred_age,
                le_sex.transform([pred_sex])[0],
                le_sport.transform([pred_sport])[0],
                le_bp.transform([pred_bp])[0],
                le_dx.transform([pred_dx])[0],
            ]])
            prob = clf.predict_proba(row)[0][1]
        except ValueError:
            st.error("A selected value was not seen during training. Please try different inputs.")
            return

        if prob < 0.20:
            col, label = SUCCESS, "LOW RISK"
        elif prob < 0.50:
            col, label = ACCENT, "MODERATE RISK"
        else:
            col, label = DANGER, "HIGH RISK"

        st.markdown(f"""
        <div style="background:{col}18;border-left:6px solid {col};
                    border-radius:8px;padding:20px 24px;margin-top:16px;">
            <p style="margin:0 0 4px 0;font-size:0.8rem;color:{col};
                      font-weight:700;text-transform:uppercase;">{label}</p>
            <h2 style="margin:0 0 6px 0;color:{col};">{prob:.1%} hospitalization probability</h2>
            <p style="margin:0;color:#555;font-size:0.88rem;">
                Patient profile: {pred_age}-year-old {pred_sex} · {pred_sport} injury ·
                {pred_bp} affected · {pred_dx}
            </p>
        </div>
        """, unsafe_allow_html=True)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    if not check_password():
        return

    df, sports = load_data()

    st.sidebar.markdown("""
    <div style="text-align:center;padding:8px 0 4px 0;">
        <span style="font-size:1.8rem;">⚕️</span><br>
        <b style="color:#1f4e79;font-size:0.95rem;">Sports Injury Intelligence</b><br>
        <span style="color:#888;font-size:0.72rem;">MSBA382 Healthcare Analytics</span>
    </div>
    """, unsafe_allow_html=True)
    st.sidebar.markdown("---")

    pages = [
        "📊 Overview",
        "⚽ Sports Danger Index",
        "👥 Demographics",
        "🩺 Injury Profile",
        "📅 Trends",
        "🤖 Predictive Analysis"
    ]
    page = st.sidebar.radio("Navigation", pages, label_visibility="collapsed")

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"""
    <div style="font-size:0.72rem;color:#999;line-height:1.8;">
        <b>Data source:</b> NEISS 2019–2025<br>
        <b>Total records:</b> {len(df):,}<br>
        <b>Sports-related:</b> {len(sports):,}<br>
        <b>Period:</b> Jan 2019 – Dec 2025
    </div>
    """, unsafe_allow_html=True)

    dispatch = {
        "📊 Overview":            page_overview,
        "⚽ Sports Danger Index":  page_sports_danger,
        "👥 Demographics":        page_demographics,
        "🩺 Injury Profile":      page_injury_profile,
        "📅 Trends":              page_trends,
        "🤖 Predictive Analysis": page_predictive,
    }
    dispatch[page](df, sports)


if __name__ == "__main__":
    main()
