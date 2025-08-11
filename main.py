"""
AI-Powered Habit Tracker (Streamlit)
Single-file app: app.py

Features:
- SQLite storage (habits + entries)
- Add / remove habits
- Mark habit done for a date (defaults to today)
- Visualizations: completion rate bar chart + streak calculation
- Lightweight ML model to predict probability of completion tomorrow
- Daily motivational quote via ZenQuotes (optional)
"""

from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler
import streamlit as st
import matplotlib.pyplot as plt
import requests
import io

# -----------------------
# Constants & DB helpers
# -----------------------
DB_PATH = Path("habits.db")

CREATE_HABITS_TABLE = """
CREATE TABLE IF NOT EXISTS habits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL
);
"""

CREATE_ENTRIES_TABLE = """
CREATE TABLE IF NOT EXISTS entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    habit_id INTEGER NOT NULL,
    entry_date TEXT NOT NULL,
    status INTEGER NOT NULL, -- 1 done, 0 not done
    FOREIGN KEY (habit_id) REFERENCES habits(id)
);
"""

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    return conn

def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(CREATE_HABITS_TABLE)
    cur.execute(CREATE_ENTRIES_TABLE)
    conn.commit()
    conn.close()

# -----------------------
# Data functions
# -----------------------
def add_habit(name: str):
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("INSERT INTO habits (name, created_at) VALUES (?, ?)", (name.strip(), date.today().isoformat()))
        conn.commit()
    except sqlite3.IntegrityError:
        st.warning("Habit already exists.")
    conn.close()

def delete_habit(habit_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM entries WHERE habit_id = ?", (habit_id,))
    cur.execute("DELETE FROM habits WHERE id = ?", (habit_id,))
    conn.commit()
    conn.close()

def get_habits_df():
    conn = get_conn()
    df = pd.read_sql_query("SELECT * FROM habits ORDER BY id", conn, parse_dates=["created_at"])
    conn.close()
    return df

def add_entry(habit_id: int, entry_date: date, status:int=1):
    conn = get_conn()
    cur = conn.cursor()
    d = entry_date.isoformat()
    # if record exists, update
    cur.execute("SELECT id FROM entries WHERE habit_id = ? AND entry_date = ?", (habit_id, d))
    row = cur.fetchone()
    if row:
        cur.execute("UPDATE entries SET status = ? WHERE id = ?", (status, row[0]))
    else:
        cur.execute("INSERT INTO entries (habit_id, entry_date, status) VALUES (?, ?, ?)", (habit_id, d, status))
    conn.commit()
    conn.close()

def get_entries_df():
    conn = get_conn()
    df = pd.read_sql_query("SELECT * FROM entries", conn, parse_dates=["entry_date"])
    conn.close()
    if df.empty:
        return df
    df['entry_date'] = pd.to_datetime(df['entry_date']).dt.date
    return df

# -----------------------
# Feature engineering for ML
# -----------------------
def compute_features_for_habit(habit_id: int, entries_df: pd.DataFrame, days_back=30):
    """
    For a given habit, compute a dataset of rows (one per day) with features:
    - recent_rate (past 7 days completion rate)
    - streak (consecutive days done up to previous day)
    - day_of_week (0-6)
    - total_tracked_days
    label: whether done on day (0/1)
    """
    if entries_df is None or entries_df.empty:
        return pd.DataFrame()
    # consider a window of days
    today = date.today()
    start = today - timedelta(days=days_back)
    dates = [start + timedelta(days=i) for i in range(days_back+1)]
    rows = []
    habit_entries = entries_df[entries_df['habit_id']==habit_id].set_index('entry_date')['status'].to_dict()
    for i, d in enumerate(dates):
        done = int(habit_entries.get(d, 0))
        # compute recent_rate using previous 7 days (excluding current day)
        prev7 = [(d - timedelta(days=j)) for j in range(1,8)]
        prev7_vals = [habit_entries.get(p, 0) for p in prev7]
        recent_rate = np.mean(prev7_vals) if len(prev7_vals) > 0 else 0.0
        # compute previous streak (consecutive days done up to previous day)
        streak = 0
        cursor = d - timedelta(days=1)
        while habit_entries.get(cursor, 0) == 1:
            streak += 1
            cursor -= timedelta(days=1)
        day_of_week = d.weekday()
        total_tracked = sum(1 for dd in habit_entries.keys() if dd <= d)
        rows.append({
            'date': d,
            'done': done,
            'recent_rate': recent_rate,
            'streak': streak,
            'dow': day_of_week,
            'total_tracked': total_tracked
        })
    return pd.DataFrame(rows)

def prepare_training_data(entries_df: pd.DataFrame, days_back=90):
    """
    Build a combined dataset across all habits for training.
    """
    if entries_df is None or entries_df.empty:
        return pd.DataFrame()
    habit_ids = entries_df['habit_id'].unique()
    dfs = []
    for hid in habit_ids:
        dfh = compute_features_for_habit(hid, entries_df, days_back=days_back)
        if not dfh.empty:
            dfh['habit_id'] = hid
            dfs.append(dfh)
    if not dfs:
        return pd.DataFrame()
    combined = pd.concat(dfs, ignore_index=True)
    # drop first few rows where total_tracked is 0 maybe
    combined = combined[combined['total_tracked']>=0]
    return combined

# -----------------------
# Model: Train & Predict
# -----------------------
def train_model(train_df: pd.DataFrame):
    """
    Train a logistic regression model on the features:
    ['recent_rate', 'streak', 'dow', 'total_tracked'] to predict 'done'
    Returns (model, scaler)
    """
    if train_df is None or train_df.empty:
        raise ValueError("No training data")
    X = train_df[['recent_rate','streak','dow','total_tracked']].values
    y = train_df['done'].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = LogisticRegression(max_iter=200)
    model.fit(Xs, y)
    return model, scaler

def predict_for_habit(habit_id: int, entries_df: pd.DataFrame, model=None, scaler=None):
    """
    Predict probability that habit will be done tomorrow.
    If model is None (or not trained), use a heuristic.
    Returns probability in [0,1]
    """
    today = date.today()
    # prepare feature for tomorrow
    # compute recent_rate over past 7 days ending today
    habit_entries = entries_df[entries_df['habit_id']==habit_id].set_index('entry_date')['status'].to_dict()
    prev7 = [(today - timedelta(days=j)) for j in range(0,7)]
    prev7_vals = [habit_entries.get(p, 0) for p in prev7]
    recent_rate = np.mean(prev7_vals) if len(prev7_vals)>0 else 0.0
    # current streak up to today
    streak = 0
    cursor = today
    while habit_entries.get(cursor, 0) == 1:
        streak += 1
        cursor -= timedelta(days=1)
    dow = (today + timedelta(days=1)).weekday()  # dow for tomorrow
    total_tracked = sum(1 for dd in habit_entries.keys() if dd <= today)
    feat = np.array([[recent_rate, streak, dow, total_tracked]])
    if model and scaler:
        try:
            Xs = scaler.transform(feat)
            prob = float(model.predict_proba(Xs)[0,1])
            return prob
        except Exception:
            pass
    # fallback heuristic
    prob = 0.6 * recent_rate + 0.25 * (np.tanh(streak/5)) + 0.15 * min(1.0, total_tracked / 30.0)
    return float(np.clip(prob, 0.01, 0.99))

# -----------------------
# Utilities: charts, streaks
# -----------------------
def plot_completion_rate(habit_id: int, entries_df: pd.DataFrame, days_back=30):
    dfh = compute_features_for_habit(habit_id, entries_df, days_back)
    if dfh.empty:
        st.info("No data to plot yet.")
        return
    dates = dfh['date']
    done = dfh['done']
    fig, ax = plt.subplots(figsize=(8,3))
    ax.bar(dates, done)
    ax.set_ylim(-0.1,1.1)
    ax.set_yticks([0,1])
    ax.set_ylabel("Done (1) / Not done (0)")
    ax.set_title(f"Last {days_back} days completion")
    fig.autofmt_xdate()
    st.pyplot(fig)

def compute_current_streak(habit_id: int, entries_df: pd.DataFrame):
    habit_entries = entries_df[entries_df['habit_id']==habit_id].set_index('entry_date')['status'].to_dict()
    streak = 0
    cursor = date.today()
    while habit_entries.get(cursor, 0) == 1:
        streak += 1
        cursor -= timedelta(days=1)
    return streak

# -----------------------
# Quotes
# -----------------------
def fetch_quote():
    try:
        r = requests.get("https://zenquotes.io/api/random", timeout=4)
        if r.status_code == 200:
            j = r.json()
            if isinstance(j, list) and len(j)>0:
                q = j[0].get("q","")
                a = j[0].get("a","")
                return f"“{q}” — {a}"
    except Exception:
        pass
    # fallback
    return "Keep going — small steps each day add up."

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="AI Habit Tracker", layout="wide", initial_sidebar_state="auto")
init_db()

st.title("🧭 AI-powered Habit Tracker")
st.markdown("Track habits, visualize progress, and see a prediction for whether you'll complete a habit tomorrow.")

# Sidebar: add habit / settings
with st.sidebar:
    st.header("Manage Habits")
    new_name = st.text_input("New habit name")
    if st.button("Add habit"):
        if new_name.strip():
            add_habit(new_name.strip())
            st.success(f"Added habit: {new_name.strip()}")
        else:
            st.error("Enter a habit name.")
    habits_df = get_habits_df()
    if not habits_df.empty:
        st.subheader("Delete habit")
        del_choice = st.selectbox("Choose habit to delete", options=["--select--"] + habits_df['name'].tolist())
        if st.button("Delete"):
            if del_choice and del_choice != "--select--":
                row = habits_df[habits_df['name']==del_choice].iloc[0]
                delete_habit(int(row['id']))
                st.warning(f"Deleted {del_choice}")
            else:
                st.error("Pick a habit first.")
    st.markdown("---")
    st.header("Quick actions")
    entries_df = get_entries_df()
    habit_map = {row['id']:row['name'] for _,row in habits_df.iterrows()} if not habits_df.empty else {}
    if habits_df.empty:
        st.info("No habits yet. Add one above!")
    else:
        # mark done for today (multiple selection)
        st.subheader("Mark done for today")
        choices = st.multiselect("Select habits you did today", options=habits_df['name'].tolist())
        if st.button("Mark selected done"):
            for name in choices:
                hid = int(habits_df[habits_df['name']==name]['id'].iloc[0])
                add_entry(hid, date.today(), status=1)
            st.success("Marked done for selected habits.")
    st.markdown("---")
    st.header("Settings")
    days_back = st.slider("Visualization days back", min_value=7, max_value=180, value=30, step=7)
    st.caption("Model trains on historical data (up to 90 days).")

# Main layout
left, right = st.columns([2,1])
with left:
    st.subheader("Habits Overview")
    habits_df = get_habits_df()
    entries_df = get_entries_df()
    if habits_df.empty:
        st.info("Add your first habit from the sidebar.")
    else:
        for _, h in habits_df.iterrows():
            hid = int(h['id'])
            name = h['name']
            st.markdown(f"### {name}")
            col1, col2, col3 = st.columns([3,2,2])
            with col1:
                st.write("**Progress (last {} days)**".format(days_back))
                plot_completion_rate(hid, entries_df, days_back=days_back)
            with col2:
                streak = compute_current_streak(hid, entries_df)
                st.metric("Current streak (days)", streak)
                # quick manual add for specific date
                with st.expander("Add / Update entry for a date"):
                    d = st.date_input("Select date", value=date.today())
                    st.radio("Status", options=["Done","Not done"], index=0, key=f"status_{hid}")
                    if st.button("Save entry", key=f"save_{hid}"):
                        status_val = 1 if st.session_state.get(f"status_{hid}") == "Done" else 0
                        add_entry(hid, d, status=status_val)
                        st.success("Saved.")
            with col3:
                st.write("**AI Prediction**")
                # Prepare training data and model
                training_df = prepare_training_data(entries_df, days_back=90)
                model = None; scaler=None
                if not training_df.empty and training_df['done'].sum() >= 10 and len(training_df) >= 50:
                    try:
                        model, scaler = train_model(training_df)
                    except Exception:
                        model=None
                prob = predict_for_habit(hid, entries_df, model=model, scaler=scaler)
                st.progress(float(prob))
                st.caption(f"Probability of doing this habit tomorrow: {prob*100:.1f}%")
                # Suggestion based on prediction
                if prob < 0.5:
                    st.warning("Low chance tomorrow — try: reduce goal, set a reminder, or pair with existing habit.")
                else:
                    st.success("Good chance — keep the momentum!")
            st.markdown("---")

with right:
    st.subheader("Analytics & Insights")
    entries_df = get_entries_df()
    if entries_df.empty:
        st.info("No entries yet — mark habits done so stats appear.")
    else:
        # show aggregated completion rates by habit
        agg = entries_df.groupby('habit_id')['status'].mean().reset_index()
        habits_map = {r['id']: r['name'] for _,r in habits_df.iterrows()}
        agg['habit'] = agg['habit_id'].map(habits_map)
        agg = agg.sort_values('status', ascending=False)
        st.write("Completion rates (all time):")
        st.dataframe(agg[['habit','status']].rename(columns={'status':'completion_rate'}).assign(completion_rate=lambda df: (df['completion_rate']*100).round(1)))
        st.markdown("---")
        st.write("Export / Backup")
        if st.button("Download CSV backup"):
            # produce a zip-like csv in memory
            df_h = habits_df.copy()
            df_e = entries_df.copy()
            buf = io.StringIO()
            df_h.to_csv(buf, index=False)
            buf.seek(0)
            st.download_button("Download habits.csv", buf.getvalue(), file_name="habits.csv", mime="text/csv")
        st.markdown("---")
        st.write("Tip: Train the model by consistently logging — the app trains a simple model automatically when there is enough data.")

st.markdown("---")
st.write("Daily motivation:")
try:
    quote = fetch_quote()
    st.info(quote)
except Exception:
    st.info("Keep consistent — progress compounds over time.")

st.markdown("----")
st.caption("Built with ❤️ — Streamlit • SQLite • scikit-learn")