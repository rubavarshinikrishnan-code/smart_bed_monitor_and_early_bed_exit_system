import json
import os
import time
 
import pandas as pd
import streamlit as st
from supabase import create_client
 
st.set_page_config(page_title="Smart Patient Bed Monitor", layout="centered")
st.title("Smart Patient Bed Monitor")
 
COLORS = {"NONE": "#888888", "LOW": "#1a9e3a", "MEDIUM": "#e6a100", "HIGH": "#d62828"}
LABELS = {"NONE": "NO PATIENT", "LOW": "SAFE", "MEDIUM": "WARNING", "HIGH": "ALERT"}
 

# Connect to Supabase and fetch latest patient status
try:
    supabase = create_client(
        st.secrets["SUPABASE_URL"],
        st.secrets["SUPABASE_KEY"]
    )

    response = (
        supabase.table("patient_status")
        .select("*")
        .eq("id", "latest")
        .limit(1)
        .execute()
    )

    if response.data:
        status = response.data[0]
    else:
        status = {
            "level": "NONE",
            "detected": False,
            "message": "Waiting for monitor...",
            "source": "-",
            "activity": "-",
            "eyes": "-",
            "time": "-"
        }

except Exception as e:
    st.error(f"Supabase error: {e}")
    status = {
        "level": "NONE",
        "detected": False,
        "message": "Waiting for monitor...",
        "source": "-",
        "activity": "-",
        "eyes": "-",
        "time": "-"
    }
lvl = status.get("level", "NONE")
if lvl not in COLORS:
    lvl = "NONE"
st.markdown(
    f"<div style='background:{COLORS[lvl]};padding:30px;border-radius:12px;"
    f"text-align:center;color:white;font-size:42px;font-weight:700'>{LABELS[lvl]}</div>",
    unsafe_allow_html=True)
st.subheader(status["message"])
 
eyes = status.get("eyes", "-")
act = status.get("activity", "-")
if eyes == "SLEEPING" or act.startswith("SLEEPING"):
    sleep_txt, sleep_col = "SLEEPING (eyes closed)", "#3a4bd6"
elif eyes == "AWAKE":
    sleep_txt, sleep_col = "AWAKE (eyes open)", "#1a9e3a"
else:
    sleep_txt, sleep_col = "Eyes not visible", "#888888"
st.markdown(
    f"<div style='background:{sleep_col};padding:14px;border-radius:10px;margin-top:10px;"
    f"text-align:center;color:white;font-size:26px;font-weight:600'>{sleep_txt}</div>",
    unsafe_allow_html=True)
 
c1, c2, c3, c4 = st.columns(4)
c1.metric("Patient", "Detected" if status["detected"] else "Not detected")
c2.metric("Risk level", lvl)
c3.metric("Activity", status.get("activity", "-"))
c4.metric("Last update", status["time"])
 
st.divider()
st.subheader("Alert history")
if os.path.exists("alerts_log.csv"):
    df = pd.read_csv("alerts_log.csv")
    st.metric("Total alerts", len(df))
    st.dataframe(df.tail(10).iloc[::-1], use_container_width=True, hide_index=True)
else:
    st.info("No alerts yet.")
 
time.sleep(1)
st.rerun()
