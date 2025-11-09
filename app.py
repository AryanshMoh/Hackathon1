import streamlit as st
import pandas as pd
from optimizer import optimize_schedule_lp, format_schedule_readable
from train_agent_with_preferences import (
    train_agent_with_preferences,
    run_agent_with_preferences,
    calculate_comfort_score,
)
from fetch_live_prices import fetch_comed_prices
from utils.appliance_data import appliance_defaults
from datetime import datetime

# -------------------------------
# Page setup
# -------------------------------
st.set_page_config(page_title="SmartEnergy Optimizer", layout="wide")
st.title("SmartEnergy Optimizer")
st.write("Compare **Linear Programming** (pure cost) vs **AI with Preferences** (cost + comfort)")

# -------------------------------
# Fetch prices (CACHED)
# -------------------------------
@st.cache_data(ttl=3600)
def get_prices():
    return fetch_comed_prices()

st.subheader("💲 Latest Energy Prices")

if "df_prices" not in st.session_state:
    with st.spinner("Fetching latest ComEd day-ahead prices..."):
        st.session_state.df_prices = get_prices()

df_prices = st.session_state.df_prices

if df_prices is not None and not df_prices.empty:
    st.line_chart(df_prices["price"], height=200)
    st.caption("Day-ahead electricity prices ($/kWh)")

    with st.expander("📋 View Price Data Table"):
        st.dataframe(df_prices)
else:
    st.warning("⚠️ No price data available")

prices = df_prices["price"].values if df_prices is not None else []

if len(prices) > 0:
    st.success(f"Ready with {len(prices)} hourly prices (Min: ${min(prices):.4f}, Max: ${max(prices):.4f})")

# -------------------------------
# Appliance Selection
# -------------------------------
st.subheader("Select Your Appliances")

selected_appliances = []
num_appliances = st.slider("How many appliances to optimize?", 1, 10, 3)

for i in range(num_appliances):
    col1, col2, col3 = st.columns(3)
    with col1:
        name = st.selectbox(f"Appliance {i+1}", list(appliance_defaults.keys()), key=f"appliance_{i}")
    with col2:
        default_power = appliance_defaults[name]
        power = st.number_input(
            f"{name} Power (kWh)",
            min_value=0.0,
            max_value=10.0,
            value=float(default_power),
            step=0.1,
            format="%.2f",
            key=f"power_{i}",
        )
        st.caption(f"💡 Typical: {default_power:.2f} kWh")
    with col3:
        duration = st.number_input(f"{name} Duration (hours)", min_value=1, max_value=8, value=2, key=f"duration_{i}")
    selected_appliances.append({"name": name, "power": power, "duration": duration})

appliances = selected_appliances

# -------------------------------
# Time Restrictions
# -------------------------------
st.subheader("Time Restrictions")
st.info("⏰ Enter actual clock times for when appliances should NOT run (e.g., sleep hours)")

col1, col2 = st.columns(2)
with col1:
    sleep_start_time = st.time_input("Restriction Start Time", value=datetime.strptime("00:00", "%H:%M").time())
with col2:
    sleep_end_time = st.time_input("Restriction End Time", value=datetime.strptime("08:00", "%H:%M").time())


def time_to_indices(start_time, end_time, df_prices):
    """Convert clock times to array indices"""
    restricted_indices = []

    if df_prices is None or df_prices.empty:
        return restricted_indices

    start_hour = start_time.hour
    end_hour = end_time.hour

    for idx, row in df_prices.iterrows():
        time_str = row["time"]
        try:
            time_obj = datetime.strptime(time_str, "%I:%M %p").time()
            hour = time_obj.hour

            if start_hour <= end_hour:
                if start_hour <= hour < end_hour:
                    restricted_indices.append(idx)
            else:
                if hour >= start_hour or hour < end_hour:
                    restricted_indices.append(idx)
        except:
            continue

    return restricted_indices


restricted_hours = time_to_indices(sleep_start_time, sleep_end_time, df_prices)

if restricted_hours:
    st.caption(f"Restricted hours: {', '.join([df_prices.iloc[i]['time'] for i in restricted_hours if i < len(df_prices)])}")

# -------------------------------
# USER PREFERENCES (NEW!)
# -------------------------------
st.subheader("⭐ User Comfort Preferences")
st.info("💡 Tell the AI when you prefer (or want to avoid) running each appliance. Click hours to toggle!")

preferences = {}

# Initialize session state for selections
if "preference_selections" not in st.session_state:
    st.session_state.preference_selections = {}

with st.expander("🎛️ Configure Preferences (Optional - expand to customize)", expanded=False):
    for idx, appliance in enumerate(appliances):
        appliance_name = appliance["name"]
        unique_key = f"{idx}_{appliance_name}"

        st.markdown(f"### {appliance_name}")

        if unique_key not in st.session_state.preference_selections:
            st.session_state.preference_selections[unique_key] = {"avoid": set(), "prefer": set()}

        col1, col2 = st.columns(2)

        # Avoid hours
        with col1:
            st.markdown("**🚫 Hours to Avoid** (click to toggle)")
            avoid_cols = st.columns(8)
            for hour in range(24):
                col_idx = hour % 8
                with avoid_cols[col_idx]:
                    is_avoided = hour in st.session_state.preference_selections[unique_key]["avoid"]
                    button_type = "primary" if is_avoided else "secondary"
                    button_label = f"{'✓ ' if is_avoided else ''}{hour:02d}"

                    if st.button(
                        button_label,
                        key=f"avoid_{unique_key}_{hour}",
                        type=button_type,
                        use_container_width=True,
                    ):
                        if hour in st.session_state.preference_selections[unique_key]["avoid"]:
                            st.session_state.preference_selections[unique_key]["avoid"].remove(hour)
                        else:
                            st.session_state.preference_selections[unique_key]["avoid"].add(hour)
                            st.session_state.preference_selections[unique_key]["prefer"].discard(hour)
                        st.rerun()

            avoid_penalty = st.slider(
                "Avoid penalty (how important?)", 0.0, 5.0, 2.0, 0.5, key=f"avoid_penalty_{unique_key}"
            )

            if st.session_state.preference_selections[unique_key]["avoid"]:
                st.caption(f"Avoiding: {sorted(st.session_state.preference_selections[unique_key]['avoid'])}")

            if st.button(f"Clear Avoid", key=f"clear_avoid_{unique_key}"):
                st.session_state.preference_selections[unique_key]["avoid"].clear()
                st.rerun()

        # Preferred hours
        with col2:
            st.markdown("**⭐ Preferred Hours** (click to toggle)")
            prefer_cols = st.columns(8)
            for hour in range(24):
                col_idx = hour % 8
                with prefer_cols[col_idx]:
                    is_preferred = hour in st.session_state.preference_selections[unique_key]["prefer"]
                    button_type = "primary" if is_preferred else "secondary"
                    button_label = f"{'✓ ' if is_preferred else ''}{hour:02d}"

                    if st.button(
                        button_label,
                        key=f"prefer_{unique_key}_{hour}",
                        type=button_type,
                        use_container_width=True,
                    ):
                        if hour in st.session_state.preference_selections[unique_key]["prefer"]:
                            st.session_state.preference_selections[unique_key]["prefer"].remove(hour)
                        else:
                            st.session_state.preference_selections[unique_key]["prefer"].add(hour)
                            st.session_state.preference_selections[unique_key]["avoid"].discard(hour)
                        st.rerun()

            prefer_bonus = st.slider(
                "Prefer bonus (how much bonus?)", 0.0, 5.0, 1.0, 0.5, key=f"prefer_bonus_{unique_key}"
            )

            if st.session_state.preference_selections[unique_key]["prefer"]:
                st.caption(f"Preferring: {sorted(st.session_state.preference_selections[unique_key]['prefer'])}")

            if st.button(f"Clear Prefer", key=f"clear_prefer_{unique_key}"):
                st.session_state.preference_selections[unique_key]["prefer"].clear()
                st.rerun()

        preferences[appliance_name] = {
            "avoid_hours": list(st.session_state.preference_selections[unique_key]["avoid"]),
            "avoid_penalty": avoid_penalty,
            "preferred_hours": list(st.session_state.preference_selections[unique_key]["prefer"]),
            "preferred_bonus": prefer_bonus,
        }

        st.divider()

# Quick presets
st.markdown("### 🎯 Quick Presets")
preset_col1, preset_col2, preset_col3, preset_col4 = st.columns(4)

with preset_col1:
    if st.button("🌙 Night Sleeper", help="Avoid 10PM-8AM"):
        for idx, appliance in enumerate(appliances):
            name = appliance["name"]
            unique_key = f"{idx}_{name}"
            st.session_state.preference_selections[unique_key] = {
                "avoid": set(list(range(22, 24)) + list(range(0, 8))),
                "prefer": set(range(10, 18)),
            }
        st.rerun()

with preset_col2:
    if st.button("🌅 Early Bird", help="Prefer morning hours"):
        for idx, appliance in enumerate(appliances):
            name = appliance["name"]
            unique_key = f"{idx}_{name}"
            st.session_state.preference_selections[unique_key] = {
                "avoid": set(range(20, 24)),
                "prefer": set(range(6, 12)),
            }
        st.rerun()

with preset_col3:
    if st.button("🦉 Night Owl", help="Prefer evening hours"):
        for idx, appliance in enumerate(appliances):
            name = appliance["name"]
            unique_key = f"{idx}_{name}"
            st.session_state.preference_selections[unique_key] = {
                "avoid": set(range(6, 12)),
                "prefer": set(range(18, 23)),
            }
        st.rerun()

with preset_col4:
    if st.button("🧹 Clear All", help="Remove all preferences"):
        for idx, appliance in enumerate(appliances):
            name = appliance["name"]
            unique_key = f"{idx}_{name}"
            st.session_state.preference_selections[unique_key] = {"avoid": set(), "prefer": set()}
        st.rerun()

# -------------------------------
# 🚀 Run Optimization
# -------------------------------
st.header("🚀 Run Optimization")

col1, col2 = st.columns(2)

# Initialize placeholders in session state
if "schedule_lp" not in st.session_state:
    st.session_state.schedule_lp = None
    st.session_state.total_cost_lp = None
    st.session_state.comfort_lp = None

if "schedule_ai" not in st.session_state:
    st.session_state.schedule_ai = None
    st.session_state.total_cost_ai = None
    st.session_state.comfort_ai = None

# --- LP Optimizer ---
if col1.button("💡 Run LP Optimizer"):
    with st.spinner("Running Linear Programming optimizer..."):
        schedule_lp, total_cost_lp = optimize_schedule_lp(prices, appliances, restricted_hours)
        st.session_state.schedule_lp = schedule_lp
        st.session_state.total_cost_lp = total_cost_lp

        # Evaluate LP comfort too (usually low)
        comfort_lp = calculate_comfort_score(schedule_lp, preferences)
        st.session_state.comfort_lp = comfort_lp

        st.success("✅ Linear Programming optimization complete!")
        st.metric("Total Cost ($)", f"{total_cost_lp:.4f}")
        st.metric("Comfort Score", f"{comfort_lp:.2f}/10")
        st.json(format_schedule_readable(schedule_lp, appliances))

# --- AI Optimizer ---
if col2.button("🤖 Run AI Optimizer (with Preferences)"):
    with st.spinner("Training AI agent (may take ~30–45 seconds)..."):
        model = train_agent_with_preferences(prices, appliances, restricted_hours, preferences)
        schedule_ai = run_agent_with_preferences(model, prices, appliances, restricted_hours, preferences)

        # Compute AI cost
        ai_total_cost = 0.0
        for appliance in appliances:
            name = appliance["name"]
            power = appliance["power"]
            for hour in schedule_ai.get(name, []):
                if 0 <= hour < len(prices):
                    ai_total_cost += prices[int(hour)] * power

        # Compute AI comfort
        comfort_ai = calculate_comfort_score(schedule_ai, preferences)

        # Store in session
        st.session_state.schedule_ai = schedule_ai
        st.session_state.total_cost_ai = ai_total_cost
        st.session_state.comfort_ai = comfort_ai

        st.success("✅ AI optimization complete!")
        st.metric("Total Cost ($)", f"{ai_total_cost:.4f}")
        st.metric("Comfort Score", f"{comfort_ai:.2f}/10")
        st.json(format_schedule_readable(schedule_ai, appliances))

# -------------------------------
# ⚖️ Comparison Section
# -------------------------------
st.divider()
st.header("⚖️ Compare LP vs AI Optimizer")

if st.button("Compare Results"):
    if (
        st.session_state.schedule_lp is None
        or st.session_state.schedule_ai is None
        or st.session_state.total_cost_lp is None
        or st.session_state.total_cost_ai is None
        or st.session_state.comfort_ai is None
    ):
        st.warning("⚠️ You must run both the LP and AI optimizers before comparing.")
    else:
        st.success("✅ Comparison ready!")

        colA, colB = st.columns(2)
        with colA:
            st.subheader("💡 Linear Programming")
            st.metric("Total Cost ($)", f"{st.session_state.total_cost_lp:.4f}")
            st.metric("Comfort Score", f"{st.session_state.comfort_lp:.2f}/10")
            st.json(format_schedule_readable(st.session_state.schedule_lp, appliances))

        with colB:
            st.subheader("🤖 AI Optimizer (Preferences)")
            st.metric("Total Cost ($)", f"{st.session_state.total_cost_ai:.4f}")
            st.metric("Comfort Score", f"{st.session_state.comfort_ai:.2f}/10")
            st.json(format_schedule_readable(st.session_state.schedule_ai, appliances))

        # -------------------------------
        # Cost difference summary + projected savings
        # -------------------------------
        st.divider()

        cost_diff = st.session_state.total_cost_ai - st.session_state.total_cost_lp
        abs_diff = abs(cost_diff)
        comfort_diff = st.session_state.comfort_ai - st.session_state.comfort_lp

        # Calculate projected monthly/yearly impact
        daily_savings = -cost_diff  # positive means LP saves money
        monthly_savings = daily_savings * 30
        yearly_savings = daily_savings * 365

        if cost_diff > 0:
            emoji = "🔺"
            cost_text = f"AI costs **${abs_diff:.4f} more** per day than LP"
        elif cost_diff < 0:
            emoji = "🟢"
            cost_text = f"AI costs **${abs_diff:.4f} less** per day than LP"
        else:
            emoji = "⚖️"
            cost_text = "AI and LP have identical daily costs"

        comfort_text = (
            f"Comfort improved by **{comfort_diff:+.2f} points**."
            if comfort_diff != 0
            else "No comfort difference detected."
        )

        st.markdown(
            f"""
            ### 📊 Summary:
            - **LP Total Cost:** ${st.session_state.total_cost_lp:.4f}  
            - **LP Comfort:** {st.session_state.comfort_lp:.2f}/10  
            - **AI Total Cost:** ${st.session_state.total_cost_ai:.4f}  
            - **AI Comfort:** {st.session_state.comfort_ai:.2f}/10  
            ---
            {emoji} {cost_text}  
            💬 {comfort_text}
            """
        )

        # Show projected monthly/yearly cost impact
        st.subheader("💰 Projected Savings (if schedule repeats daily)")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Daily Difference ($)", f"{daily_savings:+.4f}")
        with col2:
            st.metric("Monthly (~30 days)", f"{monthly_savings:+.2f}")
        with col3:
            st.metric("Yearly (~365 days)", f"{yearly_savings:+.2f}")
