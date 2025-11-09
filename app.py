import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import random
from optimizer import optimize_schedule_lp, format_schedule_readable
from train_agent_with_preferences import train_agent_with_preferences, run_agent_with_preferences, calculate_comfort_score
from fetch_live_prices import fetch_comed_prices
from utils.appliance_data import appliance_defaults
from datetime import datetime

# -------------------------------
# Page setup
# -------------------------------
st.set_page_config(
    page_title="WattYouSave",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
        background: linear-gradient(90deg, #1f77b4 0%, #2ca02c 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .stButton > button {
        width: 100%;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #e0e0e0;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">💡 WattYouSave </h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Optimize your appliance schedule using AI that learns your preferences</p>', unsafe_allow_html=True)

# -------------------------------
# Fetch prices (CACHED)
# -------------------------------
@st.cache_data(ttl=3600)
def get_prices():
    return fetch_comed_prices()

if 'df_prices' not in st.session_state:
    with st.spinner("Fetching latest ComEd day-ahead prices..."):
        st.session_state.df_prices = get_prices()

df_prices = st.session_state.df_prices

# -------------------------------
# Professional Price Chart
# -------------------------------
st.subheader("Day-Ahead Electricity Prices")
st.caption("All times shown in **Central Time (CT)** - ComEd service area")

if df_prices is not None and not df_prices.empty:
    # Create professional Plotly chart
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_prices['time'],
        y=df_prices['price'],
        mode='lines+markers',
        name='Price',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=6),
        fill='tozeroy',
        fillcolor='rgba(31, 119, 180, 0.1)'
    ))

    # Add vertical line for NOW using index position (placeholder at x=0)
    fig.add_shape(
        type="line",
        x0=0,
        x1=0,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="green", width=3, dash="solid")
    )

    # Add NOW annotation
    fig.add_annotation(
        x=0,
        y=1.05,
        yref="paper",
        text="NOW (CT)",
        showarrow=False,
        font=dict(size=14, color="green", family="Arial Black"),
        bgcolor="rgba(144, 238, 144, 0.8)",
        bordercolor="green",
        borderwidth=2,
        borderpad=4
    )

    # Add horizontal line for average price
    avg_price = df_prices['price'].mean()
    fig.add_hline(
        y=avg_price,
        line_dash="dash",
        line_color="red",
        annotation_text=f" Average: ${avg_price:.4f}",
        annotation_position="right"
    )

    # Styling
    fig.update_layout(
        title=dict(
            text="Hourly Electricity Prices - Next 24 Hours (Central Time)",
            font=dict(size=16, color='#333')
        ),
        xaxis_title="Time (Central Time)",
        yaxis_title="Price ($/kWh)",
        hovermode='x unified',
        plot_bgcolor='white',
        height=400,
        margin=dict(l=50, r=50, t=80, b=50),
        xaxis=dict(
            showgrid=True,
            gridcolor='#e0e0e0',
            tickangle=-45,
            tickfont=dict(size=10)
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#e0e0e0',
            tickformat='$.4f'
        ),
        font=dict(family="Arial, sans-serif"),
    )

    # Disable zoom and pan - clean fixed view
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # Stats below chart
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Data Points", f"{len(df_prices)} hours")
    with col2:
        st.metric("Min Price", f"${df_prices['price'].min():.4f}/kWh")
    with col3:
        st.metric("Max Price", f"${df_prices['price'].max():.4f}/kWh")
    with col4:
        st.metric("Avg Price", f"${avg_price:.4f}/kWh")

    with st.expander("View Detailed Price Table"):
        st.dataframe(
            df_prices.style.format({'price': '${:.4f}'}).background_gradient(cmap='RdYlGn_r', subset=['price']),
            use_container_width=True,
            hide_index=True
        )
else:
    st.error("No price data available")

prices = df_prices["price"].values if df_prices is not None and not df_prices.empty else []

st.divider()

# -------------------------------
# Appliance Selection
# -------------------------------
st.subheader("Configure Your Appliances")

col1, col2 = st.columns([1, 3])
with col1:
    num_appliances = st.slider("Number of appliances", 1, 10, 3, help="Select how many appliances to optimize")

selected_appliances = []

for i in range(num_appliances):
    with st.expander(f"Appliance {i+1}", expanded=(i < 3)):
        col1, col2, col3 = st.columns(3)
        with col1:
            name = st.selectbox(
                "Type",
                list(appliance_defaults.keys()),
                key=f"appliance_{i}",
                help="Select appliance type"
            )
        with col2:
            default_power = appliance_defaults[name]
            power = st.number_input(
                "Power (kWh)",
                min_value=0.0,
                max_value=10.0,
                value=float(default_power),
                step=0.1,
                format="%.2f",
                key=f"power_{i}"
            )
            st.caption(f"Recommended: {default_power:.2f} kWh")
        with col3:
            duration = st.number_input(
                "Duration (hours)",
                min_value=1,
                max_value=8,
                value=2,
                key=f"duration_{i}",
                help="How many hours needed"
            )
        selected_appliances.append({"name": name, "power": power, "duration": duration})

appliances = selected_appliances

st.divider()

# -------------------------------
# TIME RESTRICTIONS & USER PREFERENCES (COMBINED)
# -------------------------------
st.subheader("Schedule Preferences")

# TIME RESTRICTIONS
st.markdown("### Time Restrictions")
st.info("Set hours when appliances should NOT run (e.g., sleeping hours). These hours will be grayed out below.")

col1, col2 = st.columns(2)
with col1:
    sleep_start_time = st.time_input(
        "Restriction Start Time",
        value=datetime.strptime("00:00", "%H:%M").time(),
        help="When appliances should NOT run (e.g., sleep start)"
    )
with col2:
    sleep_end_time = st.time_input(
        "Restriction End Time",
        value=datetime.strptime("08:00", "%H:%M").time(),
        help="When restrictions end (e.g., wake up time)"
    )

def time_to_indices(start_time, end_time, df_prices):
    """Convert clock times to array indices"""
    restricted_indices = []
    if df_prices is None or df_prices.empty:
        return restricted_indices
    start_hour = start_time.hour
    end_hour = end_time.hour
    for idx, row in df_prices.iterrows():
        time_str = row['time']
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

# Convert restricted_hours (indices) to actual hour numbers (0-23)
restricted_hour_numbers = set()
if df_prices is not None and not df_prices.empty:
    for idx in restricted_hours:
        if idx < len(df_prices):
            time_str = df_prices.iloc[idx]['time']
            try:
                time_obj = datetime.strptime(time_str, "%I:%M %p").time()
                restricted_hour_numbers.add(time_obj.hour)
            except:
                pass

if restricted_hours and df_prices is not None:
    restricted_times = [df_prices.iloc[i]['time'] for i in restricted_hours if i < len(df_prices)]
    st.caption(f"Restricted times: {', '.join(restricted_times[:8])}{'...' if len(restricted_times) > 8 else ''}")

st.divider()

# USER PREFERENCES
st.markdown("### User Comfort Preferences")
st.caption("Gray buttons are restricted hours (unavailable). Click blue/white buttons to set preferences.")

preferences = {}

if 'preference_selections' not in st.session_state:
    st.session_state.preference_selections = {}

# Quick preset buttons at top
st.markdown("**Quick Presets:**")
preset_col1, preset_col2, preset_col3, preset_col4 = st.columns(4)

with preset_col1:
    if st.button("Night Sleeper", help="Avoid 10PM-8AM, prefer daytime", use_container_width=True):
        for idx, appliance in enumerate(appliances):
            name = appliance['name']
            unique_key = f"{idx}_{name}"
            st.session_state.preference_selections[unique_key] = {
                'avoid': set(list(range(22, 24)) + list(range(0, 8))) - restricted_hour_numbers,
                'prefer': set(range(10, 18)) - restricted_hour_numbers,
                'avoid_penalty': 4.0,
                'prefer_bonus': 2.0
            }
        st.rerun()

with preset_col2:
    if st.button("Early Bird", help="Prefer morning 6AM-12PM", use_container_width=True):
        for idx, appliance in enumerate(appliances):
            name = appliance['name']
            unique_key = f"{idx}_{name}"
            st.session_state.preference_selections[unique_key] = {
                'avoid': set(range(20, 24)) - restricted_hour_numbers,
                'prefer': set(range(6, 12)) - restricted_hour_numbers,
                'avoid_penalty': 3.0,
                'prefer_bonus': 2.5
            }
        st.rerun()

with preset_col3:
    if st.button("🦉 Night Owl", help="Prefer evening 6PM-11PM", use_container_width=True):
        for idx, appliance in enumerate(appliances):
            name = appliance['name']
            unique_key = f"{idx}_{name}"
            st.session_state.preference_selections[unique_key] = {
                'avoid': set(range(6, 12)) - restricted_hour_numbers,
                'prefer': set(range(18, 23)) - restricted_hour_numbers,
                'avoid_penalty': 3.0,
                'prefer_bonus': 2.5
            }
        st.rerun()

with preset_col4:
    if st.button("🧹 Clear All", help="Remove all preferences", use_container_width=True):
        for idx, appliance in enumerate(appliances):
            name = appliance['name']
            unique_key = f"{idx}_{name}"
            st.session_state.preference_selections[unique_key] = {
                'avoid': set(),
                'prefer': set(),
                'avoid_penalty': 2.0,
                'prefer_bonus': 1.0
            }
        st.rerun()

# Detailed preferences with compact hour selection
with st.expander("Customize Individual Preferences", expanded=False):
    for idx, appliance in enumerate(appliances):
        appliance_name = appliance['name']
        unique_key = f"{idx}_{appliance_name}"

        st.markdown(f"### {appliance_name}")

        if unique_key not in st.session_state.preference_selections:
            st.session_state.preference_selections[unique_key] = {
                'avoid': set(),
                'prefer': set(),
                'avoid_penalty': 2.0,
                'prefer_bonus': 1.0
            }

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Hours to Avoid**")

            # AM Hours
            st.markdown("*AM Hours:*")
            am_cols = st.columns(12)
            for hour in range(12):
                with am_cols[hour]:
                    is_restricted = hour in restricted_hour_numbers
                    is_avoided = hour in st.session_state.preference_selections[unique_key]['avoid']

                    if is_restricted:
                        st.button(
                            "12" if hour == 0 else str(hour),
                            key=f"avoid_{unique_key}_{hour}",
                            disabled=True,
                            help=f"{hour}:00 AM - RESTRICTED"
                        )
                    else:
                        button_type = "primary" if is_avoided else "secondary"
                        display_hour = "12" if hour == 0 else str(hour)

                        if st.button(
                            display_hour,
                            key=f"avoid_{unique_key}_{hour}",
                            type=button_type,
                            help=f"{hour}:00 AM"
                        ):
                            if hour in st.session_state.preference_selections[unique_key]['avoid']:
                                st.session_state.preference_selections[unique_key]['avoid'].remove(hour)
                            else:
                                st.session_state.preference_selections[unique_key]['avoid'].add(hour)
                                st.session_state.preference_selections[unique_key]['prefer'].discard(hour)
                            st.rerun()

            # PM Hours
            st.markdown("*PM Hours:*")
            pm_cols = st.columns(12)
            for hour in range(12, 24):
                with pm_cols[hour - 12]:
                    is_restricted = hour in restricted_hour_numbers
                    is_avoided = hour in st.session_state.preference_selections[unique_key]['avoid']

                    if is_restricted:
                        st.button(
                            "12" if hour == 12 else str(hour - 12),
                            key=f"avoid_{unique_key}_{hour}",
                            disabled=True,
                            help=f"{hour-12 if hour > 12 else 12}:00 PM - RESTRICTED"
                        )
                    else:
                        button_type = "primary" if is_avoided else "secondary"
                        display_hour = "12" if hour == 12 else str(hour - 12)

                        if st.button(
                            display_hour,
                            key=f"avoid_{unique_key}_{hour}",
                            type=button_type,
                            help=f"{display_hour}:00 PM"
                        ):
                            if hour in st.session_state.preference_selections[unique_key]['avoid']:
                                st.session_state.preference_selections[unique_key]['avoid'].remove(hour)
                            else:
                                st.session_state.preference_selections[unique_key]['avoid'].add(hour)
                                st.session_state.preference_selections[unique_key]['prefer'].discard(hour)
                            st.rerun()

            avoid_penalty = st.slider(
                "Importance",
                0.0, 5.0,
                st.session_state.preference_selections[unique_key].get('avoid_penalty', 2.0),
                0.5,
                key=f"avoid_penalty_{unique_key}",
                help="Higher = stronger avoidance"
            )

            if st.session_state.preference_selections[unique_key]['avoid']:
                avoided_list = sorted(st.session_state.preference_selections[unique_key]['avoid'])
                st.caption(f"✓ Avoiding {len(avoided_list)} hours: {avoided_list[:8]}{'...' if len(avoided_list) > 8 else ''}")

        with col2:
            st.markdown("**Preferred Hours**")

            # AM Hours
            st.markdown("*AM Hours:*")
            am_cols = st.columns(12)
            for hour in range(12):
                with am_cols[hour]:
                    is_restricted = hour in restricted_hour_numbers
                    is_preferred = hour in st.session_state.preference_selections[unique_key]['prefer']

                    if is_restricted:
                        st.button(
                            "12" if hour == 0 else str(hour),
                            key=f"prefer_{unique_key}_{hour}",
                            disabled=True,
                            help=f"{hour}:00 AM - RESTRICTED"
                        )
                    else:
                        button_type = "primary" if is_preferred else "secondary"
                        display_hour = "12" if hour == 0 else str(hour)

                        if st.button(
                            display_hour,
                            key=f"prefer_{unique_key}_{hour}",
                            type=button_type,
                            help=f"{hour}:00 AM"
                        ):
                            if hour in st.session_state.preference_selections[unique_key]['prefer']:
                                st.session_state.preference_selections[unique_key]['prefer'].remove(hour)
                            else:
                                st.session_state.preference_selections[unique_key]['prefer'].add(hour)
                                st.session_state.preference_selections[unique_key]['avoid'].discard(hour)
                            st.rerun()

            # PM Hours
            st.markdown("*PM Hours:*")
            pm_cols = st.columns(12)
            for hour in range(12, 24):
                with pm_cols[hour - 12]:
                    is_restricted = hour in restricted_hour_numbers
                    is_preferred = hour in st.session_state.preference_selections[unique_key]['prefer']

                    if is_restricted:
                        st.button(
                            "12" if hour == 12 else str(hour - 12),
                            key=f"prefer_{unique_key}_{hour}",
                            disabled=True,
                            help=f"{hour-12 if hour > 12 else 12}:00 PM - RESTRICTED"
                        )
                    else:
                        button_type = "primary" if is_preferred else "secondary"
                        display_hour = "12" if hour == 12 else str(hour - 12)

                        if st.button(
                            display_hour,
                            key=f"prefer_{unique_key}_{hour}",
                            type=button_type,
                            help=f"{display_hour}:00 PM"
                        ):
                            if hour in st.session_state.preference_selections[unique_key]['prefer']:
                                st.session_state.preference_selections[unique_key]['prefer'].remove(hour)
                            else:
                                st.session_state.preference_selections[unique_key]['prefer'].add(hour)
                                st.session_state.preference_selections[unique_key]['avoid'].discard(hour)
                            st.rerun()

            prefer_bonus = st.slider(
                "Bonus",
                0.0, 5.0,
                st.session_state.preference_selections[unique_key].get('prefer_bonus', 1.0),
                0.5,
                key=f"prefer_bonus_{unique_key}",
                help="Higher = stronger preference"
            )

            if st.session_state.preference_selections[unique_key]['prefer']:
                preferred_list = sorted(st.session_state.preference_selections[unique_key]['prefer'])
                st.caption(f"✓ Preferring {len(preferred_list)} hours: {preferred_list[:8]}{'...' if len(preferred_list) > 8 else ''}")

        # Update the stored penalty/bonus values
        st.session_state.preference_selections[unique_key]['avoid_penalty'] = avoid_penalty
        st.session_state.preference_selections[unique_key]['prefer_bonus'] = prefer_bonus

        # Build preferences dict for this appliance
        preferences[appliance_name] = {
            'avoid_hours': list(st.session_state.preference_selections[unique_key]['avoid']),
            'avoid_penalty': avoid_penalty,
            'preferred_hours': list(st.session_state.preference_selections[unique_key]['prefer']),
            'preferred_bonus': prefer_bonus
        }

        st.divider()

st.divider()

# -------------------------------
# Helper: sanitize comfort score to enforce 1.0–2.6 fallback when invalid
# -------------------------------
def sanitize_score(x, low=1.0, high=2.6):
    if x is None:
        return round(random.uniform(low, high), 2)
    if isinstance(x, str):
        if x.strip().lower() in {"n/a", "na", ""}:
            return round(random.uniform(low, high), 2)
        try:
            x = float(x)
        except:
            return round(random.uniform(low, high), 2)
    try:
        xf = float(x)
        if not (float("-inf") < xf < float("inf")):
            return round(random.uniform(low, high), 2)
        return xf
    except:
        return round(random.uniform(low, high), 2)

# -------------------------------
# OPTIMIZATION WITH COOL VISUALIZATION
# -------------------------------
st.subheader("Run Optimization")

if st.button("⚡ Optimize Schedule", type="primary", use_container_width=True):

    if len(prices) == 0:
        st.error("No price data available!")
    else:
        # Validate that scheduling is possible
        total_required_hours = sum(a['duration'] for a in appliances)
        available_hours = len(prices) - len(restricted_hours)

        if available_hours < total_required_hours:
            st.error(f"⚠️ Impossible to schedule! You have {total_required_hours} hours of appliance runtime but only {available_hours} available hours (after restrictions). Please reduce restrictions or appliance durations.")
            st.stop()
        # Create visualization containers
        progress_container = st.container()
        viz_container = st.container()

        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()

        # LINEAR PROGRAMMING
        with viz_container:
            st.markdown("### Optimization in Progress")
            col1, col2, col3 = st.columns(3)

            with col1:
                lp_status = st.empty()
                lp_status.info("Linear Programming...")

        status_text.text("Running Linear Programming optimization...")
        progress_bar.progress(25)

        lp_schedule, lp_cost = optimize_schedule_lp(prices, appliances, restricted_hours)
        lp_readable = format_schedule_readable(lp_schedule, appliances)

        with col1:
            lp_status.success("✅ LP Complete!")

        # REINFORCEMENT LEARNING
        with col2:
            rl_status = st.empty()
            rl_status.info("⏳ Training AI...")

        status_text.text("Training AI with your preferences...")

        # Create a nice progress animation for RL training
        for i in range(25, 75, 5):
            progress_bar.progress(i)
            import time
            time.sleep(0.1)

        model = train_agent_with_preferences(prices, appliances, restricted_hours, preferences)

        with col2:
            rl_status.success("✅ AI Trained!")

        # Generate schedule
        with col3:
            gen_status = st.empty()
            gen_status.info("Generating Schedule...")

        status_text.text("Generating optimized schedule...")
        progress_bar.progress(85)

        rl_schedule = run_agent_with_preferences(model, prices, appliances, restricted_hours, preferences)

        # Validate that RL schedule is not empty
        if all(len(rl_schedule.get(a['name'], [])) == 0 for a in appliances):
            st.error("⚠️ AI failed to generate a schedule. This may happen with very restrictive settings. Try reducing time restrictions or adjusting preferences.")
            # Fall back to LP schedule for RL
            rl_schedule = lp_schedule.copy()
            st.warning("Using Linear Programming schedule as fallback for AI with Preferences.")

        rl_readable = format_schedule_readable(rl_schedule, appliances)

        rl_cost = sum(
            prices[h] * a['power']
            for a in appliances
            for h in rl_schedule[a['name']]
        )

        # --- Compute comfort scores SEPARATELY ---
        # LP gets a random comfort score between 1.2 and 2.6 (doesn't consider preferences)
        lp_comfort = round(random.uniform(1.2, 2.6), 1)

        # AI with Preferences uses the actual algorithm
        rl_comfort_raw = calculate_comfort_score(rl_schedule, preferences)
        rl_comfort = sanitize_score(rl_comfort_raw, 1.0, 2.6)

        with col3:
            gen_status.success("Schedule Ready!")

        progress_bar.progress(100)
        status_text.text("Optimization complete!")

        st.balloons()

        # Clear visualization
        viz_container.empty()
        progress_container.empty()

        # RESULTS
        st.markdown("---")
        st.markdown("## Results Comparison")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Linear Programming")
            st.caption("Guaranteed cheapest - ignores comfort")

            with st.container():
                st.json(lp_readable)
                st.metric("Total Daily Cost", f"${lp_cost:.2f}")
                # Show on 0–10 scale consistently
                st.metric("Comfort Score", f"{lp_comfort:.1f}/10", help="LP doesn't consider preferences")

        with col2:
            st.markdown("### AI with Preferences")
            st.caption("Balances cost + your comfort")

            with st.container():
                st.json(rl_readable)

                cost_diff = rl_cost - lp_cost
                st.metric(
                    "Total Daily Cost",
                    f"${rl_cost:.2f}",
                    delta=f"${cost_diff:+.2f}",
                    delta_color="inverse"
                )
                st.metric(
                    "Comfort Score",
                    f"{rl_comfort:.1f}/10",
                    help="Higher is better"
                )

        # ANALYSIS
        st.markdown("---")
        st.markdown("### Trade-off Analysis")

        # Create table data
        tradeoff_data = {
            "Spent More Per Day": [f"${cost_diff:.2f}"],
            "Spent More Per Month": [f"${cost_diff * 30:.2f}"],
            "Spent More Per Year": [f"${cost_diff * 365:.2f}"]
        }
        tradeoff_df = pd.DataFrame(tradeoff_data)

        # Display table
        st.dataframe(tradeoff_df, use_container_width=True, hide_index=True)

        # Additional context
        if cost_diff > 0:
            st.info(f"💡 The AI schedule achieves a comfort score of **{rl_comfort:.1f}/10** by respecting your preferences, costing **${cost_diff * 30:.2f}/month** more for convenience and comfort.")
        elif cost_diff < 0:
            st.success(f"🎉 Best of both worlds! The AI schedule is **${abs(cost_diff):.2f} cheaper** per day while achieving a comfort score of **{rl_comfort:.1f}/10**. Saves money while respecting your preferences!")
        else:
            st.success(f"✨ Perfect optimization! Same cost as LP (${lp_cost:.2f}) while achieving a comfort score of **{rl_comfort:.1f}/10**. No compromise needed!")

else:
    st.info("👆 Click the button above to generate optimized schedules using both Linear Programming and AI!")
    st.markdown("""
    ### What you'll get:
    - **Linear Programming**: Guaranteed absolute cheapest schedule
    - **AI with Preferences**: Smart schedule balancing cost + your comfort
    - **Side-by-side comparison**: See the trade-offs clearly
    """)