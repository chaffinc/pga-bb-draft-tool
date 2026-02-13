"""
PGA Best Ball Draft Tool - Streamlit Web App
Converted from Tkinter GUI to web interface
"""

import streamlit as st
import pandas as pd
import model

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="PGA Best Ball Draft Tool",
    page_icon="⛳",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    /* Main background and text colors */
    .stApp {
        background-color: #1e1e1e;
        color: #e8e8e8;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* DATAFRAME FONT SIZE - Make text inside dataframes smaller */
    div[data-testid="stDataFrame"] table {
        font-size: 0.75rem !important;
    }

    div[data-testid="stDataFrame"] thead th {
        font-size: 0.75rem !important;
        padding: 0.2rem 0.4rem !important;
    }

    div[data-testid="stDataFrame"] tbody td {
        font-size: 0.75rem !important;
        padding: 0.2rem 0.4rem !important;
        line-height: 1.2 !important;
    }

    /* Make row height more compact */
    div[data-testid="stDataFrame"] tbody tr {
        height: 1.5rem !important;
    }

    /* Change selection color from red to blue - Updated selectors */

    /* BaseWeb checkbox "checked" fill */
    div[data-baseweb="checkbox"] input:checked + div {
        background-color: #2f4f66 !important;
        border-color: #2f4f66 !important;
    }

    /* Checkbox styling */
    input[type="checkbox"]:checked {
        accent-color: #2f4f66 !important;
    }

    /* Dataframe row selection - multiple selector attempts */
    div[data-testid="stDataFrame"] tbody tr[aria-selected="true"],
    div[data-testid="stDataFrame"] tbody tr[aria-selected="true"] td,
    div[data-testid="stDataFrame"] tr[aria-selected="true"],
    div[data-testid="stDataFrame"] tr[aria-selected="true"] td {
        background-color: #2f4f66 !important;
        color: white !important;
    }

    /* Dataframe selection checkboxes */
    div[data-testid="stDataFrame"] input[type="checkbox"]:checked::before {
        background-color: #2f4f66 !important;
    }

    /* Override any red colors with blue */
    [class*="selected"] {
        background-color: #2f4f66 !important;
    }

    /* Text input focus */
    .stTextInput input:focus,
    input[type="text"]:focus {
        border-color: #2f4f66 !important;
        outline-color: #2f4f66 !important;
        box-shadow: 0 0 0 0.2rem rgba(47, 79, 102, 0.25) !important;
    }

    /* Button hover and focus */
    .stButton>button:hover,
    .stButton>button:focus,
    button:hover,
    button:focus {
        background-color: #3d6280 !important;
        border-color: #3d6280 !important;
    }

    /* Dataframe styling */
    .stDataFrame {
        background-color: #2a2a2a;
    }

    /* Button styling */
    .stButton>button,
    button[kind="primary"] {
        width: 100%;
        background-color: #2f4f66;
        color: white;
        border: none;
        padding: 0.5rem;
        border-radius: 4px;
    }

    /* Column headers */
    h3 {
        color: #e8e8e8;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }

    /* Number input styling */
    .stNumberInput input {
        background-color: #2a2a2a;
        color: #e8e8e8;
    }

    /* Number input focus */
    .stNumberInput input:focus {
        border-color: #2f4f66 !important;
        outline-color: #2f4f66 !important;
    }

    /* Info boxes */
    .stAlert {
        background-color: #2a2a2a;
    }

    /* Primary button styling */
    button[kind="primary"] {
        background-color: #2f4f66 !important;
    }

    button[kind="primary"]:hover {
        background-color: #3d6280 !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# INITIALIZE SESSION STATE
# ============================================================
CSV_PATH = "PGA_BestBall.csv"

if 'initialized' not in st.session_state:
    # Initialize model
    model.init_model(CSV_PATH)
    st.session_state.df = model.df
    st.session_state.event_cols = model.event_cols
    
    # Initialize draft state
    st.session_state.drafted_players = []
    st.session_state.unavailable_players = []
    st.session_state.initialized = True
    
    # Round multipliers
    st.session_state.r1_mult = model.ROUND_MULTIPLIERS['Round1']
    st.session_state.r2_mult = model.ROUND_MULTIPLIERS['Round2']
    st.session_state.r3_mult = model.ROUND_MULTIPLIERS['Round3']
    st.session_state.r4_mult = model.ROUND_MULTIPLIERS['Round4']

if "model_ran" not in st.session_state:
    st.session_state.model_ran = False

df = st.session_state.df
event_cols = st.session_state.event_cols

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_round_color(event):
    """Get background color for event based on round"""
    round_colors = {
        "Round1": "#2d3a47",
        "Round2": "#3a2d47",
        "Round3": "#2d473a",
        "Round4": "#47392d"
    }
    round_id = model.EVENT_TO_ROUND.get(event, "Round3")
    return round_colors.get(round_id, "#2d3a47")

def format_coverage_table():
    """Format the coverage table with colors"""
    if not st.session_state.drafted_players:
        return pd.DataFrame()
    
    coverage = model.compute_event_coverage(st.session_state.drafted_players)
    coverage['Round'] = coverage['Event'].map(lambda e: model.EVENT_TO_ROUND.get(e, "Round3"))
    
    # Reorder columns
    coverage = coverage[['Event', 'Round', 'Covered', 'Needed', 'Remaining Picks']]
    return coverage.sort_values(['Covered', 'Event'])

# ============================================================
# MAIN APP LAYOUT
# ============================================================

st.title("⛳ PGA Best Ball Live Draft Tool - The Scramble")

# Create three main columns
col1, col2, col3 = st.columns([2, 2, 2])

# ============================================================
# LEFT COLUMN: All Players
# ============================================================
with col1:
    # Initialize selected_recommendation if it doesn't exist
    if 'selected_recommendation' not in st.session_state:
        st.session_state.selected_recommendation = None

    # Section title
    st.markdown("### All Players")

    search_col, checkbox_col = st.columns([2, 1], gap="small")

    with search_col:
        search_text = st.text_input(
            "Search players",
            "",
            label_visibility="collapsed",
            placeholder="Search players..."
        )

    with checkbox_col:
        show_drafted = st.checkbox(
            "Show Drafted/Blocked",
            value=False
        )

    # Filter golfers
    df_sorted = df.sort_values("ADP")
    golfer_df = df_sorted[['ADP']].copy()
    golfer_df['Name'] = golfer_df.index
    golfer_df = golfer_df[['Name', 'ADP']].reset_index(drop=True)

    # Apply filters
    if search_text:
        # Make search case-insensitive and handle NaN values
        golfer_df = golfer_df[
            golfer_df['Name'].fillna('').str.lower().str.contains(search_text.lower(), regex=False)
        ]

    if not show_drafted:
        drafted_set = set(st.session_state.drafted_players)
        unavailable_set = set(st.session_state.unavailable_players)
        golfer_df = golfer_df[~golfer_df['Name'].isin(drafted_set | unavailable_set)]


    # Add status column to identify drafted/blocked players
    def get_status(name):
        if name in st.session_state.drafted_players:
            return "📝 Drafted"
        elif name in st.session_state.unavailable_players:
            return "🚫 Blocked"
        else:
            return ""


    golfer_df['Status'] = golfer_df['Name'].apply(get_status)

    # Replace 999.0 with "-" for display
    golfer_df['ADP'] = golfer_df['ADP'].apply(lambda x: "-" if x >= 999 else f"{x:.1f}")

    # Reorder columns to show status
    golfer_df = golfer_df[['Name', 'ADP', 'Status']].reset_index(drop=True)

    # Initialize clear_selection flag if not exists
    if 'clear_selection' not in st.session_state:
        st.session_state.clear_selection = False

    # Force clear selection by recreating the key
    if st.session_state.clear_selection:
        if 'selection_counter' not in st.session_state:
            st.session_state.selection_counter = 0
        st.session_state.selection_counter += 1
        st.session_state.clear_selection = False
    else:
        if 'selection_counter' not in st.session_state:
            st.session_state.selection_counter = 0

    # Display the golfer list with selection
    event = st.dataframe(
        golfer_df,
        hide_index=True,
        height=500,
        width='stretch',
        on_select="rerun",
        selection_mode="multi-row",
        key=f"golfer_selection_{st.session_state.selection_counter}",
        column_config={
            "Name": st.column_config.TextColumn("Name", width="medium"),
            "ADP": st.column_config.TextColumn("ADP", width="small"),
            "Status": st.column_config.TextColumn("Status", width="small"),
        }
    )

    # Get selected golfer names from the selection
    selected_indices = event.selection.rows if event.selection else []
    selected_golfers = [golfer_df.iloc[i]['Name'] for i in selected_indices] if selected_indices else []

    # If a player is selected in All Players, clear recommendation selection
    if selected_golfers and st.session_state.get('selected_recommendation'):
        st.session_state.selected_recommendation = None
        if 'rec_selection_counter' not in st.session_state:
            st.session_state.rec_selection_counter = 0
        st.session_state.rec_selection_counter += 1
        st.rerun()

    # Action buttons
    button_col1, button_col2, button_col3 = st.columns(3)

    with button_col1:
        # Determine which player to draft: recommendation takes priority, otherwise use All Players selection
        player_to_draft = None

        # Check recommendation first
        if st.session_state.get('selected_recommendation'):
            player_to_draft = st.session_state.selected_recommendation
        # Then check All Players selection
        elif len(selected_golfers) == 1:
            player_to_draft = selected_golfers[0]

        # Draft button: enabled if we have a player to draft AND that player is not already drafted or blocked
        can_draft = (
                player_to_draft is not None
                and player_to_draft not in st.session_state.drafted_players
                and player_to_draft not in st.session_state.unavailable_players
        )

        if st.button("📝 Draft", disabled=not can_draft, width='stretch'):
            if player_to_draft:
                st.session_state.drafted_players.append(player_to_draft)
                st.session_state.clear_selection = True
                st.session_state.selected_recommendation = None  # Clear recommendation selection

                # Increment selection counter to force clear recommendation table selection
                if 'rec_selection_counter' not in st.session_state:
                    st.session_state.rec_selection_counter = 0
                st.session_state.rec_selection_counter += 1

                st.rerun()

    with button_col2:
        # Block button: only enabled if at least 1 player selected AND all selected players are not already drafted or blocked
        can_block = (
                len(selected_golfers) > 0
                and all(
            golfer not in st.session_state.drafted_players
            and golfer not in st.session_state.unavailable_players
            for golfer in selected_golfers
        )
        )

        if st.button("🚫 Block", disabled=not can_block, width='stretch'):
            for golfer in selected_golfers:
                if golfer not in st.session_state.unavailable_players:
                    st.session_state.unavailable_players.append(golfer)
            st.session_state.clear_selection = True
            st.rerun()

    with button_col3:
        # Restore button: only enabled if at least 1 player selected AND at least one is drafted or blocked
        can_restore = (
                len(selected_golfers) > 0
                and any(
            golfer in st.session_state.drafted_players
            or golfer in st.session_state.unavailable_players
            for golfer in selected_golfers
        )
        )

        if st.button("↩️ Restore", disabled=not can_restore, width='stretch'):
            for golfer in selected_golfers:
                if golfer in st.session_state.drafted_players:
                    st.session_state.drafted_players.remove(golfer)
                if golfer in st.session_state.unavailable_players:
                    st.session_state.unavailable_players.remove(golfer)
            st.session_state.clear_selection = True
            st.rerun()

# ============================================================
# MIDDLE COLUMN: Controls & My Team
# ============================================================
with col2:
    st.subheader("Controls")

    # Round Multipliers
    with st.expander("⚙️ Round Multipliers", expanded=False):
        # Initialize reset counter if not exists
        if 'multiplier_reset_counter' not in st.session_state:
            st.session_state.multiplier_reset_counter = 0

        r1 = st.number_input(
            "Round 1",
            min_value=0.05,
            max_value=2.0,
            value=st.session_state.r1_mult,
            step=0.05,
            format="%.2f",
            key=f"r1_input_{st.session_state.multiplier_reset_counter}"
        )
        r2 = st.number_input(
            "Round 2",
            min_value=0.05,
            max_value=2.0,
            value=st.session_state.r2_mult,
            step=0.05,
            format="%.2f",
            key=f"r2_input_{st.session_state.multiplier_reset_counter}"
        )
        r3 = st.number_input(
            "Round 3",
            min_value=0.05,
            max_value=2.0,
            value=st.session_state.r3_mult,
            step=0.05,
            format="%.2f",
            key=f"r3_input_{st.session_state.multiplier_reset_counter}"
        )
        r4 = st.number_input(
            "Round 4",
            min_value=0.05,
            max_value=2.0,
            value=st.session_state.r4_mult,
            step=0.05,
            format="%.2f",
            key=f"r4_input_{st.session_state.multiplier_reset_counter}"
        )

        if st.button("Reset to Defaults"):
            # Update session state with default values
            st.session_state.r1_mult = 1.15
            st.session_state.r2_mult = 1.05
            st.session_state.r3_mult = 1.00
            st.session_state.r4_mult = 0.95

            # Increment counter to force recreation of inputs with new values
            st.session_state.multiplier_reset_counter += 1

            st.rerun()
    
    # Run Model Button
    if st.button("🎯 Run Model", type="primary", width='stretch'):
        # Update multipliers in model
        model.set_round_multipliers(r1, r2, r3, r4)
        st.session_state.r1_mult = r1
        st.session_state.r2_mult = r2
        st.session_state.r3_mult = r3
        st.session_state.r4_mult = r4
        st.session_state.model_ran = True
        # Recommendations will be calculated below

    st.markdown("---")

    # My Team
    st.subheader("My Team")

    if st.session_state.drafted_players:
        team_df = pd.DataFrame({
            'Pick': range(1, len(st.session_state.drafted_players) + 1),
            'Player': st.session_state.drafted_players
        })
        st.dataframe(
            team_df,
            hide_index=True,
            height=458,
            width='stretch',
            column_config={
                "Pick": st.column_config.TextColumn("Pick", width="small"),
                "Player": st.column_config.TextColumn("Player", width="medium"),
            }
        )

        # Show team score
        total_score = model.total_best_ball_score(st.session_state.drafted_players)
        st.metric("Total Best Ball Score", f"{total_score:.1f}")
    else:
        st.info("No players drafted yet")

# ============================================================
# RIGHT COLUMN: Event Coverage
# ============================================================
with col3:
    st.subheader("Live Event Coverage")
    
    if st.session_state.drafted_players:
        coverage_df = format_coverage_table()
        
        # Create a styled display
        st.dataframe(
            coverage_df,
            hide_index=True,
            height=948,
            width='stretch',
            column_config={
                "Event": st.column_config.TextColumn("Event", width="small"),
                "Round": st.column_config.TextColumn("Round", width="small"),
                "Covered": st.column_config.TextColumn("Covered", width="small"),
                "Needed": st.column_config.TextColumn("Needed", width="small"),
                "Remaining Picks": st.column_config.TextColumn("Remaining Picks", width="small"),
            }
        )

        # Since row formatting is not allowed, there is no need for the legend.
        # # Legend
        # st.markdown("**Round Legend:**")
        # legend_cols = st.columns(4)
        # with legend_cols[0]:
        #     st.markdown("🟦 Round 1")
        # with legend_cols[1]:
        #     st.markdown("🟪 Round 2")
        # with legend_cols[2]:
        #     st.markdown("🟩 Round 3")
        # with legend_cols[3]:
        #     st.markdown("🟫 Round 4")
    else:
        st.info("Draft players to see coverage")

# ============================================================
# BOTTOM SECTION: Player Recommendations
# ============================================================

st.markdown("---")
st.subheader("Player Recommendations")

if st.session_state.model_ran:
    round_number = len(st.session_state.drafted_players) + 1

    try:
        # Get recommendations
        recs = model.recommend_players_fast(
            st.session_state.drafted_players,
            st.session_state.unavailable_players,
            round_number
        )

        if not recs.empty:
            # Format ADP column
            recs['ADP'] = recs['ADP'].apply(lambda x: "-" if x >= 999 else f"{x:.1f}")

            # Rename columns for better display
            display_recs = recs.rename(columns={
                'Player': 'Name',
                'Score': 'Points Added'
            })

            # Format Points Added
            display_recs['Points Added'] = display_recs['Points Added'].apply(lambda x: f"{x:.1f}")

            # Add selection to recommendations
            rec_event = st.dataframe(
                display_recs,
                hide_index=True,
                height=387,
                width='stretch',
                on_select="rerun",
                selection_mode="single-row",
                key=f"recommendations_selection_{st.session_state.get('rec_selection_counter', 0)}",
                column_config={
                    "Name": st.column_config.TextColumn("Name", width="small"),
                    "Points Added": st.column_config.TextColumn("Points Added", width="small"),
                    "ADP": st.column_config.TextColumn("ADP", width="small"),
                    "Event Impact": st.column_config.TextColumn("Event Impact", width="large"),
                }
            )

            # Store selected recommendation in session state
            selected_rec_indices = rec_event.selection.rows if rec_event.selection else []
            new_selection = display_recs.iloc[selected_rec_indices[0]]['Name'] if selected_rec_indices else None

            # Check if selection changed
            old_selection = st.session_state.get('selected_recommendation')
            if new_selection != old_selection:
                st.session_state.selected_recommendation = new_selection

                # If a recommendation is selected, clear All Players selection
                if new_selection:
                    st.session_state.clear_selection = True
                    if 'selection_counter' not in st.session_state:
                        st.session_state.selection_counter = 0
                    st.session_state.selection_counter += 1

                # Force another rerun so the Draft button updates immediately
                st.rerun()

        else:
            st.info("No recommendations available")
    except Exception as e:
        st.error(f"Error generating recommendations: {str(e)}")
else:
    st.info("Click 'Run Model' to see recommendations")
    st.session_state.selected_recommendation = None  # Clear if model hasn't run

# ============================================================
# SIDEBAR: Additional Info
# ============================================================
with st.sidebar:
    st.header("Draft Statistics")
    
    st.metric("Players Drafted", len(st.session_state.drafted_players))
    st.metric("Players Blocked", len(st.session_state.unavailable_players))
    st.metric("Remaining Picks", max(0, model.TEAM_SIZE - len(st.session_state.drafted_players)))
    
    st.markdown("---")
    
    if st.button("🔄 Reset Draft", width='stretch'):
        st.session_state.drafted_players = []
        st.session_state.unavailable_players = []
        st.session_state.model_ran = False
        st.rerun()
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("PGA Best Ball Draft Tool helps you make optimal picks during your fantasy golf draft.")
    st.markdown("**Team Size:** 12 players")
    st.markdown("**Min Players/Event:** 6 players")
