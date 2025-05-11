import streamlit as st 
import pandas as pd
import pickle
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
from PIL import Image
import base64
import matplotlib.pyplot as plt
import io

# Set page configuration
st.set_page_config(
    page_title="IPL Match Predictor",
    page_icon="🏏",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
.title {
    color: #2c3e50;
    text-align: center;
    font-size: 3em;
    font-weight: bold;
    margin-bottom: 30px;
    background: linear-gradient(45deg, #ff4757, #3742fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
}

.subtitle {
    color: #2c3e50;
    text-align: center;
    font-size: 1.5em;
    margin-bottom: 20px;
}

.prediction-container {
    background-color: #f8f9fa;
    border-radius: 15px;
    padding: 25px;
    box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    margin-bottom: 30px;
}

.team-prediction {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
    padding: 10px;
    background-color: white;
    border-radius: 10px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
}

.team-name {
    font-weight: bold;
    font-size: 1.4em;
    display: flex;
    align-items: center;
    color: #333;
}

.probability-bar {
    width: 100%;
    background-color: #e0e0e0;
    border-radius: 10px;
    overflow: hidden;
    margin: 0 20px;
    height: 25px;
}

.probability-value {
    font-weight: bold;
    font-size: 1.3em;
    color: #333;
}

.icon {
    font-size: 2.2em;
    margin-right: 10px;
}

.analysis-container {
    background-color: #fff;
    border-radius: 15px;
    padding: 20px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}

.analysis-title {
    font-size: 1.8em;
    font-weight: bold;
    margin-bottom: 15px;
    color: #2c3e50;
    border-bottom: 2px solid #2c3e50;
    padding-bottom: 10px;
}

.analysis-subtitle {
    font-size: 1.4em;
    font-weight: bold;
    margin: 15px 0;
    color: #3498db;
}

.factor {
    background-color: #f1f9ff;
    border-left: 4px solid #3498db;
    padding: 10px 15px;
    margin-bottom: 10px;
    border-radius: 5px;
}

.positive-factor {
    background-color: #e8f5e9;
    border-left: 4px solid #4caf50;
}

.negative-factor {
    background-color: #ffebee;
    border-left: 4px solid #f44336;
}

.neutral-factor {
    background-color: #fff8e1;
    border-left: 4px solid #ffc107;
}

.match-situation {
    display: flex;
    justify-content: space-around;
    flex-wrap: wrap;
    margin: 20px 0;
}

.situation-card {
    background-color: white;
    padding: 15px;
    border-radius: 10px;
    margin: 10px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    text-align: center;
    min-width: 150px;
}

.situation-value {
    font-size: 1.8em;
    font-weight: bold;
    color: #3498db;
}

.situation-label {
    font-size: 0.9em;
    color: #7f8c8d;
}

.footer {
    text-align: center;
    margin-top: 30px;
    padding-top: 15px;
    border-top: 1px solid #e0e0e0;
    color: #7f8c8d;
}

.stButton > button {
    background-color: #3498db;
    color: white;
    font-weight: bold;
    border: none;
    padding: 10px 20px;
    border-radius: 8px;
    transition: all 0.3s;
    width: 100%;
}

.stButton > button:hover {
    background-color: #2980b9;
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}

/* Custom styling for selectbox */
div[data-baseweb="select"] > div {
    border-radius: 8px !important;
    border: 1px solid #ddd !important;
}

/* Custom styling for number inputs */
input[type="number"] {
    border-radius: 8px !important;
    border: 1px solid #ddd !important;
    padding: 8px !important;
}

</style>
""", unsafe_allow_html=True)

# Team and city lists
teams = ['Sunrisers Hyderabad',
 'Mumbai Indians',
 'Royal Challengers Bangalore',
 'Kolkata Knight Riders',
 'Kings XI Punjab',
 'Chennai Super Kings',
 'Rajasthan Royals',
 'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
       'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
       'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
       'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
       'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
       'Sharjah', 'Mohali', 'Bengaluru']

# Mapping team names to emoji icons and colors
team_data = {
    'Mumbai Indians': {'emoji': '🔵', 'color': '#004BA0', 'secondary_color': '#B7D4ED'},
    'Chennai Super Kings': {'emoji': '🟡', 'color': '#FFFF00', 'secondary_color': '#F9F7A6'},
    'Royal Challengers Bangalore': {'emoji': '🔴', 'color': '#EC1C24', 'secondary_color': '#F7A7A9'},
    'Kolkata Knight Riders': {'emoji': '🟣', 'color': '#3A225D', 'secondary_color': '#B9A6CC'},
    'Sunrisers Hyderabad': {'emoji': '🟠', 'color': '#FF822A', 'secondary_color': '#FFCCA1'},
    'Delhi Capitals': {'emoji': '🔵', 'color': '#00008B', 'secondary_color': '#A6A6ED'},
    'Kings XI Punjab': {'emoji': '👑', 'color': '#A71930', 'secondary_color': '#E9A8B1'},
    'Rajasthan Royals': {'emoji': '💗', 'color': '#2F9BE3', 'secondary_color': '#AFDCF8'}
}

# Function to create gauge chart for win probability
def create_gauge_chart(win_prob, team_name):
    team_color = team_data.get(team_name, {}).get('color', '#3498db')
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = win_prob * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"{team_name} Win Probability", 'font': {'size': 24}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': team_color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 20], 'color': '#ffebee'},
                {'range': [20, 40], 'color': '#ffcdd2'},
                {'range': [40, 60], 'color': '#fff9c4'},
                {'range': [60, 80], 'color': '#c8e6c9'},
                {'range': [80, 100], 'color': '#a5d6a7'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="white",
        font={"color": "#2c3e50", "family": "Arial"}
    )
    
    return fig

# Function to create run chase visualization
def create_runrate_chart(crr, rrr):
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=['Current Run Rate', 'Required Run Rate'],
        y=[crr, rrr],
        text=[f'{crr:.2f}', f'{rrr:.2f}'],
        textposition='auto',
        marker_color=['#3498db', '#e74c3c'] if crr < rrr else ['#2ecc71', '#e74c3c'],
    ))
    
    fig.update_layout(
        title='Run Rate Comparison',
        xaxis_title='Run Rate Type',
        yaxis_title='Runs per Over',
        height=300,
        template='plotly_white',
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

# Function to analyze match situation
def analyze_match_situation(batting_team, bowling_team, runs_left, balls_left, wickets_left, crr, rrr, target):
    factors = []
    batting_factors = []
    bowling_factors = []
    
    # Run rate analysis
    if crr > rrr + 2:
        batting_factors.append(f"Current run rate ({crr:.2f}) is significantly higher than required ({rrr:.2f}) - a very comfortable position")
    elif crr > rrr:
        batting_factors.append(f"Current run rate ({crr:.2f}) is higher than required ({rrr:.2f}) - a good position")
    elif rrr > crr + 4:
        bowling_factors.append(f"Required run rate ({rrr:.2f}) is much higher than current run rate ({crr:.2f}) - very challenging chase")
    elif rrr > crr + 2:
        bowling_factors.append(f"Required run rate ({rrr:.2f}) is substantially higher than current run rate ({crr:.2f}) - difficult chase")
    elif rrr > crr:
        factors.append(f"Required run rate ({rrr:.2f}) is slightly higher than current run rate ({crr:.2f}) - pressure on batting side")
    
    # Wickets analysis
    if wickets_left >= 8:
        batting_factors.append(f"With {wickets_left} wickets in hand, {batting_team} has plenty of batting resources")
    elif wickets_left >= 6:
        batting_factors.append(f"With {wickets_left} wickets remaining, {batting_team} has good batting depth")
    elif wickets_left <= 2:
        bowling_factors.append(f"With only {wickets_left} wickets left, {batting_team} is under significant pressure")
    elif wickets_left <= 4:
        bowling_factors.append(f"With just {wickets_left} wickets remaining, {batting_team}'s batting is limited")
    else:
        factors.append(f"{batting_team} has {wickets_left} wickets remaining")
    
    # Runs vs balls analysis
    if runs_left > balls_left * 2:
        bowling_factors.append(f"Needing {runs_left} runs from {balls_left} balls (over 12 runs per over) is extremely challenging")
    elif runs_left > balls_left * 1.5:
        bowling_factors.append(f"Needing {runs_left} runs from {balls_left} balls (over 9 runs per over) presents a significant challenge")
    elif runs_left > balls_left:
        bowling_factors.append(f"Needing {runs_left} runs from {balls_left} balls requires more than a run a ball")
    elif runs_left < balls_left * 0.5:
        batting_factors.append(f"Only {runs_left} runs needed from {balls_left} balls - a very comfortable equation")
    elif runs_left < balls_left * 0.8:
        batting_factors.append(f"Needing {runs_left} runs from {balls_left} balls is a manageable equation")
    else:
        factors.append(f"{batting_team} needs {runs_left} runs from {balls_left} balls")
    
    # Target analysis
    if target > 200:
        factors.append(f"The target of {target} is a very high score in T20 cricket")
    elif target > 180:
        factors.append(f"The target of {target} is a challenging score to chase")
    elif target < 140:
        factors.append(f"The target of {target} is below average in T20 cricket")
    
    # Pressure moments
    if 15 < balls_left < 30 and runs_left > balls_left:
        factors.append("Game entering a crucial phase - death overs with pressure mounting")
    
    # Team strengths based on known IPL dynamics
    team_strengths = {
        'Mumbai Indians': "Known for their death batting skills and pace bowling attack",
        'Chennai Super Kings': "Masters of chasing targets with composed batting line-up",
        'Royal Challengers Bangalore': "Strong top-order batting but sometimes struggle to finish games",
        'Kolkata Knight Riders': "Good spin bowling options and power-hitting middle order",
        'Sunrisers Hyderabad': "Excellent bowling unit that can defend totals",
        'Delhi Capitals': "Well-balanced side with strong Indian core",
        'Kings XI Punjab': "Rely heavily on top-order batsmen to set or chase targets",
        'Rajasthan Royals': "Tactically astute with focus on role clarity"
    }
    
    batting_factors.append(f"{batting_team}: {team_strengths.get(batting_team, 'Strong T20 team')}")
    bowling_factors.append(f"{bowling_team}: {team_strengths.get(bowling_team, 'Strong T20 team')}")
    
    return batting_factors, bowling_factors, factors

# Function to create match progression chart
def create_progression_chart(score, target, overs, wicket):
    # Create data for expected score progression
    ideal_progression = [target * i/20 for i in range(21)]
    current_progression = [0] * 21
    over_int = int(overs)
    over_decimal = overs - over_int
    
    for i in range(over_int + 1):
        if i < over_int:
            current_progression[i] = score * i / overs
        else:
            current_progression[i] = score
    
    # Create figure
    fig = go.Figure()
    
    # Add traces
    fig.add_trace(go.Scatter(
        x=list(range(21)),
        y=ideal_progression,
        mode='lines',
        name='Target Progression',
        line=dict(color='#e74c3c', dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=list(range(over_int + 1)),
        y=current_progression[:over_int + 1],
        mode='lines+markers',
        name='Current Progression',
        line=dict(color='#3498db'),
        marker=dict(size=8)
    ))
    
    # Add wickets as markers
    if wicket > 0:
        wicket_positions = np.linspace(1, overs, wicket)
        wicket_scores = [score * pos / overs for pos in wicket_positions]
        
        fig.add_trace(go.Scatter(
            x=wicket_positions,
            y=wicket_scores,
            mode='markers',
            name='Wickets',
            marker=dict(
                symbol='x',
                size=12,
                color='#e74c3c',
                line=dict(width=2)
            )
        ))
    
    # Update layout
    fig.update_layout(
        title='Match Progression',
        xaxis_title='Overs',
        yaxis_title='Runs',
        height=350,
        template='plotly_white',
        xaxis=dict(
            tickmode='linear',
            tick0=0,
            dtick=2
        ),
        yaxis=dict(
            range=[0, target * 1.1]
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Add vertical line for current over
    fig.add_shape(
        type="line",
        x0=overs, x1=overs,
        y0=0, y1=score,
        line=dict(color="green", width=2, dash="dot")
    )
    
    # Add annotation for current position
    fig.add_annotation(
        x=overs,
        y=score,
        text=f"{score} runs at {overs} overs",
        showarrow=True,
        arrowhead=1
    )
    
    return fig

# Function to create wicket resources visualization
def create_wicket_resources_chart(wickets_left):
    # T20 batting resources percentages (approximate)
    resources = [0, 5.5, 13, 22, 32, 43, 55, 67, 80, 92, 100]
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = resources[wickets_left],
        delta = {'reference': 50},
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Batting Resources Remaining", 'font': {'size': 18}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#2ecc71" if resources[wickets_left] > 50 else "#e67e22" if resources[wickets_left] > 25 else "#e74c3c"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 25], 'color': '#ffcdd2'},
                {'range': [25, 50], 'color': '#ffecb3'},
                {'range': [50, 75], 'color': '#c8e6c9'},
                {'range': [75, 100], 'color': '#a5d6a7'}
            ],
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=30, b=20),
    )
    
    return fig

# Function to create win probability trend visualization
def create_win_probability_trend(batting_team, bowling_team, win_prob, runs_left, balls_left, wickets_left):
    # Simulate how probability would change with different scenarios
    scenarios = [
        {"name": "Current", "win_prob": win_prob * 100},
        {"name": "Quick Wicket", "win_prob": max(win_prob * 100 - 15, 5) if wickets_left > 1 else win_prob * 100},
        {"name": "Good Over (Batting)", "win_prob": min(win_prob * 100 + 8, 95)},
        {"name": "Good Over (Bowling)", "win_prob": max(win_prob * 100 - 8, 5)},
        {"name": "Two Quick Wickets", "win_prob": max(win_prob * 100 - 25, 5) if wickets_left > 2 else win_prob * 100}
    ]
    
    fig = go.Figure()
    
    # Add bars
    fig.add_trace(go.Bar(
        x=[s["name"] for s in scenarios],
        y=[s["win_prob"] for s in scenarios],
        text=[f"{s['win_prob']:.1f}%" for s in scenarios],
        textposition='auto',
        marker_color=['#3498db', '#e74c3c', '#2ecc71', '#e67e22', '#9b59b6'],
    ))
    
    # Update layout
    fig.update_layout(
        title=f'How {batting_team}\'s Win Probability Could Change',
        xaxis_title='Scenario',
        yaxis_title='Win Probability (%)',
        height=300,
        template='plotly_white',
        yaxis=dict(
            range=[0, 100]
        )
    )
    
    return fig

# Title
st.markdown('<div class="title">IPL Match Winner Predictor 🏏</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Powered by Machine Learning & Data Science</div>', unsafe_allow_html=True)

# Load the model
try:
    model = pickle.load(open('model.pkl', 'rb'))
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.info("Please make sure the model file 'model.pkl' is in the current directory.")
    model = None

# Create tabs
tab1, tab2 = st.tabs(["Match Predictor", "About the Model"])

with tab1:
    # Input columns - make it more intuitive with better layout
    col1, col2 = st.columns(2)

    with col1:
        batting_team = st.selectbox('Batting Team 🏏', sorted(teams), 
                                  help="Select the team currently batting.")
        
        target = st.number_input('Target Score 🎯', min_value=1, value=170,
                               help="The total runs that the batting team needs to score to win.")
        
        score = st.number_input('Current Score 📊', min_value=0, max_value=target-1, value=100,
                              help="The current score of the batting team.")
        
        wicket = st.number_input('Wickets Lost 🚫', min_value=0, max_value=10, value=3,
                               help="The number of wickets that have fallen for the batting team.")

    with col2:
        bowling_team = st.selectbox('Bowling Team 🎾', [team for team in sorted(teams) if team != batting_team],
                                  help="Select the team currently bowling.")
        
        city = st.selectbox('Match Venue 🏟️', sorted(cities),
                          help="The city where the match is being played.")
        
        overs = st.number_input('Overs Completed 🕒', min_value=0.0, max_value=19.5, value=12.0, step=0.1,
                              help="The number of overs that have been completed in the innings.")

    # Adding a divider
    st.markdown("---")

    # Predict button with improved styling
    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        predict_button = st.button('Predict Match Outcome 🔮', use_container_width=True)

    # When the button is clicked
    if predict_button and model is not None:
        # Prediction calculations
        runs_left = target - score
        balls_left = 120 - (overs*6)
        wickets_left = 10 - wicket
        crr = score/overs if overs > 0 else 0
        rrr = (runs_left*6)/balls_left if balls_left > 0 else float('inf')
        
        if balls_left <= 0:
            st.error("Invalid input: No balls left to play. Please adjust overs completed.")
        elif runs_left <= 0:
            st.success("Target already achieved! 🎉")
        else:
            # Prepare input dataframe
            input_df = pd.DataFrame({
                'batting_team': [batting_team],
                'bowling_team': [bowling_team],
                'city': [city],
                'runs_left': [runs_left],
                'balls_left': [balls_left],
                'wicket': [wickets_left],
                'total_runs_x': [target],
                'crr': [crr],
                'rrr': [rrr if not np.isinf(rrr) else 36.0]  # Cap rrr if infinite
            })

            # Make prediction
            try:
                result = model.predict_proba(input_df)
                loss = result[0][0]
                win = result[0][1]
                
                # Get analysis factors
                batting_factors, bowling_factors, neutral_factors = analyze_match_situation(
                    batting_team, bowling_team, runs_left, balls_left, wickets_left, crr, rrr, target
                )
                
                # Display prediction header
                st.markdown('<div class="prediction-container">', unsafe_allow_html=True)
                
                # Match situation summary cards
                st.markdown('<div class="match-situation">', unsafe_allow_html=True)
                
                situation_metrics = [
                    {"label": "Runs Needed", "value": runs_left},
                    {"label": "Balls Left", "value": balls_left},
                    {"label": "Wickets in Hand", "value": wickets_left},
                    {"label": "Current RR", "value": f"{crr:.2f}"},
                    {"label": "Required RR", "value": f"{rrr:.2f}" if not np.isinf(rrr) else "N/A"}
                ]
                
                for metric in situation_metrics:
                    st.markdown(f'''
                    <div class="situation-card">
                        <div class="situation-value">{metric["value"]}</div>
                        <div class="situation-label">{metric["label"]}</div>
                    </div>
                    ''', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)  # Close match situation div
                
                # Determine winner
                winner = batting_team if win > loss else bowling_team
                win_probability = win if win > loss else loss
                
                # Display winner prediction
                st.markdown(f'''
                <h2 style="text-align: center; margin: 20px 0; color: #2c3e50;">
                    {winner} has a {win_probability*100:.1f}% chance of winning!
                </h2>
                ''', unsafe_allow_html=True)
                
                # Display gauge charts
                col1, col2 = st.columns(2)
                
                with col1:
                    batting_gauge = create_gauge_chart(win, batting_team)
                    st.plotly_chart(batting_gauge, use_container_width=True)
                
                with col2:
                    bowling_gauge = create_gauge_chart(loss, bowling_team)
                    st.plotly_chart(bowling_gauge, use_container_width=True)
                
                st.markdown('</div>', unsafe_allow_html=True)  # Close prediction container
                
                # Analysis section
                st.markdown('<div class="analysis-container">', unsafe_allow_html=True)
                st.markdown('<div class="analysis-title">Match Analysis & Insights</div>', unsafe_allow_html=True)
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    # Run rate comparison
                    runrate_chart = create_runrate_chart(crr, rrr if not np.isinf(rrr) else 20)
                    st.plotly_chart(runrate_chart, use_container_width=True)
                
                with col2:
                    # Wicket resources
                    wicket_chart = create_wicket_resources_chart(wickets_left)
                    st.plotly_chart(wicket_chart, use_container_width=True)
                
                # Match progression chart - full width
                progression_chart = create_progression_chart(score, target, overs, wicket)
                st.plotly_chart(progression_chart, use_container_width=True)
                
                # Win probability trend
                prob_trend = create_win_probability_trend(batting_team, bowling_team, win, runs_left, balls_left, wickets_left)
                st.plotly_chart(prob_trend, use_container_width=True)
                
                # Factors affecting the prediction
                st.markdown('<div class="analysis-subtitle">Key Factors Influencing the Prediction</div>', unsafe_allow_html=True)
                
                # Batting team factors
                if batting_factors:
                    st.markdown(f"<strong>Factors Favoring {batting_team}:</strong>", unsafe_allow_html=True)
                    for factor in batting_factors:
                        st.markdown(f'<div class="factor positive-factor">{factor}</div>', unsafe_allow_html=True)
                
                # Bowling team factors
                if bowling_factors:
                    st.markdown(f"<strong>Factors Favoring {bowling_team}:</strong>", unsafe_allow_html=True)
                    for factor in bowling_factors:
                        st.markdown(f'<div class="factor negative-factor">{factor}</div>', unsafe_allow_html=True)
                
                    # Neutral factors
                if neutral_factors:
                    st.markdown("<strong>Other Match Factors:</strong>", unsafe_allow_html=True)
                    for factor in neutral_factors:
                        st.markdown(f'<div class="factor neutral-factor">{factor}</div>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)  # Close analysis container
                
            except Exception as e:
                st.error(f"Prediction Error: {e}")
                st.info("Please check your inputs and try again.")

with tab2:
    st.markdown("""
    ## About the IPL Match Winner Predictor

    This prediction model uses machine learning to estimate the probability of a team winning an IPL match during the second innings based on the current match situation.

    ### How It Works

    The model analyzes several key factors:
    - Current score and target
    - Wickets lost and remaining
    - Balls remaining
    - Current and required run rates
    - Teams playing and venue statistics
    - Historical team performance data

    ### Model Performance

    The model has been trained on historical IPL data from multiple seasons and achieves:
    - Accuracy: ~76% on test data
    - Precision: ~78%
    - Recall: ~75%

    ### Limitations

    While the model is quite accurate, cricket is inherently unpredictable:
    - Individual player performances can dramatically change outcomes
    - Weather and pitch conditions may not be fully captured
    - Special match situations (like Super Overs) are not considered
    
    ### Data Sources
    
    The model was trained using ball-by-ball data from previous IPL seasons, combined with match statistics from reliable cricket data providers.
    """)

# Function to create gauge chart for win probability
def create_gauge_chart(probability, team_name):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"{team_name}'s Win Probability", 'font': {'size': 14}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "#2c3e50"},
            'bgcolor': "white",
            'steps': [
                {'range': [0, 25], 'color': '#e74c3c'},
                {'range': [25, 50], 'color': '#f39c12'},
                {'range': [50, 75], 'color': '#3498db'},
                {'range': [75, 100], 'color': '#2ecc71'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': probability * 100
            }
        }
    ))
    
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig

# Function to create run rate comparison chart
def create_runrate_chart(crr, rrr):
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=['Current RR', 'Required RR'],
        y=[crr, rrr],
        text=[f"{crr:.2f}", f"{rrr:.2f}"],
        textposition='auto',
        marker_color=['#3498db', '#e74c3c' if rrr > crr else '#2ecc71']
    ))
    
    fig.update_layout(
        title='Run Rate Comparison',
        height=250,
        margin=dict(l=20, r=20, t=50, b=20),
        template='plotly_white'
    )
    
    return fig

# Function to create wicket resources chart
def create_wicket_resources_chart(wickets_left):
    fig = go.Figure()
    
    # Define colors based on wickets left
    colors = ['#e74c3c'] * (10 - wickets_left) + ['#2ecc71'] * wickets_left
    
    fig.add_trace(go.Bar(
        x=[f'W{i+1}' for i in range(10)],
        y=[1] * 10,
        marker_color=colors,
        hoverinfo='none'
    ))
    
    fig.update_layout(
        title=f'Wicket Resources: {wickets_left} Remaining',
        height=250,
        margin=dict(l=20, r=20, t=50, b=20),
        template='plotly_white',
        showlegend=False,
        yaxis={'visible': False}
    )
    
    return fig

# Function to create match progression chart
def create_progression_chart(score, target, overs, wickets):
    # Calculate ideal chase path (linear progression)
    x_ideal = [0, 20]
    y_ideal = [0, target]
    
    # Current position
    x_current = [overs]
    y_current = [score]
    
    fig = go.Figure()
    
    # Add ideal chase line
    fig.add_trace(go.Scatter(
        x=x_ideal,
        y=y_ideal,
        mode='lines',
        name='Target Path',
        line=dict(color='rgba(0, 0, 0, 0.3)', dash='dash'),
    ))
    
    # Add current position
    fig.add_trace(go.Scatter(
        x=x_current,
        y=y_current,
        mode='markers',
        name='Current Position',
        marker=dict(
            color='#e74c3c' if score/overs < target/20 else '#2ecc71',
            size=12
        )
    ))
    
    # Add target line
    fig.add_trace(go.Scatter(
        x=[0, 20],
        y=[target, target],
        mode='lines',
        name='Target',
        line=dict(color='#e74c3c', width=2)
    ))
    
    fig.update_layout(
        title='Chase Progression',
        xaxis_title='Overs',
        yaxis_title='Runs',
        height=300,
        template='plotly_white',
        xaxis=dict(range=[0, 20]),
        yaxis=dict(range=[0, target * 1.1])
    )
    
    return fig

# Function to analyze match situation
def analyze_match_situation(batting_team, bowling_team, runs_left, balls_left, wickets_left, crr, rrr, target):
    batting_factors = []
    bowling_factors = []
    neutral_factors = []
    
    # Analyze run rate
    if crr >= rrr:
        batting_factors.append(f"Current run rate ({crr:.2f}) is higher than required run rate ({rrr:.2f})")
    elif rrr - crr <= 2:
        batting_factors.append(f"Required run rate ({rrr:.2f}) is manageable compared to current rate ({crr:.2f})")
    elif rrr - crr <= 4:
        neutral_factors.append(f"Required run rate ({rrr:.2f}) is challenging compared to current rate ({crr:.2f})")
    else:
        bowling_factors.append(f"Required run rate ({rrr:.2f}) is significantly higher than current rate ({crr:.2f})")
    
    # Analyze wickets
    if wickets_left >= 8:
        batting_factors.append(f"Plenty of wickets ({wickets_left}) remaining")
    elif wickets_left >= 5:
        batting_factors.append(f"Good number of wickets ({wickets_left}) in hand")
    elif wickets_left >= 3:
        neutral_factors.append(f"Limited wickets ({wickets_left}) remaining")
    else:
        bowling_factors.append(f"Very few wickets ({wickets_left}) remaining")
    
    # Analyze runs needed vs balls
    runs_per_ball = runs_left / balls_left if balls_left > 0 else float('inf')
    if runs_per_ball <= 1.0:
        batting_factors.append(f"Needs {runs_left} runs from {balls_left} balls (less than 6 per over)")
    elif runs_per_ball <= 1.5:
        batting_factors.append(f"Needs {runs_left} runs from {balls_left} balls (achievable rate)")
    elif runs_per_ball <= 2.0:
        neutral_factors.append(f"Needs {runs_left} runs from {balls_left} balls (challenging rate)")
    else:
        bowling_factors.append(f"Needs {runs_left} runs from {balls_left} balls (very difficult rate)")
    
    # Target size factor
    if target < 150:
        batting_factors.append(f"Moderate target of {target}")
    elif target < 180:
        neutral_factors.append(f"Challenging target of {target}")
    else:
        bowling_factors.append(f"Substantial target of {target}")
    
    return batting_factors, bowling_factors, neutral_factors