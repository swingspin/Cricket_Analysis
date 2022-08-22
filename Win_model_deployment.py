import streamlit as st
import pickle
import pandas as pd
#pip install sklearn


teams = ['India',
 'Afghanistan',
 'Australia',
 'South Africa',
 'New Zealand',
 'West Indies',
 'Sri Lanka',
 'Bangladesh',
 'Ireland',
 'England',
 'Pakistan',
 'Zimbabwe']

cities = ['Centurion', 'Dehra Dun', 'Adelaide', 'Brisbane', 'Melbourne',
       'Sydney', 'Canberra', 'Perth', 'East London', 'Durban',
       'Johannesburg', 'Port Elizabeth', 'Cape Town', 'Chandigarh',
       'Bengaluru', 'Delhi', 'Rajkot', 'Nagpur', 'Hyderabad',
       'Thiruvananthapuram', 'Mumbai', 'Christchurch', 'Wellington',
       'Nelson', 'Napier', 'Auckland', 'Hamilton', 'Mount Maunganui',
       'Lauderhill', 'Providence', 'Kandy', 'Lucknow', 'Dhaka',
       'Chattogram', 'Southampton', 'Manchester', 'Lahore', 'Indore',
       'Pune', "St George's", 'Basseterre', 'Greater Noida',
       'Dunedin', 'Paarl', 'Leeds', 'Ahmedabad', 'Cardiff', 'Dublin',
       'Belfast', 'Abu Dhabi', 'Coolidge', 'Bridgetown', 'Harare',
       'Colombo', 'Gros Islet', 'Bready', 'Dubai', 'Sharjah',
       'New Zealand', 'Australia', 'Jaipur', 'Ranchi', 'Kolkata',
       'Dharamsala', 'Karachi', 'Bangalore', 'Victoria', 'Taunton',
       'Chester-le-Street', 'Kanpur', 'Hobart', 'Bloemfontein',
       'Potchefstroom', 'Barbados', 'Trinidad', 'St Kitts', 'Jamaica',
       'Guwahati', 'Birmingham', 'Bristol', 'Cuttack', 'India',
       'Dehradun', 'Sylhet', 'Chennai', 'Visakhapatnam', 'Derry']

pipe = pickle.load(open('win_model1.pkl','rb'))
st.title('T20I Win Predictor, by Parth')

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team',sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the bowling team',sorted(teams))

selected_city = st.selectbox('Select host city',sorted(cities))

target = st.number_input('Target')

col3,col4,col5 = st.columns(3)

with col3:
    score = st.number_input('Score')
with col4:
    overs = st.number_input('Overs completed')
with col5:
    wickets = st.number_input('Wickets out')

if st.button('Predict Probability'):
    runs_left = target - score
    balls_left = 120 - (overs*6)
    wickets = 10 - wickets
    crr = score/overs
    rrr = (runs_left*6)/balls_left

    input_df = pd.DataFrame({'batting_team':[batting_team],'bowling_team':[bowling_team],'city':[selected_city],'runs_left':[runs_left],'balls_left':[balls_left],'wickets':[wickets],'total_runs_x':[target],'crr':[crr],'rrr':[rrr]})

    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]
    st.header(batting_team + "- " + str(round(win*100)) + "%")
    st.header(bowling_team + "- " + str(round(loss*100)) + "%")