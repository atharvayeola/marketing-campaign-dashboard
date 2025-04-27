# Step 1: Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visualization style
sns.set(style="whitegrid")

# Step 2: Load the Dataset
data = pd.read_csv('marketing_campaign_dataset.csv')

# Display the first few rows of the dataset
print("Initial Data Sample:")
print(data.head())

# Check data structure and types
print("\nDataset Info:")
print(data.info())

# Step 3: Data Cleaning and Transformation

# 3.1 Convert Acquisition_Cost (a currency string) to a numeric value
# Example: "$16,174.00" should become 16174.00
data['Acquisition_Cost'] = (
    data['Acquisition_Cost']
    .str.replace('$', '')
    .str.replace(',', '')
    .astype(float)
)

# 3.2 Convert Duration from string (e.g., "30 days") to integer (number of days)
data['Duration'] = data['Duration'].str.extract('(\d+)').astype(int)

# 3.3 Convert Date string to datetime object
data['Date'] = pd.to_datetime(data['Date'])

# 3.4 Extract Day and Month from Date
data['Day'] = data['Date'].dt.day
data['Month'] = data['Date'].dt.month

# 3.5 Create a new metric: Click-Through Rate (CTR) = Clicks / Impressions
# Handling potential division by zero safely
data['CTR'] = data['Clicks'] / data['Impressions'].replace(0, np.nan)
data['CTR'] = data['CTR'].fillna(0)

# 3.6 Create Month-Year feature for temporal trend analysis
data['Month_Year'] = data['Date'].dt.to_period('M').astype(str)

# Display the updated dataset sample with new columns
print("\nData Sample After Cleaning and Feature Engineering:")
print(data.head())
data.to_csv('cleaned_marketing_data.csv')


# 4.1 Campaign Performance Aggregation
campaign_perf = data.groupby('Campaign_Type').agg({
    'Conversion_Rate': 'mean',
    'ROI': 'mean'
}).reset_index()

# 4.2 Acquisition Cost by Channel
acq_cost_channel = data.groupby('Channel_Used').agg({
    'Acquisition_Cost': 'mean'
}).reset_index()

# 4.3 Audience Segmentation
audience_seg = data.groupby('Target_Audience').agg({
    'Conversion_Rate': 'mean',
    'Engagement_Score': 'mean'
}).reset_index()

customer_seg = data.groupby('Customer_Segment').agg({
    'Conversion_Rate': 'mean',
    'Engagement_Score': 'mean'
}).reset_index()

# 4.4 Channel Effectiveness
channel_effect = data.groupby('Channel_Used').agg({
    'Conversion_Rate': 'mean',
    'ROI': 'mean'
}).reset_index()

# 4.5 Geographical Insights
geo_insights = data.groupby('Location').agg({
    'ROI': 'mean'
}).reset_index()

# 4.6 Language Influence
language_influence = data.groupby('Language').agg({
    'Conversion_Rate': 'mean'
}).reset_index()

# 4.7 Temporal Trends (by Month-Year)
temporal_trends = data.groupby('Month_Year').agg({
    'Conversion_Rate': 'mean'
}).reset_index()


import plotly.express as px

# Campaign Performance Visuals
fig_conversion_by_campaign = px.bar(
    campaign_perf, x='Campaign_Type', y='Conversion_Rate',
    title='Average Conversion Rate by Campaign Type'
)

fig_roi_by_campaign = px.bar(
    campaign_perf, x='Campaign_Type', y='ROI',
    title='Average ROI by Campaign Type'
)

fig_acq_cost = px.bar(
    acq_cost_channel, x='Channel_Used', y='Acquisition_Cost',
    title='Average Acquisition Cost by Channel'
)

# Audience Segmentation Visuals
fig_audience_seg = px.bar(
    audience_seg, x='Target_Audience', y='Conversion_Rate',
    title='Conversion Rate by Target Audience'
)

fig_customer_seg = px.bar(
    customer_seg, x='Customer_Segment', y='Engagement_Score',
    title='Engagement Score by Customer Segment'
)

# Channel Effectiveness Visuals
fig_channel_conv = px.bar(
    channel_effect, x='Channel_Used', y='Conversion_Rate',
    title='Conversion Rate by Channel'
)

fig_channel_roi = px.bar(
    channel_effect, x='Channel_Used', y='ROI',
    title='ROI by Channel'
)

fig_duration_engagement = px.scatter(
    data, x='Duration', y='Engagement_Score',
    title='Campaign Duration vs Engagement Score'
)

# Geographical Insights Visual
fig_geo = px.bar(
    geo_insights, x='Location', y='ROI',
    title='Average ROI by Location'
)

# Language Influence Visual
fig_language = px.bar(
    language_influence, x='Language', y='Conversion_Rate',
    title='Average Conversion Rate by Language'
)

# Engagement vs Impressions Visual
fig_clicks_vs_impressions = px.scatter(
    data, x='Impressions', y='Clicks', color='Engagement_Score',
    title='Clicks vs Impressions (colored by Engagement Score)'
)

# Temporal Trends Visual
fig_temporal = px.line(
    temporal_trends, x='Month_Year', y='Conversion_Rate',
    title='Temporal Trends: Average Conversion Rate by Month'
)

# Cross Analysis Visuals
fig_roi_vs_engagement = px.scatter(
    data, x='Engagement_Score', y='ROI',
    title='ROI vs Engagement Score'
)

fig_conv_vs_cost = px.scatter(
    data, x='Acquisition_Cost', y='Conversion_Rate',
    title='Conversion Rate vs Acquisition Cost'
)

# Step 6: Build the Unified Dashboard with Tabs

import dash
from dash import dcc, html

# Initialize Dash app
app = dash.Dash(__name__)

# Layout of the app
app.layout = html.Div([
    html.H1("Marketing Campaign Performance Dashboard", style={'text-align': 'center'}),
    html.P("Explore different marketing insights by selecting tabs.", style={'text-align': 'center'}),

    dcc.Tabs([
        # Tab 1: Campaign Performance
        dcc.Tab(label='Campaign Performance', children=[
            dcc.Graph(figure=fig_conversion_by_campaign),
            dcc.Graph(figure=fig_roi_by_campaign),
            dcc.Graph(figure=fig_acq_cost)
        ]),

        # Tab 2: Audience Segmentation
        dcc.Tab(label='Audience Segmentation', children=[
            dcc.Graph(figure=fig_audience_seg),
            dcc.Graph(figure=fig_customer_seg)
        ]),

        # Tab 3: Channel Effectiveness
        dcc.Tab(label='Channel Effectiveness', children=[
            dcc.Graph(figure=fig_channel_conv),
            dcc.Graph(figure=fig_channel_roi),
            dcc.Graph(figure=fig_duration_engagement)
        ]),

        # Tab 4: Geographical Insights
        dcc.Tab(label='Geographical Insights', children=[
            dcc.Graph(figure=fig_geo)
        ]),

        # Tab 5: Language Influence
        dcc.Tab(label='Language Influence', children=[
            dcc.Graph(figure=fig_language)
        ]),

        # Tab 6: Engagement & Impressions
        dcc.Tab(label='Engagement & Impressions', children=[
            dcc.Graph(figure=fig_clicks_vs_impressions)
        ]),

        # Tab 7: Temporal Trends
        dcc.Tab(label='Temporal Trends', children=[
            dcc.Graph(figure=fig_temporal)
        ]),

        # Tab 8: Cross Analysis
        dcc.Tab(label='Cross Analysis', children=[
            dcc.Graph(figure=fig_roi_vs_engagement),
            dcc.Graph(figure=fig_conv_vs_cost)
        ])
    ])
])

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
