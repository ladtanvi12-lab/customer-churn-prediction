import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import dash
from dash import dcc, html, Input, Output
import numpy as np

print("Loading model and scaler...")
model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')

print("Loading data...")
df = pd.read_csv('cleaned_churn_data.csv')
df_original = df.copy()

print("Preprocessing...")
if 'Churn' in df.columns:
    df = df.drop('Churn', axis=1)

categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

training_columns = scaler.feature_names_in_
for col in training_columns:
    if col not in df_encoded.columns:
        df_encoded[col] = 0
df_encoded = df_encoded[training_columns]

print("Scaling and predicting...")
X_scaled = scaler.transform(df_encoded)
predictions = model.predict(X_scaled)
churn_probability = model.predict_proba(X_scaled)[:, 1]

df_original['Churn_Prediction'] = predictions
df_original['Churn_Probability'] = churn_probability
df_original['Risk_Level'] = pd.cut(churn_probability,
                                   bins=[0, 0.3, 0.7, 1],
                                   labels=['Low Risk', 'Medium Risk', 'High Risk'])

if hasattr(model, 'feature_importances_'):
    importance = model.feature_importances_
elif hasattr(model, 'coef_'):
    importance = abs(model.coef_[0])
feature_imp_df = pd.DataFrame({
    'Feature': training_columns,
    'Importance': importance
}).sort_values('Importance', ascending=False)

app = dash.Dash(__name__)

app.layout = html.Div(style={'backgroundColor': '#0e1117', 'padding': '20px'}, children=[
    html.H1('Customer Churn Prediction Live Dashboard',
            style={'textAlign': 'center', 'color': 'white', 'fontFamily': 'Arial Black'}),

    html.Div([
        html.Div([
            html.Label('Contract Type', style={'color': 'white'}),
            dcc.Dropdown(
                id='contract-filter',
                options=[{'label': 'All Contracts', 'value': 'All'}] +
                        [{'label': i, 'value': i} for i in df_original['Contract'].unique()],
                value='All'
            )
        ], style={'width': '30%', 'display': 'inline-block', 'padding': '10px'}),

        html.Div([
            html.Label('Payment Method', style={'color': 'white'}),
            dcc.Dropdown(
                id='payment-filter',
                options=[{'label': 'All Methods', 'value': 'All'}] +
                        [{'label': i, 'value': i} for i in df_original['PaymentMethod'].unique()],
                value='All'
            )
        ], style={'width': '30%', 'display': 'inline-block', 'padding': '10px'}),

        html.Div([
            html.Label('Internet Service', style={'color': 'white'}),
            dcc.Dropdown(
                id='internet-filter',
                options=[{'label': 'All Types', 'value': 'All'}] +
                        [{'label': i, 'value': i} for i in df_original['InternetService'].unique()],
                value='All'
            )
        ], style={'width': '30%', 'display': 'inline-block', 'padding': '10px'}),
    ]),

    dcc.Graph(id='dashboard-graph', style={'height': '2200px'})
])

@app.callback(
    Output('dashboard-graph', 'figure'),
    [Input('contract-filter', 'value'),
     Input('payment-filter', 'value'),
     Input('internet-filter', 'value')]
)
def update_dashboard(contract, payment, internet):
    filtered_df = df_original.copy()
    if contract!= 'All':
        filtered_df = filtered_df[filtered_df['Contract'] == contract]
    if payment!= 'All':
        filtered_df = filtered_df[filtered_df['PaymentMethod'] == payment]
    if internet!= 'All':
        filtered_df = filtered_df[filtered_df['InternetService'] == internet]

    if len(filtered_df) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data for selected filters.<br>Try different combination",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                           font=dict(size=20, color="white"))
        fig.update_layout(template='plotly_dark', paper_bgcolor='#0e1117',
                         plot_bgcolor='#1a1d29', height=2200)
        return fig

    total_customers = len(filtered_df)
    churn_count = filtered_df['Churn_Prediction'].sum()
    retained_count = total_customers - churn_count
    churn_rate = filtered_df['Churn_Prediction'].mean() * 100
    high_risk_count = (filtered_df['Risk_Level'] == 'High Risk').sum()
    total_revenue_at_risk = (filtered_df[filtered_df['Churn_Prediction']==1]['MonthlyCharges'].sum()) * 12

    fig = make_subplots(
        rows=4, cols=2,
        row_heights=[0.15, 0.3, 0.3, 0.25],
        specs=[[{"type": "indicator"}, {"type": "indicator"}],
               [{"type": "pie"}, {"type": "bar"}],
               [{"type": "scatter"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "heatmap"}]],
        subplot_titles=('', '',
                        'Customer Base Distribution', 'Churn Rate by Contract',
                        'Revenue Risk Analysis', 'Top Churn Drivers',
                        'Risk Level Distribution', 'Tenure vs Monthly Charges Heatmap'),
        vertical_spacing=0.08
    )

    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=churn_rate,
        gauge={'axis': {'range': [None, 100]},
               'bar': {'color': "#EF553B"},
               'steps': [{'range': [0, 30], 'color': "#00CC96"},
                        {'range': [30, 50], 'color': "#FFA15A"},
                        {'range': [50, 100], 'color': "#EF553B"}]},
        title={'text': f"<b>Churn Rate</b><br>{total_customers:,} Customers"},
        number={'suffix': "%"}),
        row=1, col=1
    )

    fig.add_trace(go.Indicator(
        mode="number",
        value=total_revenue_at_risk,
        number={'prefix': "$", 'valueformat': ",.0f"},
        title={'text': f"<b>Annual Revenue at Risk</b><br>{high_risk_count:,} High Risk"}),
        row=1, col=2
    )

    fig.add_trace(
        go.Pie(labels=['Retained', 'Churned'],
               values=[retained_count, churn_count],
               marker=dict(colors=['#00CC96', '#EF553B']),
               hole=0.6,
               textinfo='label+percent'),
        row=2, col=1
    )

    if len(filtered_df['Contract'].unique()) > 0:
        contract_churn = filtered_df.groupby('Contract')['Churn_Prediction'].mean() * 100
        fig.add_trace(
            go.Bar(x=contract_churn.index,
                   y=contract_churn.values,
                   marker_color='#EF553B',
                   text=[f'{x:.1f}%' for x in contract_churn.values],
                   textposition='outside'),
            row=2, col=2
        )

    if len(filtered_df) > 1:
        fig.add_trace(
            go.Scatter(x=filtered_df['MonthlyCharges'],
                       y=filtered_df['Churn_Probability'],
                       mode='markers',
                       marker=dict(color=filtered_df['Churn_Probability'],
                                 colorscale='RdYlGn_r',
                                 size=5,
                                 opacity=0.6),
                       name='Customers'),
            row=3, col=1
        )

    top_10 = feature_imp_df.nlargest(10, 'Importance').sort_values('Importance', ascending=True)
    fig.add_trace(
        go.Bar(x=top_10['Importance'],
               y=top_10['Feature'],
               orientation='h',
               marker=dict(color=top_10['Importance'], colorscale='Turbo')),
        row=3, col=2
    )

    risk_counts = filtered_df['Risk_Level'].value_counts()
    risk_colors = {'Low Risk': '#00CC96', 'Medium Risk': '#FFA15A', 'High Risk': '#EF553B'}
    fig.add_trace(
        go.Bar(x=risk_counts.index,
               y=risk_counts.values,
               marker_color=[risk_colors.get(x, '#636EFA') for x in risk_counts.index],
               text=[f'{v:,}<br>({v/len(filtered_df)*100:.1f}%)' for v in risk_counts.values],
               textposition='outside'),
        row=4, col=1
    )

    if 'tenure' in filtered_df.columns and len(filtered_df) > 0:
        tenure_bins = pd.cut(filtered_df['tenure'], bins=10)
        charge_bins = pd.cut(filtered_df['MonthlyCharges'], bins=10)
        heatmap_data = filtered_df.groupby([tenure_bins, charge_bins], observed=False)['Churn_Prediction'].mean().unstack()

        fig.add_trace(
            go.Heatmap(z=heatmap_data.values,
                      x=[f'${int(i.left)}-${int(i.right)}' for i in heatmap_data.columns],
                      y=[f'{int(i.left)}-{int(i.right)}mo' for i in heatmap_data.index],
                      colorscale='RdYlGn_r',
                      colorbar=dict(title="Churn Rate"),
                      hovertemplate='Tenure: %{y}<br>Charges: %{x}<br>Churn Rate: %{z:.1%}<extra></extra>'),
            row=4, col=2
        )

    fig.update_layout(
        height=2200,
        template='plotly_dark',
        paper_bgcolor='#0e1117',
        plot_bgcolor='#1a1d29',
        font=dict(color='white'),
        showlegend=False
    )

    fig.update_xaxes(title_text="Monthly Charges ($)", row=3, col=1)
    fig.update_yaxes(title_text="Churn Probability", row=3, col=1)
    fig.update_xaxes(title_text="Monthly Charges", row=4, col=2)
    fig.update_yaxes(title_text="Tenure", row=4, col=2)

    return fig

if __name__ == '__main__':
    print("\n🚀 DASH SERVER STARTING...")
    print("👉 Open browser: http://127.0.0.1:8050")
    app.run(debug=True)