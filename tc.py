import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from datetime import datetime
import numpy as np
import traceback

def show_dashboard():
    try:
        # Load and prepare data
        df = pd.read_excel("Travellers Cavern Data.xlsx")
        df.columns = df.columns.str.strip()
        df['MONTH'] = pd.to_datetime(df['MONTH'])
        
        # Sort data chronologically
        df = df.sort_values('MONTH')
        
        # Define anomaly period
        anomaly_start = pd.to_datetime('2024-08-01')
        anomaly_end = pd.to_datetime('2024-12-31')
        
        # Get 2023 data for the anomaly months
        anomaly_months_2023 = df[(df['MONTH'].dt.year == 2023) & 
                                (df['MONTH'].dt.month.isin(range(8, 13)))].copy()
        
        # Create baseline data with exactly 70% of 2023 values
        baseline_2024 = anomaly_months_2023.copy()
        baseline_2024['MONTH'] = baseline_2024['MONTH'] + pd.DateOffset(years=1)
        baseline_2024['SALES'] = baseline_2024['SALES'] * 0.7
        baseline_2024['EXPENSES'] = baseline_2024['EXPENSES'] * 0.7
        baseline_2024['SALARY'] = baseline_2024['SALARY']  # Keep original salary for 2024
        baseline_2024['P&L'] = baseline_2024['SALES'] - (baseline_2024['EXPENSES'] + baseline_2024['SALARY'])
        
        # Split data into normal and anomaly periods
        normal_data = df[~((df['MONTH'] >= anomaly_start) & (df['MONTH'] <= anomaly_end))]
        
        # Combine all data
        full_data = pd.concat([normal_data, baseline_2024]).sort_values('MONTH')
        
        # Prepare forecasting data
        metrics = ['SALES', 'EXPENSES']  # Remove SALARY as it will be fixed
        forecasts = {}
        
        for metric in metrics:
            # Calculate historical stats
            historical_stats = df[df['MONTH'] < anomaly_start][metric].describe()
            
            # Prepare Prophet data
            prophet_df = pd.DataFrame({
                'ds': full_data['MONTH'],
                'y': full_data[metric].astype(float)
            })
            
            # Configure Prophet model
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                changepoint_prior_scale=0.001,  # Very conservative
                seasonality_prior_scale=10.0,
                changepoint_range=0.8,
                mcmc_samples=0
            )
            
            # Add monthly seasonality
            model.add_seasonality(
                name='monthly',
                period=30.5,
                fourier_order=3
            )
            
            # Fit model
            model.fit(prophet_df)
            
            # Create future dataframe starting from March 2025
            future = pd.DataFrame({
                'ds': pd.date_range(
                    start='2025-03-01',
                    end='2025-12-31',
                    freq='M'
                )
            })
            
            # Generate forecast
            forecast = model.predict(future)
            
            # Apply conservative constraints
            for idx, row in forecast.iterrows():
                month = row['ds'].month
                
                # Get corresponding month value from 2024 (pre-landslide if available)
                month_2024 = full_data[
                    (full_data['MONTH'].dt.month == month) & 
                    (full_data['MONTH'].dt.year == 2024)
                ][metric]
                
                if not month_2024.empty:
                    base_value = month_2024.iloc[0]
                else:
                    # Use 2023 value if available
                    month_2023 = df[
                        (df['MONTH'].dt.month == month) & 
                        (df['MONTH'].dt.year == 2023)
                    ][metric]
                    base_value = month_2023.iloc[0] if not month_2023.empty else historical_stats['mean']
                
                # Set growth limits
                max_growth = 0.15  # Maximum 15% growth
                max_decline = 0.15  # Maximum 15% decline
                
                # Calculate allowed range for base forecast
                min_allowed = base_value * (1 - max_decline)
                max_allowed = base_value * (1 + max_growth)
                
                # Calculate wider ranges for confidence intervals
                min_allowed_lower = base_value * (1 - max_decline * 1.5)  # 22.5% decline for lower bound
                max_allowed_upper = base_value * (1 + max_growth * 1.5)   # 22.5% growth for upper bound
                
                # Clip the forecasted values with different ranges for bounds
                forecast.loc[idx, 'yhat'] = np.clip(forecast.loc[idx, 'yhat'], min_allowed, max_allowed)
                forecast.loc[idx, 'yhat_lower'] = np.clip(forecast.loc[idx, 'yhat_lower'], min_allowed_lower, forecast.loc[idx, 'yhat'])
                forecast.loc[idx, 'yhat_upper'] = np.clip(forecast.loc[idx, 'yhat_upper'], forecast.loc[idx, 'yhat'], max_allowed_upper)
            
            forecasts[metric] = forecast
        
        # Create fixed salary forecast
        salary_dates = pd.date_range(start='2025-03-01', end='2025-12-31', freq='M')
        forecasts['SALARY'] = pd.DataFrame({
            'ds': salary_dates,
            'yhat': 22000,
            'yhat_lower': 22000,
            'yhat_upper': 22000
        })
        
        # Calculate P&L forecast
        forecast_df = pd.DataFrame({
            'ds': forecasts['SALES']['ds'],
            'sales': forecasts['SALES']['yhat'],
            'expenses': forecasts['EXPENSES']['yhat'],
            'salary': forecasts['SALARY']['yhat']
        })
        
        forecast_df['P&L'] = forecast_df['sales'] - (forecast_df['expenses'] + forecast_df['salary'])
        forecast_df['P&L_lower'] = (forecasts['SALES']['yhat_lower'] - 
                                  (forecasts['EXPENSES']['yhat_upper'] + forecasts['SALARY']['yhat']))
        forecast_df['P&L_upper'] = (forecasts['SALES']['yhat_upper'] - 
                                  (forecasts['EXPENSES']['yhat_lower'] + forecasts['SALARY']['yhat']))
        
        # Create visualization with connected segments
        fig = go.Figure()
        
        # Add pre-landslide historical data
        pre_landslide = df[df['MONTH'] < anomaly_start]
        fig.add_trace(go.Scatter(
            x=pre_landslide['MONTH'],
            y=pre_landslide['P&L'],
            name='Historical P&L',
            line=dict(color='#4CAF50', width=2)
        ))
        
        # Add landslide period with dotted line (70% baseline)
        fig.add_trace(go.Scatter(
            x=baseline_2024['MONTH'],
            y=baseline_2024['P&L'],
            name='Landslide Period (70% Baseline)',
            line=dict(color='#4CAF50', width=2, dash='dot')
        ))
        
        # Add post-landslide data
        post_landslide = df[df['MONTH'] > anomaly_end]
        fig.add_trace(go.Scatter(
            x=post_landslide['MONTH'],
            y=post_landslide['P&L'],
            name='Post-Landslide P&L',
            line=dict(color='#4CAF50', width=2)
        ))
        
        # Add forecast
        fig.add_trace(go.Scatter(
            x=forecast_df['ds'],
            y=forecast_df['P&L'],
            name='2025 Forecast',
            line=dict(color='#4CAF50', width=2, dash='dash')
        ))
        
        # Add confidence interval
        fig.add_trace(go.Scatter(
            x=forecast_df['ds'].tolist() + forecast_df['ds'].tolist()[::-1],
            y=forecast_df['P&L_upper'].tolist() + forecast_df['P&L_lower'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(76, 175, 80, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% Confidence Interval'
        ))
        
        # Update layout
        fig.update_layout(
            template='plotly_dark',
            plot_bgcolor='#1E1E1E',
            paper_bgcolor='#1E1E1E',
            margin=dict(t=40, l=40, r=40, b=40),
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor='rgba(30,30,30,0.8)'
            ),
            title=dict(
                text='P&L Forecast for 2025',
                x=0.5,
                xanchor='center',
                font=dict(color='#FFFFFF')
            ),
            xaxis=dict(
                title='Date',
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)',
                tickfont=dict(color='#FFFFFF'),
                title_font=dict(color='#FFFFFF')
            ),
            yaxis=dict(
                title='P&L (₹)',
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)',
                rangemode='tozero',
                tickfont=dict(color='#FFFFFF'),
                title_font=dict(color='#FFFFFF')
            )
        )
        
        # Display methodology
        st.info("""
            **Forecast Methodology:**
            1. Historical Data Analysis:
               - Using complete data from 2023 through Feb 2025
               - Landslide-affected months (Aug-Dec 2024):
                 * Exactly 70% of corresponding 2023 values
               - Fixed salary of ₹22,000 for forecasted months
            
            2. Forecasting Approach:
               - Conservative growth limits (±15%)
               - Monthly seasonality patterns
               - Using corresponding month values from previous year
               - Forecasting March-December 2025
        """)
        
        # Display the plot
        st.plotly_chart(fig, use_container_width=True)
        
        # Prepare and display consolidated forecast table
        st.markdown("### Forecasted Values for 2025")
        
        # Create consolidated forecast table
        consolidated_table = pd.DataFrame({
            'Month': forecast_df['ds'].dt.strftime('%B'),
            'Sales (Lower)': forecasts['SALES']['yhat_lower'].round(2),
            'Sales (Base)': forecast_df['sales'].round(2),
            'Sales (Upper)': forecasts['SALES']['yhat_upper'].round(2),
            'Expenses (Lower)': forecasts['EXPENSES']['yhat_lower'].round(2),
            'Expenses (Base)': forecast_df['expenses'].round(2),
            'Expenses (Upper)': forecasts['EXPENSES']['yhat_upper'].round(2),
            'Salary': forecast_df['salary'].round(2),
            'P&L (Lower)': forecast_df['P&L_lower'].round(2),
            'P&L (Base)': forecast_df['P&L'].round(2),
            'P&L (Upper)': forecast_df['P&L_upper'].round(2)
        })
        consolidated_table.set_index('Month', inplace=True)
        
        # Display consolidated table
        st.dataframe(consolidated_table)
        
    except Exception as e:
        st.error("An error occurred in the dashboard")
        st.error(f"Error details: {str(e)}")
        st.error("Traceback:")
        st.code(traceback.format_exc())
