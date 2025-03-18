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
        df = pd.read_excel("Travelicious Restaurant Data.xlsx")
        df.columns = df.columns.str.strip()
        
        # Print column names for debugging
        print("Available columns:", df.columns.tolist())
        
        # Ensure correct column names - check actual column names in the dataframe
        required_columns = ['MONTH', 'RESTAURANT', 'SALE', 'EXTRA INCOME', 
                           'TOTAL   SALE', 'GST', 'FOOD  EXPENSES', 'EXPENSES', 
                           'SALARY', 'P&L']
        
        # Map to actual column names if needed
        column_mapping = {}
        for col in df.columns:
            # Check if the column name contains any of the required names (case insensitive)
            for req_col in required_columns:
                if req_col.lower().replace(' ', '') in col.lower().replace(' ', ''):
                    column_mapping[req_col] = col
        
        # Rename columns if mapping exists
        if column_mapping:
            # Only rename if mappings were found
            if len(column_mapping) > 0:
                df = df.rename(columns=column_mapping)
                print("Columns renamed:", column_mapping)
        
        # Make sure MONTH is datetime and set to end of month
        df['MONTH'] = pd.to_datetime(df['MONTH'])
        df['MONTH'] = df['MONTH'] + pd.offsets.MonthEnd(0)  # Ensure all dates are end of month
        
        # Sort data chronologically
        df = df.sort_values('MONTH')
        
        # Define anomaly period
        anomaly_start = pd.to_datetime('2024-08-01')
        anomaly_end = pd.to_datetime('2024-12-31')
        
        # Get 2023 data for the anomaly months
        anomaly_months_2023 = df[(df['MONTH'].dt.year == 2023) & 
                                (df['MONTH'].dt.month.isin(range(8, 13)))].copy()
        
        # Get actual column names for calculations
        sale_col = 'SALE' if 'SALE' in df.columns else df.columns[df.columns.str.upper().str.contains('SALE')][0]
        extra_income_col = 'EXTRA INCOME' if 'EXTRA INCOME' in df.columns else df.columns[df.columns.str.upper().str.contains('EXTRA.*INCOME')][0]
        total_sale_col = 'TOTAL   SALE' if 'TOTAL   SALE' in df.columns else df.columns[df.columns.str.upper().str.contains('TOTAL.*SALE')][0]
        gst_col = 'GST' if 'GST' in df.columns else df.columns[df.columns.str.upper().str.contains('GST')][0]
        food_expenses_col = 'FOOD  EXPENSES' if 'FOOD  EXPENSES' in df.columns else df.columns[df.columns.str.upper().str.contains('FOOD.*EXPENSES')][0]
        expenses_col = 'EXPENSES' if 'EXPENSES' in df.columns else df.columns[df.columns.str.upper().str.contains('EXPENSES')][0]
        salary_col = 'SALARY' if 'SALARY' in df.columns else df.columns[df.columns.str.upper().str.contains('SALARY')][0]
        pnl_col = 'P&L' if 'P&L' in df.columns else df.columns[df.columns.str.upper().str.contains('P.*L')][0]
        
        print(f"Using columns: {sale_col}, {extra_income_col}, {total_sale_col}, {gst_col}, {food_expenses_col}, {expenses_col}, {salary_col}, {pnl_col}")
        
        # Create baseline data with exactly 70% of 2023 values
        baseline_2024 = anomaly_months_2023.copy()
        baseline_2024['MONTH'] = baseline_2024['MONTH'] + pd.DateOffset(years=1)
        # Calculate exactly 70% for each metric while preserving original patterns
        baseline_2024[sale_col] = baseline_2024[sale_col] * 0.7
        baseline_2024[extra_income_col] = baseline_2024[extra_income_col] * 0.7
        baseline_2024[total_sale_col] = baseline_2024[total_sale_col] * 0.7
        baseline_2024[gst_col] = baseline_2024[gst_col] * 0.7
        baseline_2024[food_expenses_col] = baseline_2024[food_expenses_col] * 0.7
        baseline_2024[expenses_col] = baseline_2024[expenses_col] * 0.7
        baseline_2024[salary_col] = baseline_2024[salary_col]  # Keep original salary for 2024
        baseline_2024[pnl_col] = baseline_2024[total_sale_col] - (baseline_2024[food_expenses_col] + baseline_2024[expenses_col] + baseline_2024[salary_col])
        
        # Split data into normal and anomaly periods
        normal_data = df[~((df['MONTH'] >= anomaly_start) & (df['MONTH'] <= anomaly_end))]
        
        # Combine all data
        full_data = pd.concat([normal_data, baseline_2024]).sort_values('MONTH')
        
        # Prepare forecasting data
        metrics = [sale_col, extra_income_col, food_expenses_col, expenses_col]
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
                    start='2025-03-31',  # End of March 2025
                    periods=10,          # March through December 2025
                    freq='M'             # Month end frequency
                )
            })
            
            # Generate forecast
            forecast = model.predict(future)
            
            # Apply constraints
            for idx, row in forecast.iterrows():
                month = row['ds'].month
                
                # Get corresponding month from historical data for better reference
                month_in_2023 = None
                month_2024 = full_data[
                    (full_data['MONTH'].dt.month == month) & 
                    (full_data['MONTH'].dt.year == 2024)
                ][metric]
                
                month_2023 = df[
                    (df['MONTH'].dt.month == month) & 
                    (df['MONTH'].dt.year == 2023)
                ][metric]
                
                if not month_2023.empty:
                    month_in_2023 = month_2023.iloc[0]
                
                # Find the same month from previous years for reference
                if not month_2024.empty:
                    base_value = month_2024.iloc[0]
                elif month_in_2023 is not None:
                    base_value = month_in_2023
                else:
                    base_value = historical_stats['mean']
                
                # Ensure the forecast stays within 10-15% of historical data
                max_growth = 0.10  # Maximum 10% growth
                max_decline = 0.10  # Maximum 10% decline
                
                # If month has historical precedent, use that as reference point
                if month_in_2023 is not None:
                    max_allowed = month_in_2023 * 1.10  # Allow at most 10% increase from 2023
                    min_allowed = month_in_2023 * 0.90  # Allow at most 10% decrease from 2023
                else:
                    # Otherwise use the current calculated base value
                    max_allowed = base_value * (1 + max_growth)
                    min_allowed = base_value * (1 - max_decline)
                
                # Calculate confidence interval limits - strict 5% variance
                min_allowed_lower = min_allowed * 0.95  # 5% below the min allowed
                max_allowed_upper = max_allowed * 1.05  # 5% above the max allowed
                
                # Clip the forecasted values with different ranges for bounds
                forecast.loc[idx, 'yhat'] = np.clip(forecast.loc[idx, 'yhat'], min_allowed, max_allowed)
                forecast.loc[idx, 'yhat_lower'] = np.clip(forecast.loc[idx, 'yhat_lower'], min_allowed_lower, forecast.loc[idx, 'yhat'])
                forecast.loc[idx, 'yhat_upper'] = np.clip(forecast.loc[idx, 'yhat_upper'], forecast.loc[idx, 'yhat'], max_allowed_upper)
            
            forecasts[metric] = forecast
        
        # Calculate TOTAL SALE forecasts
        total_sale_dates = forecasts[sale_col]['ds']
        total_sale_forecast = pd.DataFrame({
            'ds': total_sale_dates,
            'yhat': forecasts[sale_col]['yhat'] + forecasts[extra_income_col]['yhat'],
            'yhat_lower': forecasts[sale_col]['yhat_lower'] + forecasts[extra_income_col]['yhat_lower'],
            'yhat_upper': forecasts[sale_col]['yhat_upper'] + forecasts[extra_income_col]['yhat_upper']
        })
        forecasts[total_sale_col] = total_sale_forecast
        
        # Calculate SALARY forecasts based on historical patterns
        salary_historical = df[df['MONTH'] < anomaly_start][salary_col].mean()
        salary_dates = forecasts[sale_col]['ds']
        salary_forecast = pd.DataFrame({
            'ds': salary_dates,
            'yhat': forecasts[sale_col]['yhat'] * 0.3,  # 30% of sales as per constraints
            'yhat_lower': forecasts[sale_col]['yhat_lower'] * 0.3,
            'yhat_upper': forecasts[sale_col]['yhat_upper'] * 0.3
        })
        forecasts[salary_col] = salary_forecast
        
        # Calculate GST forecasts (5% of TOTAL SALE as per constraints)
        gst_dates = forecasts[total_sale_col]['ds']
        gst_forecast = pd.DataFrame({
            'ds': gst_dates,
            'yhat': forecasts[total_sale_col]['yhat'] * 0.05,  # 5% of total sale
            'yhat_lower': forecasts[total_sale_col]['yhat_lower'] * 0.05,
            'yhat_upper': forecasts[total_sale_col]['yhat_upper'] * 0.05
        })
        forecasts[gst_col] = gst_forecast
        
        # Apply additional constraints to match requirements
        # Food expenses: 30% of total sales
        forecasts[food_expenses_col]['yhat'] = forecasts[total_sale_col]['yhat'] * 0.3
        forecasts[food_expenses_col]['yhat_lower'] = forecasts[total_sale_col]['yhat_lower'] * 0.3
        forecasts[food_expenses_col]['yhat_upper'] = forecasts[total_sale_col]['yhat_upper'] * 0.3
        
        # Expenses: 20% of total sales
        forecasts[expenses_col]['yhat'] = forecasts[total_sale_col]['yhat'] * 0.2
        forecasts[expenses_col]['yhat_lower'] = forecasts[total_sale_col]['yhat_lower'] * 0.2
        forecasts[expenses_col]['yhat_upper'] = forecasts[total_sale_col]['yhat_upper'] * 0.2
        
        # Calculate P&L forecast
        forecast_df = pd.DataFrame({
            'ds': forecasts[sale_col]['ds'],
            'sale': forecasts[sale_col]['yhat'],
            'extra_income': forecasts[extra_income_col]['yhat'],
            'total_sale': forecasts[total_sale_col]['yhat'],
            'gst': forecasts[gst_col]['yhat'],
            'food_expenses': forecasts[food_expenses_col]['yhat'],
            'expenses': forecasts[expenses_col]['yhat'],
            'salary': forecasts[salary_col]['yhat']
        })
        
        forecast_df['P&L'] = forecast_df['total_sale'] - (forecast_df['food_expenses'] + forecast_df['expenses'] + forecast_df['salary'])
        forecast_df['P&L_lower'] = (forecasts[total_sale_col]['yhat_lower'] - 
                                  (forecasts[food_expenses_col]['yhat_upper'] + 
                                   forecasts[expenses_col]['yhat_upper'] + 
                                   forecasts[salary_col]['yhat_upper']))
        forecast_df['P&L_upper'] = (forecasts[total_sale_col]['yhat_upper'] - 
                                  (forecasts[food_expenses_col]['yhat_lower'] + 
                                   forecasts[expenses_col]['yhat_lower'] + 
                                   forecasts[salary_col]['yhat_lower']))
        
        # Create visualization with connected segments
        fig = go.Figure()
        
        # Get the last historical data point and first forecast point
        last_historical_date = df[df['MONTH'] < pd.to_datetime('2025-03-01')].iloc[-1]['MONTH']
        last_historical_value = df[df['MONTH'] < pd.to_datetime('2025-03-01')].iloc[-1][pnl_col]
        first_forecast_date = forecast_df['ds'].iloc[0]
        first_forecast_value = forecast_df['P&L'].iloc[0]
        
        # Calculate intermediate point for smoother transition
        intermediate_date = last_historical_date + (first_forecast_date - last_historical_date) * 0.5
        intermediate_value = last_historical_value + (first_forecast_value - last_historical_value) * 0.5
        
        # Create connection points for smoother transition
        connection_points = pd.DataFrame({
            'MONTH': [last_historical_date, intermediate_date, first_forecast_date],
            pnl_col: [last_historical_value, intermediate_value, first_forecast_value]
        })
        
        # Create continuous historical line (pre-landslide)
        pre_landslide = df[df['MONTH'] < anomaly_start]
        fig.add_trace(go.Scatter(
            x=pre_landslide['MONTH'],
            y=pre_landslide[pnl_col],
            name='Historical P&L',
            line=dict(color='#4CAF50', width=2)
        ))
        
        # Add landslide period with dotted line (70% baseline)
        fig.add_trace(go.Scatter(
            x=baseline_2024['MONTH'],
            y=baseline_2024[pnl_col],
            name='Landslide Period (70% Baseline)',
            line=dict(color='#4CAF50', width=2, dash='dot')
        ))
        
        # Add post-landslide historical data
        post_landslide = df[(df['MONTH'] > anomaly_end) & (df['MONTH'] < pd.to_datetime('2025-03-01'))]
        if not post_landslide.empty:
            fig.add_trace(go.Scatter(
                x=post_landslide['MONTH'],
                y=post_landslide[pnl_col],
                name='Post-Landslide P&L',
                line=dict(color='#4CAF50', width=2)
            ))
        
        # Add connection to forecast (smooth transition)
        fig.add_trace(go.Scatter(
            x=connection_points['MONTH'],
            y=connection_points[pnl_col],
            name='Connection',
            line=dict(color='#4CAF50', width=2),
            showlegend=False
        ))
        
        # Add forecast
        fig.add_trace(go.Scatter(
            x=forecast_df['ds'],
            y=forecast_df['P&L'],
            name='2025 Forecast',
            line=dict(color='#4CAF50', width=2, dash='dash')
        ))
        
        # Add confidence interval with 5% variance
        fig.add_trace(go.Scatter(
            x=forecast_df['ds'].tolist() + forecast_df['ds'].tolist()[::-1],
            y=forecast_df['P&L_upper'].tolist() + forecast_df['P&L_lower'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(76, 175, 80, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='5% Confidence Interval'
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
                text='Travelicious Restaurant - P&L Forecast for 2025',
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
            **Forecast Methodology for Travelicious Restaurant:**
            
            1. Historical Data Analysis:
               - Using complete data from 2023 through Feb 2025
               - Landslide-affected months (Aug-Dec 2024):
                 * Exactly 70% of corresponding 2023 values
            
            2. Forecasting Approach:
               - Conservative growth limits (±10%)
               - Monthly seasonality patterns
               - Forecasting March-December 2025
               
            3. Optimization Constraints Applied:
               - Salary: 30% of Sales
               - Expenses: 20% of Total Sales
               - GST: 5% of Total Sales
               - Food Expenses: 30% of Total Sales
        """)
        
        # Display the plot
        st.plotly_chart(fig, use_container_width=True)
        
        # Prepare and display forecast tables
        st.markdown("### Forecasted Values for 2025")
        
        # Create normal forecast table first
        normal_forecast = pd.DataFrame({
            'Month': forecast_df['ds'].dt.strftime('%B'),
            'Sale': forecast_df['sale'].round(2),
            'Extra Income': forecast_df['extra_income'].round(2),
            'Total Sale': forecast_df['total_sale'].round(2),
            'GST': forecast_df['gst'].round(2),
            'Food Expenses': forecast_df['food_expenses'].round(2),
            'Expenses': forecast_df['expenses'].round(2),
            'Salary': forecast_df['salary'].round(2),
            'P&L': forecast_df['P&L'].round(2)
        })
        normal_forecast.set_index('Month', inplace=True)
        
        # Display normal forecast table
        st.markdown("#### Base Forecast")
        st.dataframe(normal_forecast)
        
        # Create lower bound forecast table
        lower_bound = pd.DataFrame({
            'Month': forecast_df['ds'].dt.strftime('%B'),
            'Sale': forecasts[sale_col]['yhat_lower'].round(2),
            'Extra Income': forecasts[extra_income_col]['yhat_lower'].round(2),
            'Total Sale': forecasts[total_sale_col]['yhat_lower'].round(2),
            'GST': forecasts[gst_col]['yhat_lower'].round(2),
            'Food Expenses': forecasts[food_expenses_col]['yhat_lower'].round(2),
            'Expenses': forecasts[expenses_col]['yhat_lower'].round(2),
            'Salary': forecasts[salary_col]['yhat_lower'].round(2),
            'P&L': forecast_df['P&L_lower'].round(2)
        })
        lower_bound.set_index('Month', inplace=True)
        
        # Display lower bound table
        st.markdown("#### Optimized Lower Bound (−5%)")
        st.dataframe(lower_bound)
        
        # Create upper bound forecast table
        upper_bound = pd.DataFrame({
            'Month': forecast_df['ds'].dt.strftime('%B'),
            'Sale': forecasts[sale_col]['yhat_upper'].round(2),
            'Extra Income': forecasts[extra_income_col]['yhat_upper'].round(2),
            'Total Sale': forecasts[total_sale_col]['yhat_upper'].round(2),
            'GST': forecasts[gst_col]['yhat_upper'].round(2),
            'Food Expenses': forecasts[food_expenses_col]['yhat_upper'].round(2),
            'Expenses': forecasts[expenses_col]['yhat_upper'].round(2),
            'Salary': forecasts[salary_col]['yhat_upper'].round(2),
            'P&L': forecast_df['P&L_upper'].round(2)
        })
        upper_bound.set_index('Month', inplace=True)
        
        # Display upper bound table
        st.markdown("#### Optimized Upper Bound (+5%)")
        st.dataframe(upper_bound)
        
        # Create combined optimized forecast table
        optimized_forecast = pd.DataFrame({
            'Month': forecast_df['ds'].dt.strftime('%B'),
            'Sale (Lower)': forecasts[sale_col]['yhat_lower'].round(2),
            'Sale (Base)': forecast_df['sale'].round(2),
            'Sale (Upper)': forecasts[sale_col]['yhat_upper'].round(2),
            'Extra Income (Lower)': forecasts[extra_income_col]['yhat_lower'].round(2),
            'Extra Income (Base)': forecast_df['extra_income'].round(2),
            'Extra Income (Upper)': forecasts[extra_income_col]['yhat_upper'].round(2),
            'Total Sale (Lower)': forecasts[total_sale_col]['yhat_lower'].round(2),
            'Total Sale (Base)': forecast_df['total_sale'].round(2),
            'Total Sale (Upper)': forecasts[total_sale_col]['yhat_upper'].round(2),
            'GST (Lower)': forecasts[gst_col]['yhat_lower'].round(2),
            'GST (Base)': forecast_df['gst'].round(2),
            'GST (Upper)': forecasts[gst_col]['yhat_upper'].round(2),
            'Food Expenses (Lower)': forecasts[food_expenses_col]['yhat_lower'].round(2),
            'Food Expenses (Base)': forecast_df['food_expenses'].round(2),
            'Food Expenses (Upper)': forecasts[food_expenses_col]['yhat_upper'].round(2),
            'Expenses (Lower)': forecasts[expenses_col]['yhat_lower'].round(2),
            'Expenses (Base)': forecast_df['expenses'].round(2),
            'Expenses (Upper)': forecasts[expenses_col]['yhat_upper'].round(2),
            'Salary (Lower)': forecasts[salary_col]['yhat_lower'].round(2),
            'Salary (Base)': forecast_df['salary'].round(2),
            'Salary (Upper)': forecasts[salary_col]['yhat_upper'].round(2),
            'P&L (Lower)': forecast_df['P&L_lower'].round(2),
            'P&L (Base)': forecast_df['P&L'].round(2),
            'P&L (Upper)': forecast_df['P&L_upper'].round(2)
        })
        optimized_forecast.set_index('Month', inplace=True)
        
        # Display combined optimized view
        st.markdown("#### Combined Optimized View")
        st.dataframe(optimized_forecast)
        
        # Display optimized ratio table
        st.markdown("### Optimization Ratios Applied")
        ratio_table = pd.DataFrame({
            'Metric': ['Salary', 'Expenses', 'GST', 'Food Expenses'],
            'Percentage of Sales/Total Sales': ['30% of Sale', '20% of Total Sale', '5% of Total Sale', '30% of Total Sale']
        })
        st.dataframe(ratio_table)
        
    except Exception as e:
        st.error("An error occurred in the dashboard")
        st.error(f"Error details: {str(e)}")
        st.error("Traceback:")
        st.code(traceback.format_exc())