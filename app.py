import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Load the saved model and preprocessing objects
model = joblib.load('project_success_model.pkl')
scaler = joblib.load('scaler.pkl')
le_dict = joblib.load('label_encoders.pkl')

def main():
    st.title('Software Project Success Predictor')
    
    st.write('Enter project details to predict success probability')
    
    # Create input form
    with st.form("project_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            project_type = st.selectbox('Project Type', 
                ['Web Application', 'Mobile App', 'Desktop Application', 
                 'API Development', 'Enterprise System', 'IoT Solution',
                 'AI/ML System', 'Database Implementation', 'Cloud Migration'])
            
            industry = st.selectbox('Industry',
                ['Technology', 'Finance', 'Healthcare', 'Retail', 
                 'Telecom', 'Entertainment', 'Education', 'Manufacturing',
                 'Government', 'Energy'])
            
            team_size = st.selectbox('Team Size',
                ['Small (1-5)', 'Medium (6-15)', 'Large (16-30)', 'Very Large (31+)'])
            
            methodology = st.selectbox('Development Methodology',
                ['Agile', 'Waterfall', 'Scrum', 'Kanban', 'DevOps', 'Hybrid'])
            
            risk_level = st.selectbox('Risk Level', ['Low', 'Medium', 'High'])
            client_type = st.selectbox('Client Type', ['Internal', 'External'])
            
        with col2:
            pm_exp = st.number_input('Project Manager Experience (years)', 
                min_value=0, max_value=30, value=5)
            
            team_exp = st.number_input('Team Experience (years)', 
                min_value=0.0, max_value=10.0, value=4.0, step=0.1)
            
            initial_cost = st.number_input('Initial Cost Estimate ($)', 
                min_value=10000, max_value=1000000, value=200000)
            
            initial_time = st.number_input('Initial Time Estimate (months)', 
                min_value=1, max_value=36, value=6)
            
            requirements_stability = st.slider('Requirements Stability (%)', 
                0, 100, 70)
            
            stakeholders = st.number_input('Number of Stakeholders', 
                min_value=1, max_value=20, value=8)
            
            tech_complexity = st.slider('Technology Complexity (1-10)', 
                1, 10, 5)
            
            integrations = st.number_input('Number of Integrations', 
                min_value=0, max_value=15, value=4)

        submitted = st.form_submit_button("Predict Success Probability")

        if submitted:
            # Create DataFrame with input values
            project_data = pd.DataFrame({
                'project_type': [project_type],
                'industry': [industry],
                'team_size': [team_size],
                'development_methodology': [methodology],
                'risk_level': [risk_level],
                'client_type': [client_type],
                'days_since_start': [(datetime.now() - datetime(2020, 1, 1)).days],
                'project_manager_experience': [pm_exp],
                'team_experience': [team_exp],
                'initial_cost_estimate': [initial_cost],
                'actual_cost': [initial_cost * 1.1],  # Estimated
                'cost_overrun_percentage': [10],  # Estimated
                'initial_time_estimate': [initial_time],
                'actual_time': [initial_time * 1.2],  # Estimated 20% overrun
                'time_overrun_percentage': [20],  # Estimated
                'requirements_stability': [requirements_stability],
                'num_stakeholders': [stakeholders],
                'stakeholder_satisfaction': [7.5],  # Default value
                'technology_complexity': [tech_complexity],
                'num_integrations': [integrations],
                'quality_defects': [3],  # Default values
                'code_base_size': [50],
                'velocity': [2.5],
                'total_defects': [150]
            })

            # Preprocess the new data
            for col in project_data.columns:
                if col in le_dict:
                    project_data[col] = le_dict[col].transform(project_data[col])

            # Scale the features
            project_data_scaled = scaler.transform(project_data)

            # Make prediction
            prediction = model.predict(project_data_scaled)
            probability = model.predict_proba(project_data_scaled)

            # Display results
            st.header('Prediction Results')
            success_prob = probability[0][1] * 100
            
            if prediction[0]:
                st.success(f'Project is predicted to SUCCEED with {success_prob:.2f}% probability!')
            else:
                st.error(f'Project is predicted to FAIL with {(100-success_prob):.2f}% probability of failure!')

            # Display probability gauge
            st.write('Success Probability:')
            st.progress(success_prob/100)
            
            # Display detailed analysis automatically
            st.header('Detailed Project Analysis')
            
            # Financial Analysis
            st.subheader('Financial Analysis')
            actual_cost = initial_cost * 1.1
            estimated_revenue = initial_cost * 1.5
            profit_loss = estimated_revenue - actual_cost
            profit_margin = ((estimated_revenue - actual_cost) / actual_cost) * 100
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Initial Cost", f"${initial_cost:,.2f}")
                st.metric("Estimated Final Cost", f"${actual_cost:,.2f}", 
                        delta=f"${actual_cost - initial_cost:,.2f}")
            with col2:
                st.metric("Projected Revenue", f"${estimated_revenue:,.2f}")
                st.metric("Expected Profit/Loss", 
                        f"${profit_loss:,.2f}",
                        delta=f"{profit_margin:.1f}%")

            # Schedule Analysis
            st.subheader('Schedule Analysis')
            actual_time = initial_time * 1.2
            time_overrun = actual_time - initial_time
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Initial Timeline", f"{initial_time} months")
                st.metric("Estimated Completion Time", f"{actual_time:.1f} months",
                        delta=f"{time_overrun:.1f} months")
            with col2:
                st.metric("Schedule Overrun", f"{(time_overrun/initial_time)*100:.1f}%")
                st.metric("Development Velocity", "2.5 story points/sprint")

            # Risk Analysis
            st.subheader('Risk Analysis')
            
            # Calculate risk factors
            technical_risk = tech_complexity / 10 * 100
            team_risk = (10 - team_exp) / 10 * 100
            schedule_risk = (time_overrun / initial_time) * 100
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Technical Risk", f"{technical_risk:.1f}%")
                st.metric("Team Risk", f"{team_risk:.1f}%")
            with col2:
                st.metric("Schedule Risk", f"{schedule_risk:.1f}%")
                st.metric("Overall Risk Score", f"{(technical_risk + team_risk + schedule_risk)/3:.1f}%")

            # Recommendations for 99% Success
            st.subheader('Recommendations for Project Success')
            recommendations = []
            
            if requirements_stability < 80:
                recommendations.append("- Increase requirements stability to at least 80%")
            if team_exp < 5:
                recommendations.append("- Increase team experience or provide additional training")
            if tech_complexity > 7:
                recommendations.append("- Consider reducing technical complexity or adding more experienced team members")
            if stakeholders > 10:
                recommendations.append("- Streamline stakeholder management and communication channels")
            
            st.markdown("\n".join(recommendations))
            
            # Success Factors
            st.subheader('Key Success Factors')
            success_factors = {
                'Requirements Stability': requirements_stability,
                'Team Experience': team_exp * 10,
                'PM Experience': pm_exp * 3.33,
                'Technical Readiness': (10 - tech_complexity) * 10,
                'Stakeholder Alignment': (20 - stakeholders) * 5
            }
            
            fig, ax = plt.subplots(figsize=(10, 5))
            plt.bar(success_factors.keys(), success_factors.values())
            plt.title('Project Success Factors (Higher is Better)')
            plt.xticks(rotation=45)
            plt.ylim(0, 100)
            st.pyplot(fig)

if __name__ == '__main__':
    main()
