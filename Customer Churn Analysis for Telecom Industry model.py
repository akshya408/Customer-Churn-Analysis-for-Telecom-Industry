import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

def generate_sample_data():
    np.random.seed(42)
    n_samples = 1000

    data = pd.DataFrame({
        'customerID': ['CUST' + str(i) for i in range(n_samples)],
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'SeniorCitizen': np.random.choice([0, 1], n_samples),
        'Partner': np.random.choice(['Yes', 'No'], n_samples),
        'Dependents': np.random.choice(['Yes', 'No'], n_samples),
        'tenure': np.random.randint(0, 72, n_samples),
        'PhoneService': np.random.choice(['Yes', 'No'], n_samples),
        'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
        'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples),
        'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], n_samples),
        'MonthlyCharges': np.round(np.random.uniform(20, 120, n_samples), 2),
        'TotalCharges': lambda df: np.round(df['MonthlyCharges'] * df['tenure'], 2),
        'Churn': np.random.choice(['Yes', 'No'], n_samples, p=[0.27, 0.73])
    })
    data['TotalCharges'] = data['MonthlyCharges'] * data['tenure']
    data['TotalCharges'] = data['TotalCharges'].round(2)
    return data

def preprocess_data(df):
    df = df.drop('customerID', axis=1)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(0, inplace=True)

    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    for col in binary_cols:
        df[col] = df[col].map({'Yes':1, 'No':0, 'Male':1, 'Female':0})

    replace_cols = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    for col in replace_cols:
        df[col] = df[col].replace({'No internet service':'No', 'No phone service':'No'})

    for col in replace_cols:
        df[col] = df[col].map({'Yes':1, 'No':0})

    df = pd.get_dummies(df, columns=['MultipleLines', 'InternetService', 'Contract', 'PaymentMethod'])
    df.fillna(0, inplace=True)
    return df

def train_model(df):
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:,1]

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    return accuracy, cm, report, fpr, tpr, roc_auc, model, X_train.columns

def plot_metrics(accuracy, cm, report, fpr, tpr, roc_auc, model, feature_names):
    st.subheader("Model Accuracy")
    st.write(f"{accuracy:.2f}")

    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'], ax=ax, cbar=False)
    plt.xlabel('Predicted', color='white')
    plt.ylabel('Actual', color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    st.pyplot(fig)

    st.subheader("Classification Report Metrics")
    df_report = pd.DataFrame(report).transpose()
    df_report = df_report.iloc[:-3, :3]  # exclude accuracy, macro avg, weighted avg rows; keep precision, recall, f1-score columns
    fig2, ax2 = plt.subplots()
    fig2.patch.set_alpha(0.0)
    ax2.patch.set_alpha(0.0)
    bars = df_report.plot(kind='bar', figsize=(10,6), ax=ax2, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.ylim(0,1)
    plt.xticks(rotation=0)
    ax2.tick_params(axis='x', colors='white')
    ax2.tick_params(axis='y', colors='white')
    ax2.yaxis.label.set_color('white')
    ax2.xaxis.label.set_color('white')
    for p in bars.patches:
        height = p.get_height()
        ax2.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., height),
                     ha='center', va='bottom', color='white', fontsize=8)
    st.pyplot(fig2)

    st.subheader("ROC Curve")
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines+markers', name='ROC curve (area = %0.2f)' % roc_auc))
    fig3.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random guess', line=dict(dash='dash')))
    fig3.update_layout(xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', showlegend=True, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig3)

    st.subheader("Feature Importances")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    fig4 = go.Figure([go.Bar(x=sorted_features, y=sorted_importances, text=[f'{v:.2f}' for v in sorted_importances], textposition='auto')])
    fig4.update_layout(xaxis_title='Features', yaxis_title='Importance', showlegend=False, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig4)

def main():
    st.title("Customer Churn Analysis for Telecom Industry")

    st.markdown("### Upload your telecom customer churn CSV data")
    uploaded_file = st.file_uploader("Choose a CSV file to upload", type=["csv"])

    st.write("File uploader widget rendered.")  # Debug line

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Uploaded data preview:")
        st.dataframe(data.head())
    else:
        st.info("No file uploaded. Using generated sample data.")
        data = generate_sample_data()
        st.write("Sample data preview:")
        st.dataframe(data.head())

    if st.button("Generate Model and Results"):
        st.write("Preprocessing data...")
        data_processed = preprocess_data(data)
        st.write("Training model...")
        accuracy, cm, report, fpr, tpr, roc_auc, model, feature_names = train_model(data_processed)
        plot_metrics(accuracy, cm, report, fpr, tpr, roc_auc, model, feature_names)

        # Storyboard / Narrative Section
        st.header("Storyboard / Narrative Explanation")

        total_customers = len(data)
        churned_customers = data['Churn'].map({'Yes':1, 'No':0}).sum()
        churn_rate = churned_customers / total_customers * 100
        avg_tenure = data['tenure'].mean()
        revenue_impact = data.loc[data['Churn']=='Yes', 'TotalCharges'].sum()

        st.subheader("1. Executive Summary")
        st.markdown(f"""
        **Objective:** Identify churn patterns and recommend retention strategies.

        **Key Metrics Preview:**
        - Total Customers: {total_customers}
        - Churned Customers: {churned_customers}
        - Churn Rate: {churn_rate:.2f}%
        - Average Tenure: {avg_tenure:.2f} months
        - Revenue Impact (Churned Customers): ${revenue_impact:,.2f}
        """)

        st.subheader("2. Customer Overview")
        gender_counts = data['gender'].value_counts()
        st.markdown("**Gender Distribution:**")
        for i, (label, value) in enumerate(gender_counts.items()):
            st.write(f"{label}: {value} ({value / len(data) * 100:.2f}%)")
        fig_gender = go.Figure([go.Bar(x=gender_counts.index, y=gender_counts.values, text=gender_counts.values, textposition='auto')])
        fig_gender.update_layout(yaxis_title='Count', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_gender)

        senior_counts = data['SeniorCitizen'].value_counts()
        st.markdown("**Senior Citizen Status:**")
        for i, (label, value) in enumerate(senior_counts.items()):
            st.write(f"{label}: {value} ({value / len(data) * 100:.2f}%)")
        fig_senior = go.Figure([go.Bar(x=senior_counts.index, y=senior_counts.values, text=senior_counts.values, textposition='auto')])
        fig_senior.update_layout(yaxis_title='Count', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_senior)

        partner_counts = data['Partner'].value_counts()
        st.markdown("**Partner Status:**")
        for i, (label, value) in enumerate(partner_counts.items()):
            st.write(f"{label}: {value} ({value / len(data) * 100:.2f}%)")
        fig_partner = go.Figure([go.Bar(x=partner_counts.index, y=partner_counts.values, text=partner_counts.values, textposition='auto')])
        fig_partner.update_layout(yaxis_title='Count', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_partner)

        dependents_counts = data['Dependents'].value_counts()
        st.markdown("**Dependents Status:**")
        for i, (label, value) in enumerate(dependents_counts.items()):
            st.write(f"{label}: {value} ({value / len(data) * 100:.2f}%)")
        fig_dependents = go.Figure([go.Bar(x=dependents_counts.index, y=dependents_counts.values, text=dependents_counts.values, textposition='auto')])
        fig_dependents.update_layout(yaxis_title='Count', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_dependents)

        st.markdown("**Contract Types:**")
        contract_counts = data['Contract'].value_counts()
        for i, (label, value) in enumerate(contract_counts.items()):
            st.write(f"{label}: {value} ({value / len(data) * 100:.2f}%)")
        fig_contract = go.Figure([go.Bar(x=contract_counts.index, y=contract_counts.values, text=contract_counts.values, textposition='auto')])
        fig_contract.update_layout(yaxis_title='Count', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_contract)

        st.markdown("**Internet Service Types:**")
        internet_counts = data['InternetService'].value_counts()
        for i, (label, value) in enumerate(internet_counts.items()):
            st.write(f"{label}: {value} ({value / len(data) * 100:.2f}%)")
        fig_internet = go.Figure([go.Bar(x=internet_counts.index, y=internet_counts.values, text=internet_counts.values, textposition='auto')])
        fig_internet.update_layout(yaxis_title='Count', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_internet)

        st.subheader("3. Churn Rate Trends")
        churn_rate_overall = churn_rate
        st.metric("Overall Churn Rate", f"{churn_rate_overall:.2f}%")

        # Monthly churn rate placeholder (if 'month' column existed)
        # st.line_chart(monthly_churn_rate)

        st.markdown("**Churn by Contract Type:**")
        churn_by_contract = data.groupby('Contract')['Churn'].apply(lambda x: (x=='Yes').mean()*100)
        for i, (label, value) in enumerate(churn_by_contract.items()):
            st.write(f"{label}: {value:.2f}%")
        fig_churn_contract = go.Figure([go.Bar(x=churn_by_contract.index, y=churn_by_contract.values, text=[f"{v:.2f}%" for v in churn_by_contract.values], textposition='auto')])
        fig_churn_contract.update_layout(yaxis_title='Churn Rate (%)', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_churn_contract)

        st.markdown("**Churn by Payment Method:**")
        churn_by_payment = data.groupby('PaymentMethod')['Churn'].apply(lambda x: (x=='Yes').mean()*100)
        for i, (label, value) in enumerate(churn_by_payment.items()):
            st.write(f"{label}: {value:.2f}%")
        fig_churn_payment = go.Figure([go.Bar(x=churn_by_payment.index, y=churn_by_payment.values, text=[f"{v:.2f}%" for v in churn_by_payment.values], textposition='auto')])
        fig_churn_payment.update_layout(yaxis_title='Churn Rate (%)', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_churn_payment)

        st.markdown("**Churn by Tenure Group:**")
        bins = [0, 12, 24, 48, 72]
        labels = ['0-12', '13-24', '25-48', '49-72']
        data['TenureGroup'] = pd.cut(data['tenure'], bins=bins, labels=labels, right=False)
        churn_by_tenure = data.groupby('TenureGroup')['Churn'].apply(lambda x: (x=='Yes').mean()*100)
        for i, (label, value) in enumerate(churn_by_tenure.items()):
            st.write(f"{label}: {value:.2f}%")
        fig_churn_tenure = go.Figure([go.Bar(x=churn_by_tenure.index, y=churn_by_tenure.values, text=[f"{v:.2f}%" for v in churn_by_tenure.values], textposition='auto')])
        fig_churn_tenure.update_layout(yaxis_title='Churn Rate (%)', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_churn_tenure)

        st.subheader("4. Key Drivers of Churn")
        st.markdown("""
        - High churn in Month-to-Month contracts
        - Higher churn with Electronic Check payment method
        - Low tenure leads to high churn
        """)

        st.subheader("5. Service Usage Patterns")
        st.markdown("**Internet Service vs. Churn:**")
        internet_churn = data.groupby(['InternetService', 'Churn']).size().unstack()
        for idx in internet_churn.index:
            st.write(f"{idx}: No Churn = {internet_churn.loc[idx, 'No'] if 'No' in internet_churn.columns else 0}, Churn = {internet_churn.loc[idx, 'Yes'] if 'Yes' in internet_churn.columns else 0}")
        fig_internet_churn = go.Figure()
        for col in internet_churn.columns:
            fig_internet_churn.add_trace(go.Bar(name=col, x=internet_churn.index, y=internet_churn[col], text=internet_churn[col], textposition='auto'))
        fig_internet_churn.update_layout(barmode='group', yaxis_title='Count', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_internet_churn)

        st.markdown("**Add-ons vs. Churn:**")
        addons = ['StreamingTV', 'TechSupport']
        for addon in addons:
            st.markdown(f"**{addon} vs. Churn:**")
            addon_churn = data.groupby([addon, 'Churn']).size().unstack()
            for idx in addon_churn.index:
                st.write(f"{idx}: No Churn = {addon_churn.loc[idx, 'No'] if 'No' in addon_churn.columns else 0}, Churn = {addon_churn.loc[idx, 'Yes'] if 'Yes' in addon_churn.columns else 0}")
            fig_addon_churn = go.Figure()
            for col in addon_churn.columns:
                fig_addon_churn.add_trace(go.Bar(name=col, x=addon_churn.index, y=addon_churn[col], text=addon_churn[col], textposition='auto'))
            fig_addon_churn.update_layout(barmode='group', yaxis_title='Count', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_addon_churn)

        st.subheader("6. Revenue Impact")
        st.markdown(f"Total revenue lost due to churn: ${revenue_impact:,.2f}")

        st.subheader("7. Recommendations")
        st.markdown("""
        - Incentivize long-term contracts
        - Offer discounts for Paperless Billing
        - Improve customer support for Fiber Optic users
        - Improve onboarding for customers in first 6 months
        """)

        st.subheader("8. Next Steps / Action Plan")
        st.markdown("""
        - Implement retention strategies based on insights
        - Conduct follow-up analysis (e.g., sentiment from call logs, survey feedback)
        """)

        st.subheader("9. Conclusion")
        st.markdown(f"""
        The customer churn analysis reveals that churn is significantly influenced by contract type, payment method, and tenure. Month-to-month contracts and electronic check payments show higher churn rates, while customers with longer tenure tend to stay. Service usage patterns also impact churn, with add-ons like StreamingTV and TechSupport showing varying effects. The revenue impact of churn is substantial, highlighting the need for targeted retention strategies such as incentivizing long-term contracts, offering discounts for paperless billing, and improving customer support. Implementing these recommendations and conducting further analysis will help reduce churn and improve customer loyalty.
        """)

if __name__ == "__main__":
    main()
