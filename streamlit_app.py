import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
import matplotlib.pyplot as plt
from fraud_detection_system import AdaptiveFraudDetector
import joblib

# Page configuration
st.set_page_config(
    page_title="Adaptive Credit Card Fraud Detection",
    page_icon="🔒",
    layout="wide"
)

# Initialize session state
if 'detector' not in st.session_state:
    st.session_state.detector = None
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False

def load_sample_data():
    """Load sample data for demonstration"""
    try:
        data = pd.read_csv('creditcard.csv')
        return data
    except FileNotFoundError:
        st.error("Please upload the creditcard.csv file to the project directory")
        return None

def plot_class_distribution(data):
    """Plot class distribution"""
    class_counts = data['Class'].value_counts()
    
    fig = px.pie(
        values=class_counts.values,
        names=['Normal', 'Fraud'],
        title="Transaction Class Distribution"
    )
    return fig

def plot_feature_correlation(data, features):
    """Plot correlation heatmap for selected features"""
    corr_matrix = data[features].corr()
    
    fig = px.imshow(
        corr_matrix,
        title="Feature Correlation Matrix",
        color_continuous_scale="RdBu_r"
    )
    return fig

def plot_model_performance(results):
    """Plot model performance comparison"""
    models = list(results.keys())
    auc_scores = [results[model]['auc'] for model in models]
    
    fig = px.bar(
        x=models,
        y=auc_scores,
        title="Model Performance Comparison (AUC Score)",
        labels={'x': 'Models', 'y': 'AUC Score'}
    )
    fig.update_layout(showlegend=False)
    return fig

def plot_shap_waterfall(shap_values, feature_names, base_value, prediction_value):
    """Create SHAP waterfall plot"""
    # Sort features by absolute SHAP value
    sorted_indices = np.argsort(np.abs(shap_values))[::-1][:10]
    
    features = [feature_names[i] for i in sorted_indices]
    values = [shap_values[i] for i in sorted_indices]
    
    # Create waterfall chart
    fig = go.Figure()
    
    cumulative = base_value
    x_pos = 0
    
    # Add base value
    fig.add_trace(go.Bar(
        x=[x_pos],
        y=[base_value],
        name='Base Value',
        marker_color='lightgray'
    ))
    
    # Add feature contributions
    for i, (feature, value) in enumerate(zip(features, values)):
        x_pos += 1
        color = 'red' if value > 0 else 'blue'
        
        fig.add_trace(go.Bar(
            x=[x_pos],
            y=[value],
            name=f'{feature}: {value:.3f}',
            marker_color=color,
            base=cumulative
        ))
        cumulative += value
    
    # Add final prediction
    x_pos += 1
    fig.add_trace(go.Bar(
        x=[x_pos],
        y=[prediction_value],
        name='Final Prediction',
        marker_color='green'
    ))
    
    fig.update_layout(
        title="SHAP Feature Contributions",
        xaxis_title="Features",
        yaxis_title="SHAP Value",
        showlegend=True
    )
    
    return fig

def main():
    st.title("🔒 Adaptive Credit Card Fraud Detection System")
    st.markdown("### AI-Powered Fraud Detection with Explainable Insights")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Data Overview", "Model Training", "Fraud Detection", "Model Explanation", "Performance Monitoring"]
    )
    
    # Load data
    data = load_sample_data()
    if data is None:
        return
    
    if page == "Data Overview":
        st.header("📊 Data Overview")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Transactions", f"{len(data):,}")
        
        with col2:
            fraud_count = data['Class'].sum()
            st.metric("Fraud Cases", f"{fraud_count:,}")
        
        with col3:
            fraud_rate = data['Class'].mean() * 100
            st.metric("Fraud Rate", f"{fraud_rate:.3f}%")
        
        # Class distribution
        st.subheader("Class Distribution")
        fig_dist = plot_class_distribution(data)
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Feature statistics
        st.subheader("Feature Statistics")
        st.dataframe(data.describe())
        
        # Amount distribution by class
        st.subheader("Transaction Amount Distribution")
        fig_amount = px.box(
            data, 
            x='Class', 
            y='Amount',
            title="Transaction Amount by Class"
        )
        st.plotly_chart(fig_amount, use_container_width=True)
    
    elif page == "Model Training":
        st.header("🤖 Model Training")
        
        if st.button("Train Models", type="primary"):
            with st.spinner("Training ensemble models..."):
                # Initialize detector
                detector = AdaptiveFraudDetector()
                
                # Load and preprocess data
                X, y = detector.load_data('creditcard.csv')
                detector.preprocess_data()
                
                # Train models
                detector.train_ensemble_models()
                detector.create_explainers()
                
                # Store in session state
                st.session_state.detector = detector
                st.session_state.models_trained = True
                
                st.success("Models trained successfully!")
        
        if st.session_state.models_trained:
            st.subheader("Model Performance")
            
            # Evaluate models
            results = st.session_state.detector.evaluate_models()
            
            # Performance chart
            fig_perf = plot_model_performance(results)
            st.plotly_chart(fig_perf, use_container_width=True)
            
            # Detailed results
            st.subheader("Detailed Results")
            for model_name, result in results.items():
                with st.expander(f"{model_name.upper()} Results"):
                    st.write(f"**AUC Score:** {result['auc']:.4f}")
                    
                    # Classification report
                    report = result['classification_report']
                    metrics_df = pd.DataFrame({
                        'Precision': [report['0']['precision'], report['1']['precision']],
                        'Recall': [report['0']['recall'], report['1']['recall']],
                        'F1-Score': [report['0']['f1-score'], report['1']['f1-score']]
                    }, index=['Normal', 'Fraud'])
                    
                    st.dataframe(metrics_df)
    
    elif page == "Fraud Detection":
        st.header("🔍 Real-time Fraud Detection")
        
        if not st.session_state.models_trained:
            st.warning("Please train the models first in the Model Training page.")
            return
        
        st.subheader("Single Transaction Analysis")
        
        # Transaction input
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Transaction Features:**")
            time = st.number_input("Time", value=0.0)
            amount = st.number_input("Amount", value=100.0, min_value=0.0)
        
        with col2:
            st.write("**PCA Features (V1-V28):**")
            v_features = {}
            for i in range(1, 29):
                v_features[f'V{i}'] = st.number_input(f'V{i}', value=0.0, key=f'v{i}')
        
        if st.button("Analyze Transaction"):
            # Create transaction dataframe
            transaction_data = {'Time': [time], 'Amount': [amount]}
            transaction_data.update({k: [v] for k, v in v_features.items()})
            transaction_df = pd.DataFrame(transaction_data)
            
            # Reorder columns to match training data
            transaction_df = transaction_df[st.session_state.detector.feature_names]
            
            # Scale the data
            transaction_scaled = st.session_state.detector.scaler.transform(transaction_df)
            
            # Make prediction
            ensemble_pred, ensemble_proba, individual_preds, individual_probas = \
                st.session_state.detector.predict_ensemble(transaction_scaled)
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Prediction Result")
                if ensemble_pred[0] == 1:
                    st.error(f"🚨 FRAUD DETECTED (Confidence: {ensemble_proba[0]:.3f})")
                else:
                    st.success(f"✅ Normal Transaction (Confidence: {1-ensemble_proba[0]:.3f})")
            
            with col2:
                st.subheader("Individual Model Predictions")
                for model_name, prob in individual_probas.items():
                    st.write(f"**{model_name}:** {prob[0]:.3f}")
        
        # Batch analysis
        st.subheader("Batch Transaction Analysis")
        uploaded_file = st.file_uploader("Upload CSV file with transactions", type="csv")
        
        if uploaded_file is not None:
            batch_data = pd.read_csv(uploaded_file)
            
            if st.button("Analyze Batch"):
                # Process batch
                X_batch = batch_data[st.session_state.detector.feature_names]
                X_batch_scaled = st.session_state.detector.scaler.transform(X_batch)
                
                ensemble_pred, ensemble_proba, _, _ = \
                    st.session_state.detector.predict_ensemble(X_batch_scaled)
                
                # Add results to dataframe
                batch_data['Fraud_Probability'] = ensemble_proba
                batch_data['Prediction'] = ensemble_pred
                
                # Display results
                st.subheader("Batch Analysis Results")
                fraud_count = ensemble_pred.sum()
                st.metric("Detected Frauds", f"{fraud_count} / {len(batch_data)}")
                
                # Show high-risk transactions
                high_risk = batch_data[batch_data['Fraud_Probability'] > 0.5].sort_values(
                    'Fraud_Probability', ascending=False
                )
                
                if len(high_risk) > 0:
                    st.subheader("High-Risk Transactions")
                    st.dataframe(high_risk[['Fraud_Probability', 'Prediction', 'Amount']])
    
    elif page == "Model Explanation":
        st.header("🔬 Model Explanation & Interpretability")
        
        if not st.session_state.models_trained:
            st.warning("Please train the models first in the Model Training page.")
            return
        
        # Feature importance
        st.subheader("Global Feature Importance")
        
        model_choice = st.selectbox(
            "Select model for feature importance",
            ['xgboost', 'random_forest', 'lightgbm', 'logistic_regression']
        )
        
        if st.button("Show Feature Importance"):
            # Get feature importance
            detector = st.session_state.detector
            
            if model_choice in detector.models:
                model = detector.models[model_choice]
                
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                elif model_choice == 'logistic_regression':
                    classifier = model.named_steps['classifier']
                    importances = np.abs(classifier.coef_[0])
                
                # Create dataframe
                feature_imp_df = pd.DataFrame({
                    'Feature': detector.feature_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                
                # Plot
                fig = px.bar(
                    feature_imp_df.head(15),
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title=f"Top 15 Feature Importances - {model_choice.upper()}"
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
        
        # SHAP explanation for individual predictions
        st.subheader("Individual Prediction Explanation")
        
        # Sample some test transactions
        detector = st.session_state.detector
        if hasattr(detector, 'X_test'):
            sample_indices = np.random.choice(len(detector.X_test), 10, replace=False)
            
            selected_idx = st.selectbox(
                "Select transaction to explain",
                sample_indices,
                format_func=lambda x: f"Transaction {x} (Actual: {'Fraud' if detector.y_test.iloc[x] == 1 else 'Normal'})"
            )
            
            if st.button("Explain Prediction"):
                # Get SHAP values
                shap_values = detector.explain_prediction(selected_idx, model_choice)
                
                if shap_values is not None:
                    # Show transaction details
                    st.subheader("Transaction Details")
                    transaction = detector.X_test.iloc[selected_idx]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Amount:** ${transaction['Amount']:.2f}")
                        st.write(f"**Time:** {transaction['Time']:.0f}")
                    
                    with col2:
                        actual_class = detector.y_test.iloc[selected_idx]
                        st.write(f"**Actual Class:** {'Fraud' if actual_class == 1 else 'Normal'}")
                    
                    # SHAP explanation
                    st.subheader("SHAP Feature Contributions")
                    
                    # Create feature contribution dataframe
                    shap_df = pd.DataFrame({
                        'Feature': detector.feature_names,
                        'SHAP_Value': shap_values,
                        'Feature_Value': transaction.values
                    }).sort_values('SHAP_Value', key=abs, ascending=False)
                    
                    # Show top contributing features
                    st.dataframe(shap_df.head(10))
                    
                    # Plot SHAP values
                    fig = px.bar(
                        shap_df.head(15),
                        x='SHAP_Value',
                        y='Feature',
                        orientation='h',
                        title="SHAP Feature Contributions",
                        color='SHAP_Value',
                        color_continuous_scale='RdBu_r'
                    )
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Performance Monitoring":
        st.header("📈 Performance Monitoring & Adaptive Learning")
        
        if not st.session_state.models_trained:
            st.warning("Please train the models first in the Model Training page.")
            return
        
        st.subheader("Model Performance Over Time")
        
        # Simulate performance monitoring
        detector = st.session_state.detector
        
        # Show current performance
        if hasattr(detector, 'performance_history') and detector.performance_history:
            current_perf = detector.performance_history[-1]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                ensemble_auc = current_perf['ensemble']['auc']
                st.metric("Ensemble AUC", f"{ensemble_auc:.4f}")
            
            with col2:
                ensemble_precision = current_perf['ensemble']['classification_report']['1']['precision']
                st.metric("Fraud Precision", f"{ensemble_precision:.4f}")
            
            with col3:
                ensemble_recall = current_perf['ensemble']['classification_report']['1']['recall']
                st.metric("Fraud Recall", f"{ensemble_recall:.4f}")
        
        # Adaptive retraining
        st.subheader("Adaptive Model Retraining")
        
        st.write("""
        The system monitors model performance and automatically triggers retraining when:
        - Performance drops below threshold
        - New fraud patterns are detected
        - Significant data drift is observed
        """)
        
        retrain_threshold = st.slider("Retraining Threshold (AUC drop)", 0.01, 0.1, 0.05)
        
        if st.button("Check Retraining Need"):
            needs_retrain = detector.adaptive_retrain(None, retrain_threshold)
            
            if needs_retrain:
                st.warning("Model performance has degraded. Retraining recommended!")
            else:
                st.success("Model performance is stable. No retraining needed.")
        
        # Model deployment status
        st.subheader("Model Deployment Status")
        
        deployment_status = {
            'Model Version': '1.0.0',
            'Last Updated': '2024-01-15',
            'Status': 'Active',
            'Transactions Processed': '1,234,567',
            'Frauds Detected': '1,234'
        }
        
        for key, value in deployment_status.items():
            st.write(f"**{key}:** {value}")

if __name__ == "__main__":
    main()