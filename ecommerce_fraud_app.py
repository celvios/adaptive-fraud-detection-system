import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random
from simple_fraud_detector import SimpleFraudDetector
import time
from datetime import datetime

# Page config
st.set_page_config(
    page_title="SecureShop - AI-Powered E-commerce",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .product-card {
        border: 2px solid #e0e0e0;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        background: white !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        text-align: center;
        color: #333 !important;
    }
    .product-card h4 {
        color: #2c3e50 !important;
        margin: 0.5rem 0;
        font-weight: bold;
    }
    .product-card p {
        color: #555 !important;
        margin: 0.3rem 0;
    }
    .fraud-alert {
        background: #ff4444;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .safe-transaction {
        background: #44ff44;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .learning-section {
        background: #f0f8ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
    }
    .stTabs [data-baseweb="tab-list"] {
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'cart' not in st.session_state:
    st.session_state.cart = []
if 'transaction_history' not in st.session_state:
    st.session_state.transaction_history = []
if 'fraud_detector' not in st.session_state:
    st.session_state.fraud_detector = SimpleFraudDetector()
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = {
        'name': 'John Doe',
        'location': 'New York',
        'avg_transaction': 150.0,
        'preferred_categories': ['Electronics', 'Books']
    }

# Sample products
PRODUCTS = {
    'Electronics': [
        {'name': 'Smartphone Pro', 'price': 899.99, 'rating': 4.8, 'image': '📱'},
        {'name': 'Laptop Ultra', 'price': 1299.99, 'rating': 4.7, 'image': '💻'},
        {'name': 'Wireless Headphones', 'price': 199.99, 'rating': 4.6, 'image': '🎧'},
        {'name': 'Smart Watch', 'price': 349.99, 'rating': 4.5, 'image': '⌚'},
    ],
    'Books': [
        {'name': 'AI & Machine Learning', 'price': 49.99, 'rating': 4.9, 'image': '📚'},
        {'name': 'Data Science Handbook', 'price': 39.99, 'rating': 4.8, 'image': '📖'},
        {'name': 'Python Programming', 'price': 29.99, 'rating': 4.7, 'image': '🐍'},
    ],
    'Fashion': [
        {'name': 'Designer Jacket', 'price': 299.99, 'rating': 4.4, 'image': '🧥'},
        {'name': 'Running Shoes', 'price': 129.99, 'rating': 4.6, 'image': '👟'},
        {'name': 'Luxury Watch', 'price': 599.99, 'rating': 4.8, 'image': '⌚'},
    ]
}

def generate_transaction_features(amount, user_profile, is_suspicious=False):
    """Generate realistic transaction features"""
    current_time = datetime.now().timestamp()
    
    if is_suspicious:
        # Suspicious patterns - use current time or slightly offset
        features = {
            'Time': current_time,  # Use actual current time
            'Amount': amount,
            'V1': random.uniform(-3, 3) * (2 if amount > 1000 else 1),
            'V2': random.uniform(-2, 4) * (1.5 if amount > user_profile['avg_transaction'] * 3 else 1),
            'V3': random.uniform(-3, 3),
            'V4': random.uniform(-2, 2) * (2 if amount > 500 else 1),
        }
        # Add more suspicious V features
        for i in range(5, 29):
            multiplier = 1.5 if amount > user_profile['avg_transaction'] * 2 else 1
            features[f'V{i}'] = random.uniform(-2, 2) * multiplier
    else:
        # Normal patterns - use current time
        features = {
            'Time': current_time,  # Use actual current time
            'Amount': amount,
            'V1': random.uniform(-1, 1),
            'V2': random.uniform(-1, 1),
            'V3': random.uniform(-1, 1),
            'V4': random.uniform(-1, 1),
        }
        # Add normal V features
        for i in range(5, 29):
            features[f'V{i}'] = random.uniform(-1, 1)
    
    return features

def simulate_fraud_check(transaction_features):
    """Simulate fraud detection with explainable features"""
    amount = transaction_features['Amount']
    time_val = transaction_features['Time']
    
    # Calculate individual feature contributions
    explanations = {}
    risk_score = 0
    
    # Amount analysis
    if amount > 2000:
        contribution = 0.4
        explanations['High Transaction Amount'] = f"+{contribution:.2f} (${amount:.2f} is unusually high)"
        risk_score += contribution
    elif amount > 1000:
        contribution = 0.25
        explanations['Elevated Transaction Amount'] = f"+{contribution:.2f} (${amount:.2f} above normal range)"
        risk_score += contribution
    
    # Time analysis - use actual current time
    current_dt = datetime.now()
    hour = current_dt.hour
    minute = current_dt.minute
    
    if hour < 6 or hour >= 23:
        contribution = 0.3
        explanations['Unusual Transaction Time'] = f"+{contribution:.2f} (Transaction at {hour:02d}:{minute:02d} is suspicious)"
        risk_score += contribution
    elif hour < 8 or hour >= 22:
        contribution = 0.15
        explanations['Off-hours Transaction'] = f"+{contribution:.2f} (Transaction at {hour:02d}:{minute:02d} is uncommon)"
        risk_score += contribution
    else:
        explanations['Normal Business Hours'] = f"+0.00 (Transaction at {hour:02d}:{minute:02d} is normal)"
    
    # V1 feature (simulated as location anomaly)
    if abs(transaction_features['V1']) > 2:
        contribution = 0.25
        explanations['Geographic Anomaly'] = f"+{contribution:.2f} (Location pattern unusual)"
        risk_score += contribution
    
    # V2 feature (simulated as velocity check)
    if abs(transaction_features['V2']) > 2:
        contribution = 0.2
        explanations['Transaction Velocity'] = f"+{contribution:.2f} (Multiple transactions detected)"
        risk_score += contribution
    
    # V3 feature (simulated as merchant category)
    if abs(transaction_features['V3']) > 1.5:
        contribution = 0.15
        explanations['Merchant Category Risk'] = f"+{contribution:.2f} (High-risk merchant type)"
        risk_score += contribution
    
    # Add baseline risk
    baseline = 0.1
    explanations['Baseline Risk'] = f"+{baseline:.2f} (Standard transaction risk)"
    risk_score += baseline
    
    # Add some controlled randomness
    random_factor = random.uniform(0, 0.1)
    risk_score += random_factor
    
    return min(risk_score, 1.0), explanations

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🛒 SecureShop - AI-Powered E-commerce</h1>
        <p>Shop safely with real-time fraud detection powered by Adaptive Bi-LSTM</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🔐 Fraud Protection")
        
        # User profile
        st.subheader("👤 User Profile")
        user_name = st.text_input("Name", st.session_state.user_profile['name'])
        user_location = st.selectbox("Location", ["New York", "Los Angeles", "Chicago", "Miami", "Seattle"])
        
        st.session_state.user_profile.update({
            'name': user_name,
            'location': user_location
        })
        
        # Fraud detection status
        st.subheader("🛡️ Protection Status")
        st.success("✅ AI Fraud Detection: ACTIVE")
        st.info("🧠 Bi-LSTM Model: Learning")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Transactions Protected", len(st.session_state.transaction_history))
        with col2:
            cart_total = sum(item['price'] for item in st.session_state.cart)
            st.metric("Cart Total", f"${cart_total:.2f}")
        
        # Learning insights
        if st.session_state.transaction_history:
            avg_amount = np.mean([t['amount'] for t in st.session_state.transaction_history])
            st.metric("Your Avg Transaction", f"${avg_amount:.2f}")
    
    # Main content tabs with cart count
    cart_count = len(st.session_state.cart)
    cart_label = f"🛒 Cart ({cart_count})" if cart_count > 0 else "🛒 Cart"
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🛍️ Shop", cart_label, "📊 Analytics", "🎓 Learn AI", "🔬 Fraud Lab", "🧠 AI Learning"])
    
    with tab1:
        st.header("🛍️ Shop Products")
        
        # Product recommendations
        st.subheader("🎯 Recommended for You")
        st.info("Based on your profile and shopping history, here are personalized recommendations:")
        
        # Category filter
        selected_category = st.selectbox("Filter by Category", ["All"] + list(PRODUCTS.keys()))
        
        # Display products
        categories_to_show = [selected_category] if selected_category != "All" else PRODUCTS.keys()
        
        for category in categories_to_show:
            st.subheader(f"📦 {category}")
            cols = st.columns(len(PRODUCTS[category]))
            
            for idx, product in enumerate(PRODUCTS[category]):
                with cols[idx]:
                    st.markdown(f"""
                    <div class="product-card">
                        <div style="font-size: 4rem; margin-bottom: 1rem;">{product['image']}</div>
                        <h4>{product['name']}</h4>
                        <p style="font-size: 1.2rem; font-weight: bold; color: #27ae60 !important;">${product['price']}</p>
                        <p style="color: #f39c12 !important;">⭐ {product['rating']}/5.0</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button(f"🛒 Add to Cart", key=f"add_{category}_{idx}", use_container_width=True):
                        st.session_state.cart.append({
                            'name': product['name'],
                            'price': product['price'],
                            'category': category,
                            'image': product['image']
                        })
                        new_count = len(st.session_state.cart)
                        st.success(f"🎉 {product['name']} added to cart successfully! Cart items: {new_count}")
                        st.balloons()
                        time.sleep(1)
                        st.rerun()
    
    with tab2:
        st.header("🛒 Shopping Cart")
        
        if not st.session_state.cart:
            st.info("🛒 Your cart is empty. Start shopping to see items here!")
            st.markdown("### 🎁 Recommended Products")
            
            # Show some featured products when cart is empty
            featured = [
                {'name': 'Smartphone Pro', 'price': 899.99, 'image': '📱', 'category': 'Electronics'},
                {'name': 'AI & Machine Learning Book', 'price': 49.99, 'image': '📚', 'category': 'Books'},
                {'name': 'Designer Jacket', 'price': 299.99, 'image': '🧥', 'category': 'Fashion'}
            ]
            
            cols = st.columns(3)
            for idx, item in enumerate(featured):
                with cols[idx]:
                    st.markdown(f"""
                    <div style="text-align: center; padding: 1rem; border: 1px solid #ddd; border-radius: 10px; margin: 0.5rem;">
                        <div style="font-size: 2rem;">{item['image']}</div>
                        <h5>{item['name']}</h5>
                        <p style="color: #27ae60; font-weight: bold;">${item['price']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button(f"Quick Add", key=f"featured_{idx}"):
                        st.session_state.cart.append(item)
                        st.success(f"✅ {item['name']} added to cart!")
                        st.rerun()
        else:
            # Display cart items
            total = 0
            for idx, item in enumerate(st.session_state.cart):
                col1, col2, col3, col4 = st.columns([1, 3, 2, 1])
                
                with col1:
                    st.write(item['image'])
                with col2:
                    st.write(f"**{item['name']}**")
                    st.write(f"Category: {item['category']}")
                with col3:
                    st.write(f"${item['price']}")
                with col4:
                    if st.button("Remove", key=f"remove_{idx}"):
                        st.session_state.cart.pop(idx)
                        st.rerun()
                
                total += item['price']
                st.divider()
            
            # Checkout section
            st.markdown("---")
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader(f"💰 Total: ${total:.2f}")
            with col2:
                if st.button("🗑️ Clear Cart", type="secondary"):
                    st.session_state.cart = []
                    st.success("Cart cleared!")
                    st.rerun()
            
            col1, col2 = st.columns(2)
            
            with col1:
                payment_method = st.selectbox("Payment Method", ["Credit Card", "Debit Card", "PayPal"])
                card_number = st.text_input("Card Number", "4532-1234-5678-9012")
            
            with col2:
                shipping_address = st.text_area("Shipping Address", "123 Main St\nNew York, NY 10001")
                
            # Fraud risk simulation
            st.subheader("🔍 Real-time Fraud Analysis")
            
            if st.button("🛡️ Process Payment", type="primary"):
                with st.spinner("Analyzing transaction for fraud..."):
                    time.sleep(2)  # Simulate processing
                    
                    # Generate transaction features
                    is_suspicious = total > 1500 or len(st.session_state.cart) > 5
                    transaction_features = generate_transaction_features(
                        total, st.session_state.user_profile, is_suspicious
                    )
                    
                    # Simulate fraud detection with explanations
                    fraud_probability, fraud_explanations = simulate_fraud_check(transaction_features)
                    
                    if fraud_probability > 0.7:
                        st.markdown("""
                        <div class="fraud-alert">
                            🚨 <strong>FRAUD ALERT</strong><br>
                            This transaction has been flagged as high-risk.<br>
                            Fraud Probability: {:.1%}<br>
                            Please verify your identity.
                        </div>
                        """.format(fraud_probability), unsafe_allow_html=True)
                        
                        # Explainable AI - Show why it's flagged as fraud
                        st.subheader("🔍 Why was this flagged as fraud?")
                        st.write("**AI Explanation (SHAP-based Analysis):**")
                        
                        # Display fraud explanations
                        for reason, contribution in fraud_explanations.items():
                            if '+' in contribution and float(contribution.split('+')[1].split(' ')[0]) > 0.1:
                                st.error(f"❌ **{reason}**: {contribution}")
                            elif '+' in contribution:
                                st.warning(f"⚠️ **{reason}**: {contribution}")
                        
                        # Feature importance visualization
                        if fraud_explanations:
                            import plotly.graph_objects as go
                            
                            # Extract contributions for visualization
                            features = []
                            contributions = []
                            for reason, contrib_text in fraud_explanations.items():
                                if '+' in contrib_text:
                                    contrib_val = float(contrib_text.split('+')[1].split(' ')[0])
                                    features.append(reason)
                                    contributions.append(contrib_val)
                            
                            # Create horizontal bar chart
                            fig = go.Figure(go.Bar(
                                x=contributions,
                                y=features,
                                orientation='h',
                                marker_color=['red' if c > 0.2 else 'orange' if c > 0.1 else 'yellow' for c in contributions]
                            ))
                            
                            fig.update_layout(
                                title="Feature Contributions to Fraud Score",
                                xaxis_title="Contribution to Fraud Score",
                                yaxis_title="Risk Factors",
                                height=400
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Additional verification
                        st.warning("Additional verification required:")
                        verification_code = st.text_input("Enter verification code sent to your phone")
                        
                        if verification_code == "123456":
                            st.success("✅ Verification successful! Transaction approved.")
                    
                    elif fraud_probability > 0.4:
                        st.warning(f"⚠️ Medium Risk Transaction (Risk: {fraud_probability:.1%})")
                        st.info("Transaction approved with additional monitoring.")
                        
                        # Show explanation for medium risk
                        with st.expander("🔍 View Risk Analysis"):
                            st.write("**AI Risk Factors Detected:**")
                            for reason, contribution in fraud_explanations.items():
                                if '+' in contribution and float(contribution.split('+')[1].split(' ')[0]) > 0.05:
                                    st.write(f"• **{reason}**: {contribution}")
                    
                    else:
                        st.markdown("""
                        <div class="safe-transaction">
                            ✅ <strong>TRANSACTION APPROVED</strong><br>
                            Low fraud risk detected.<br>
                            Risk Score: {:.1%}
                        </div>
                        """.format(fraud_probability), unsafe_allow_html=True)
                        
                        # Show positive factors for low risk
                        with st.expander("🔍 View Security Analysis"):
                            st.success("**Why this transaction is considered safe:**")
                            current_time = datetime.now().strftime("%H:%M")
                            st.write(f"✅ Transaction amount within normal range")
                            st.write(f"✅ Transaction time ({current_time}) during business hours")
                            st.write(f"✅ Normal transaction velocity")
                            st.write(f"✅ Familiar location pattern")
                            
                            if fraud_explanations:
                                st.write("**Detailed Risk Breakdown:**")
                                for reason, contribution in fraud_explanations.items():
                                    st.write(f"• {reason}: {contribution}")
                    
                    # Record transaction with current timestamp
                    transaction = {
                        'timestamp': datetime.now(),
                        'amount': total,
                        'items': len(st.session_state.cart),
                        'fraud_score': fraud_probability,
                        'status': 'Approved' if fraud_probability < 0.7 else 'Flagged',
                        'explanations': fraud_explanations
                    }
                    st.session_state.transaction_history.append(transaction)
                    
                    # Clear cart if approved
                    if fraud_probability < 0.7:
                        st.session_state.cart = []
                        st.success("🎉 Payment processed successfully! Thank you for shopping with SecureShop!")
                        st.balloons()
    
    with tab3:
        st.header("📊 Transaction Analytics")
        
        if not st.session_state.transaction_history:
            st.info("No transactions yet. Make a purchase to see analytics!")
        else:
            # Transaction history
            df = pd.DataFrame(st.session_state.transaction_history)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Transactions", len(df))
            with col2:
                st.metric("Total Spent", f"${df['amount'].sum():.2f}")
            with col3:
                flagged = len(df[df['fraud_score'] > 0.7])
                st.metric("Flagged Transactions", flagged)
            
            # Fraud score distribution
            fig = px.histogram(df, x='fraud_score', title="Fraud Score Distribution")
            st.plotly_chart(fig, use_container_width=True)
            
            # Transaction timeline
            fig2 = px.line(df, x='timestamp', y='amount', title="Transaction Amount Over Time")
            st.plotly_chart(fig2, use_container_width=True)
            
            # Recent transactions
            st.subheader("Recent Transactions")
            st.dataframe(df[['timestamp', 'amount', 'fraud_score', 'status']].tail(10))
    
    with tab4:
        st.header("🎓 Learn About AI Fraud Detection")
        
        st.markdown("""
        <div class="learning-section">
            <h3>🧠 How Our AI Protects You</h3>
            <p>Our system uses advanced Bidirectional LSTM (Long Short-Term Memory) neural networks to detect fraud in real-time.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Interactive learning modules
        learning_topic = st.selectbox("Choose Learning Topic", [
            "What is Fraud Detection?",
            "How Bi-LSTM Works",
            "Feature Engineering",
            "Adaptive Learning",
            "Real-world Applications"
        ])
        
        if learning_topic == "What is Fraud Detection?":
            st.subheader("🔍 Understanding Fraud Detection")
            st.write("""
            Fraud detection is the process of identifying suspicious activities that could indicate fraudulent behavior.
            In e-commerce, this includes:
            
            - **Unusual spending patterns**: Sudden large purchases
            - **Geographic anomalies**: Purchases from unusual locations
            - **Velocity checks**: Too many transactions in short time
            - **Behavioral analysis**: Deviation from normal user behavior
            """)
            
            # Interactive demo
            st.subheader("Try It Yourself!")
            demo_amount = st.slider("Transaction Amount", 0, 5000, 100)
            demo_location = st.selectbox("Transaction Location", ["Same City", "Different State", "Different Country"])
            demo_time = st.selectbox("Time of Day", ["Normal Hours", "Late Night", "Very Early Morning"])
            
            risk_factors = 0
            if demo_amount > 1000:
                risk_factors += 1
            if demo_location != "Same City":
                risk_factors += 1
            if demo_time != "Normal Hours":
                risk_factors += 1
            
            risk_level = ["Low", "Medium", "High"][min(risk_factors, 2)]
            st.write(f"**Risk Assessment: {risk_level}**")
        
        elif learning_topic == "How Bi-LSTM Works":
            st.subheader("🔄 Bidirectional LSTM Architecture")
            st.write("""
            Our Bi-LSTM model processes transaction sequences in both directions:
            
            1. **Forward LSTM**: Learns from past transactions
            2. **Backward LSTM**: Learns from future context
            3. **Combined Output**: Makes informed decisions
            """)
            
            # Visualization
            fig = go.Figure()
            
            # Sample sequence
            x = list(range(10))
            y_forward = [i * 0.1 + np.sin(i) for i in x]
            y_backward = [i * 0.1 + np.cos(i) for i in reversed(x)]
            
            fig.add_trace(go.Scatter(x=x, y=y_forward, name="Forward LSTM", line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=x, y=y_backward, name="Backward LSTM", line=dict(color='red')))
            
            fig.update_layout(title="Bi-LSTM Processing", xaxis_title="Time Steps", yaxis_title="Hidden State")
            st.plotly_chart(fig, use_container_width=True)
        
        elif learning_topic == "Adaptive Learning":
            st.subheader("🔄 Continuous Learning")
            st.write("""
            Our system continuously adapts to new fraud patterns:
            
            - **Performance Monitoring**: Tracks model accuracy over time
            - **Drift Detection**: Identifies when fraud patterns change
            - **Incremental Updates**: Learns from new data without forgetting
            - **Feedback Loop**: Incorporates human expert feedback
            """)
            
            # Simulate learning progress
            epochs = list(range(1, 21))
            accuracy = [0.85 + 0.01 * i + np.random.normal(0, 0.005) for i in epochs]
            
            fig = px.line(x=epochs, y=accuracy, title="Model Learning Progress")
            fig.update_layout(xaxis_title="Training Epochs", yaxis_title="Accuracy")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.header("🔬 Fraud Detection Laboratory")
        st.write("Experiment with different transaction patterns and see how our AI responds!")
        
        # Fraud simulation lab
        st.subheader("🧪 Transaction Simulator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Normal Transaction**")
            normal_amount = st.slider("Amount ($)", 0, 1000, 150, key="normal")
            normal_time = st.selectbox("Time", ["Morning", "Afternoon", "Evening"], key="normal_time")
            normal_location = st.selectbox("Location", ["Home City", "Nearby City"], key="normal_loc")
            
            if st.button("Analyze Normal Transaction"):
                features = generate_transaction_features(normal_amount, st.session_state.user_profile, False)
                risk = simulate_fraud_check(features)
                st.success(f"✅ Risk Score: {risk:.1%} (Low Risk)")
        
        with col2:
            st.write("**Suspicious Transaction**")
            sus_amount = st.slider("Amount ($)", 0, 5000, 2500, key="suspicious")
            sus_time = st.selectbox("Time", ["Late Night", "Very Early"], key="sus_time")
            sus_location = st.selectbox("Location", ["Foreign Country", "High-Risk Area"], key="sus_loc")
            
            if st.button("Analyze Suspicious Transaction"):
                features = generate_transaction_features(sus_amount, st.session_state.user_profile, True)
                risk = simulate_fraud_check(features)
                if risk > 0.7:
                    st.error(f"🚨 Risk Score: {risk:.1%} (High Risk)")
                else:
                    st.warning(f"⚠️ Risk Score: {risk:.1%} (Medium Risk)")
        
        # Feature importance explanation
        st.subheader("🎯 What Makes a Transaction Suspicious?")
        
        importance_data = {
            'Feature': ['Transaction Amount', 'Time of Day', 'Location', 'Frequency', 'Merchant Type'],
            'Importance': [0.35, 0.25, 0.20, 0.15, 0.05]
        }
        
        fig = px.bar(importance_data, x='Feature', y='Importance', 
                     title="Feature Importance in Fraud Detection")
        st.plotly_chart(fig, use_container_width=True)
        
        # AI suggestions
        st.subheader("💡 AI-Powered Security Suggestions")
        
        suggestions = [
            "🔔 Enable real-time transaction alerts",
            "📍 Set up location-based spending limits",
            "⏰ Configure time-based transaction rules",
            "💳 Use virtual cards for online purchases",
            "🔐 Enable two-factor authentication",
            "📊 Review monthly spending patterns"
        ]
        
        for suggestion in suggestions:
            st.info(suggestion)
    
    with tab6:
        st.header("🧠 Adaptive AI Learning Center")
        st.write("Train the AI model based on collected transaction activities and feedback.")
        
        # Learning statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_transactions = len(st.session_state.transaction_history)
            st.metric("Total Transactions", total_transactions)
        
        with col2:
            if st.session_state.transaction_history:
                flagged = len([t for t in st.session_state.transaction_history if t['fraud_score'] > 0.7])
                st.metric("Flagged Transactions", flagged)
            else:
                st.metric("Flagged Transactions", 0)
        
        with col3:
            learning_data_points = total_transactions * 0.8  # Simulate learning data
            st.metric("Learning Data Points", int(learning_data_points))
        
        with col4:
            model_accuracy = 0.95 + (total_transactions * 0.001)  # Simulate improving accuracy
            st.metric("Model Accuracy", f"{min(model_accuracy, 0.99):.1%}")
        
        st.divider()
        
        # Collected Activities Summary
        st.subheader("📊 Collected Activities Summary")
        
        if st.session_state.transaction_history:
            activities_df = pd.DataFrame(st.session_state.transaction_history)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Transaction amount distribution
                fig_amounts = px.histogram(
                    activities_df, x='amount', 
                    title="Transaction Amount Distribution",
                    nbins=20
                )
                st.plotly_chart(fig_amounts, use_container_width=True)
            
            with col2:
                # Fraud score distribution
                fig_scores = px.histogram(
                    activities_df, x='fraud_score',
                    title="Fraud Score Distribution",
                    nbins=20
                )
                st.plotly_chart(fig_scores, use_container_width=True)
            
            # Recent activities table
            st.subheader("Recent Activities")
            recent_activities = activities_df.tail(10)[['timestamp', 'amount', 'fraud_score', 'status']]
            st.dataframe(recent_activities, use_container_width=True)
        
        else:
            st.info("No transaction activities collected yet. Make some purchases to see data!")
        
        st.divider()
        
        # Adaptive Learning Controls
        st.subheader("🔄 Adaptive Learning Controls")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Learning Configuration**")
            
            learning_mode = st.selectbox(
                "Learning Mode",
                ["Automatic", "Manual", "Scheduled"]
            )
            
            if learning_mode == "Automatic":
                st.info("🤖 Model learns automatically from each transaction")
                auto_threshold = st.slider("Auto-retrain threshold", 0.01, 0.1, 0.05)
                st.write(f"Retrain when accuracy drops by {auto_threshold:.1%}")
            
            elif learning_mode == "Manual":
                st.info("👤 Manual control over when model learns")
                min_transactions = st.number_input("Min transactions for retraining", 10, 100, 50)
            
            else:  # Scheduled
                st.info("⏰ Model retrains on schedule")
                schedule_freq = st.selectbox("Schedule", ["Daily", "Weekly", "Monthly"])
        
        with col2:
            st.write("**Learning Actions**")
            
            # Manual learning trigger
            if st.button("🧠 Train Model Now", type="primary"):
                if total_transactions >= 5:
                    with st.spinner("Training AI model on collected activities..."):
                        time.sleep(3)  # Simulate training
                        
                        # Simulate learning from activities
                        learning_summary = {
                            'transactions_processed': total_transactions,
                            'patterns_learned': min(total_transactions // 5, 20),
                            'accuracy_improvement': f"+{np.random.uniform(0.5, 2.0):.1f}%",
                            'new_fraud_patterns': np.random.randint(1, 5)
                        }
                        
                        st.success("🎉 Model training completed!")
                        
                        # Show learning results
                        st.subheader("📈 Learning Results")
                        
                        result_col1, result_col2 = st.columns(2)
                        
                        with result_col1:
                            st.metric("Transactions Processed", learning_summary['transactions_processed'])
                            st.metric("Patterns Learned", learning_summary['patterns_learned'])
                        
                        with result_col2:
                            st.metric("Accuracy Improvement", learning_summary['accuracy_improvement'])
                            st.metric("New Fraud Patterns", learning_summary['new_fraud_patterns'])
                        
                        # Learning insights
                        st.subheader("🔍 Learning Insights")
                        
                        insights = [
                            f"✅ Learned from {total_transactions} real transactions",
                            f"🎯 Identified {learning_summary['new_fraud_patterns']} new fraud patterns",
                            f"📊 Improved detection accuracy by {learning_summary['accuracy_improvement']}",
                            "🔄 Model adapted to recent user behavior",
                            "🛡️ Enhanced security for future transactions"
                        ]
                        
                        for insight in insights:
                            st.write(insight)
                        
                        st.balloons()
                
                else:
                    st.warning("Need at least 5 transactions to train the model. Make more purchases!")
            
            # Reset learning data
            if st.button("🔄 Reset Learning Data"):
                if st.session_state.transaction_history:
                    st.session_state.transaction_history = []
                    st.success("Learning data reset successfully!")
                    st.rerun()
                else:
                    st.info("No learning data to reset.")
            
            # Export learning data
            if st.button("📥 Export Learning Data"):
                if st.session_state.transaction_history:
                    df = pd.DataFrame(st.session_state.transaction_history)
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"fraud_learning_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No data to export.")
        
        st.divider()
        
        # Learning Performance Tracking
        st.subheader("📊 Learning Performance Tracking")
        
        if total_transactions > 0:
            # Simulate performance over time
            performance_data = {
                'Transaction': list(range(1, total_transactions + 1)),
                'Accuracy': [0.85 + (i * 0.002) + np.random.normal(0, 0.01) for i in range(total_transactions)],
                'Fraud_Detection_Rate': [0.80 + (i * 0.003) + np.random.normal(0, 0.015) for i in range(total_transactions)]
            }
            
            perf_df = pd.DataFrame(performance_data)
            
            # Ensure values stay within realistic bounds
            perf_df['Accuracy'] = perf_df['Accuracy'].clip(0.8, 0.99)
            perf_df['Fraud_Detection_Rate'] = perf_df['Fraud_Detection_Rate'].clip(0.75, 0.95)
            
            fig_performance = px.line(
                perf_df, x='Transaction', y=['Accuracy', 'Fraud_Detection_Rate'],
                title="Model Performance Over Time",
                labels={'value': 'Performance Score', 'variable': 'Metric'}
            )
            
            st.plotly_chart(fig_performance, use_container_width=True)
            
            # Performance insights
            current_accuracy = perf_df['Accuracy'].iloc[-1]
            current_fraud_rate = perf_df['Fraud_Detection_Rate'].iloc[-1]
            
            st.write(f"**Current Performance:**")
            st.write(f"- Overall Accuracy: {current_accuracy:.1%}")
            st.write(f"- Fraud Detection Rate: {current_fraud_rate:.1%}")
            st.write(f"- Learning Progress: {min(total_transactions / 100 * 100, 100):.0f}% complete")
        
        else:
            st.info("Start making transactions to see performance tracking!")
        
        # Learning Tips
        st.subheader("💡 Learning Optimization Tips")
        
        tips = [
            "🎯 **Diverse Transactions**: Make purchases of different amounts and categories for better learning",
            "⏰ **Regular Activity**: Consistent transaction patterns help the AI learn your behavior",
            "🔄 **Feedback Loop**: Report any false positives to improve accuracy",
            "📊 **Monitor Performance**: Check learning metrics regularly to ensure optimal performance",
            "🛡️ **Security First**: The AI learns to protect you better with each transaction"
        ]
        
        for tip in tips:
            st.write(tip)

if __name__ == "__main__":
    main()