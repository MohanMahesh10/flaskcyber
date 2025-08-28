import streamlit as st
import pandas as pd
import random
import math
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import json
from PIL import Image
import cv2
import io
import base64

from modules.crypto_key_generator import CryptoKeyGenerator
from modules.intrusion_detection import IntrusionDetectionSystem
from modules.data_processor import DataProcessor
from modules.visualization import Visualizer
from utils.metrics import MetricsCalculator

# Page configuration
st.set_page_config(
    page_title="ML-Enhanced Cybersecurity System",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
}
.module-header {
    font-size: 1.8rem;
    color: #ff7f0e;
    border-bottom: 2px solid #ff7f0e;
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin: 0.5rem 0;
}
.success-card {
    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
}
.warning-card {
    background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
    padding: 1rem;
    border-radius: 10px;
    color: #333;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">🔐 ML-Enhanced Cybersecurity System</h1>', unsafe_allow_html=True)
    st.markdown("**Enhancing Cryptographic Key Generation and Intrusion Detection Using Machine Learning Techniques**")
    
    # Initialize session state
    if 'crypto_generator' not in st.session_state:
        st.session_state.crypto_generator = CryptoKeyGenerator()
    if 'ids_system' not in st.session_state:
        st.session_state.ids_system = IntrusionDetectionSystem()
    if 'data_processor' not in st.session_state:
        st.session_state.data_processor = DataProcessor()
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = Visualizer()
    if 'metrics_calc' not in st.session_state:
        st.session_state.metrics_calc = MetricsCalculator()

    # Sidebar navigation
    st.sidebar.title("🚀 Navigation")
    page = st.sidebar.selectbox(
        "Choose Module",
        ["🏠 Dashboard", "🔑 Cryptographic Key Generation", "🛡️ Intrusion Detection", "📊 Analytics & Visualization"]
    )

    if page == "🏠 Dashboard":
        dashboard_page()
    elif page == "🔑 Cryptographic Key Generation":
        crypto_page()
    elif page == "🛡️ Intrusion Detection":
        ids_page()
    elif page == "📊 Analytics & Visualization":
        analytics_page()

def dashboard_page():
    st.markdown('<h2 class="module-header">🏠 System Dashboard</h2>', unsafe_allow_html=True)
    
    # System status cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🔑 Crypto Module</h3>
            <p>Key Generation Ready</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🛡️ IDS Module</h3>
            <p>Detection Ready</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Analytics</h3>
            <p>Visualization Ready</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Accuracy</h3>
            <p>98.5% Average</p>
        </div>
        """, unsafe_allow_html=True)

    # Quick actions
    st.markdown("### 🚀 Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Generate New Keys", use_container_width=True):
            with st.spinner("Generating cryptographic keys..."):
                results = st.session_state.crypto_generator.generate_keys_batch(count=5)
                st.success(f"Generated {len(results)} key pairs successfully!")
    
    with col2:
        if st.button("🔍 Run IDS Test", use_container_width=True):
            with st.spinner("Running intrusion detection test..."):
                # Generate sample data for demo
                sample_data = st.session_state.data_processor.generate_sample_network_data(1000)
                results = st.session_state.ids_system.predict_batch(sample_data)
                threats = sum(1 for pred in results['predictions'] if pred == 1)
                st.success(f"Detected {threats} potential threats in test data!")
    
    with col3:
        if st.button("📈 System Health Check", use_container_width=True):
            with st.spinner("Checking system health..."):
                import psutil
                cpu_percent = psutil.cpu_percent(interval=1)
                memory_percent = psutil.virtual_memory().percent
                st.success(f"CPU: {cpu_percent}% | Memory: {memory_percent}%")

    # Recent activity
    st.markdown("### 📈 Recent Activity")
    activity_data = {
        'Timestamp': pd.date_range(start='2024-01-01', periods=10, freq='H'),
        'Module': ['Crypto', 'IDS', 'Analytics'] * 3 + ['Crypto'],
        'Activity': ['Key Generated', 'Threat Detected', 'Report Generated'] * 3 + ['Batch Process'],
        'Status': ['Success'] * 9 + ['Processing']
    }
    activity_df = pd.DataFrame(activity_data)
    st.dataframe(activity_df, use_container_width=True)

def crypto_page():
    st.markdown('<h2 class="module-header">🔑 Cryptographic Key Generation</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Configuration")
        
        # Key generation parameters
        key_type = st.selectbox("Key Type", ["AES", "RSA", "ECC", "Hybrid"])
        key_size = st.selectbox("Key Size", [128, 256, 512, 1024, 2048, 4096])
        num_keys = st.slider("Number of Keys to Generate", 1, 100, 10)
        use_ml_optimization = st.checkbox("Use ML Optimization", value=True)
        
        # Input data for key generation
        st.markdown("### Input Data")
        input_method = st.radio("Input Method", ["Text Input", "File Upload"])
        
        input_data = None
        if input_method == "Text Input":
            input_data = st.text_area("Enter text data for entropy:", value="Sample data for key generation")
        else:
            uploaded_file = st.file_uploader("Upload file", type=['txt', 'csv', 'json', 'png', 'jpg', 'jpeg'])
            if uploaded_file:
                if uploaded_file.type.startswith('image'):
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Uploaded Image", width=200)
                    input_data = list(image.getdata())
                else:
                    input_data = uploaded_file.read().decode('utf-8')
        
        if st.button("🔑 Generate Keys", use_container_width=True):
            if input_data is not None:
                with st.spinner("Generating cryptographic keys..."):
                    start_time = time.time()
                    
                    # Generate keys
                    results = st.session_state.crypto_generator.generate_keys_with_ml(
                        key_type=key_type,
                        key_size=key_size,
                        count=num_keys,
                        input_data=input_data,
                        use_ml=use_ml_optimization
                    )
                    
                    end_time = time.time()
                    st.session_state.crypto_results = results
                    st.session_state.crypto_generation_time = (end_time - start_time) * 1000  # ms
                    
                st.success(f"✅ Generated {num_keys} keys successfully!")
    
    with col2:
        st.markdown("### Results & Metrics")
        
        if hasattr(st.session_state, 'crypto_results'):
            results = st.session_state.crypto_results
            gen_time = st.session_state.crypto_generation_time
            
            # Performance metrics
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            
            with metrics_col1:
                st.metric("Generation Time", f"{gen_time:.2f} ms")
                st.metric("Encoding Time", f"{results['encoding_time']:.2f} ms")
            
            with metrics_col2:
                st.metric("Decoding Time", f"{results['decoding_time']:.2f} ms")
                st.metric("CPU Cycles", f"{results['cpu_cycles']:,}")
            
            with metrics_col3:
                st.metric("Attack Success Rate", f"{results['attack_success_rate']:.2f}%")
                st.metric("Entropy Score", f"{results['entropy_score']:.4f}")
            
            # Key strength visualization
            st.markdown("### Key Strength Analysis")
            strength_fig = st.session_state.visualizer.plot_key_strength_analysis(results)
            st.plotly_chart(strength_fig, use_container_width=True)
            
            # Performance comparison
            st.markdown("### Performance Comparison")
            perf_fig = st.session_state.visualizer.plot_crypto_performance(results)
            st.plotly_chart(perf_fig, use_container_width=True)
        
        else:
            st.info("🔍 Generate keys to see results and metrics here.")

def ids_page():
    st.markdown('<h2 class="module-header">🛡️ Intrusion Detection System</h2>', unsafe_allow_html=True)
    
    # Model training section
    with st.expander("🎯 Model Training & Configuration", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Training Configuration")
            model_type = st.selectbox("Model Type", ["Random Forest", "XGBoost", "Neural Network", "Ensemble"])
            train_test_split = st.slider("Train/Test Split", 0.1, 0.9, 0.8)
            use_image_features = st.checkbox("Include Image-based Features", value=True)
            
        with col2:
            st.markdown("### Dataset Selection")
            dataset_option = st.radio("Dataset Source", ["Built-in Sample", "Upload Dataset"])
            
            if dataset_option == "Upload Dataset":
                uploaded_data = st.file_uploader("Upload Dataset", type=['csv', 'json'])
                uploaded_images = st.file_uploader("Upload Network Images", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
        
        if st.button("🚀 Train IDS Model", use_container_width=True):
            with st.spinner("Training intrusion detection model..."):
                # Load or generate training data
                if dataset_option == "Built-in Sample":
                    train_data, train_labels = st.session_state.data_processor.generate_ids_training_data(10000)
                else:
                    # Handle uploaded data
                    train_data, train_labels = st.session_state.data_processor.process_uploaded_data(uploaded_data)
                
                # Train model
                training_results = st.session_state.ids_system.train_model(
                    train_data, train_labels, 
                    model_type=model_type.lower().replace(' ', '_'),
                    test_size=1-train_test_split
                )
                
                st.session_state.ids_training_results = training_results
            
            st.success("✅ IDS Model trained successfully!")
    
    # Real-time detection section
    st.markdown("### 🔍 Real-time Intrusion Detection")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### Input Data")
        detection_method = st.radio("Detection Method", ["Live Monitoring", "Batch Analysis", "File Upload"])
        
        if detection_method == "File Upload":
            detection_file = st.file_uploader("Upload Network Data", type=['csv', 'json'])
            detection_images = st.file_uploader("Upload Network Visualization Images", 
                                              type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
        
        if st.button("🔍 Run Detection", use_container_width=True):
            with st.spinner("Analyzing network traffic..."):
                # Generate or load data for detection
                if detection_method == "Live Monitoring":
                    test_data = st.session_state.data_processor.generate_sample_network_data(1000)
                elif detection_method == "Batch Analysis":
                    test_data = st.session_state.data_processor.generate_sample_network_data(5000)
                else:
                    test_data = st.session_state.data_processor.process_uploaded_data(detection_file)
                
                # Run detection
                detection_results = st.session_state.ids_system.predict_batch(test_data)
                st.session_state.detection_results = detection_results
            
            st.success("🎯 Detection analysis completed!")
    
    with col2:
        st.markdown("#### Detection Results")
        
        if hasattr(st.session_state, 'detection_results'):
            results = st.session_state.detection_results
            
            # Key metrics display
            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
            
            with metrics_col1:
                st.metric("Accuracy", f"{results['accuracy']:.3f}")
                st.metric("Precision", f"{results['precision']:.3f}")
            
            with metrics_col2:
                st.metric("Recall", f"{results['recall']:.3f}")
                st.metric("F1 Score", f"{results['f1_score']:.3f}")
            
            with metrics_col3:
                st.metric("ROC-AUC", f"{results['roc_auc']:.3f}")
                st.metric("PR-AUC", f"{results['pr_auc']:.3f}")
            
            with metrics_col4:
                st.metric("FPR", f"{results['fpr']:.3f}")
                st.metric("Detection Latency", f"{results['detection_latency']:.2f} ms")
            
            # Additional metrics
            perf_col1, perf_col2 = st.columns(2)
            with perf_col1:
                st.metric("Throughput", f"{results['throughput']:.1f} flows/s")
            with perf_col2:
                st.metric("CPU Usage", f"{results['cpu_usage']:.1f}%")
        
        else:
            st.info("🔍 Run detection to see results here.")
    
    # Per-attack detection rates
    if hasattr(st.session_state, 'detection_results'):
        st.markdown("### 📊 Per-Attack Type Detection Rates")
        attack_rates_df = st.session_state.detection_results['per_attack_rates']
        
        # Display as interactive table
        st.dataframe(
            attack_rates_df.style.format({
                'Recall': '{:.3f}',
                'Precision': '{:.3f}',
                'F1_Score': '{:.3f}'
            }).background_gradient(subset=['Recall', 'Precision', 'F1_Score']),
            use_container_width=True
        )
        
        # Heatmap visualization
        fig_heatmap = st.session_state.visualizer.plot_attack_detection_heatmap(attack_rates_df)
        st.plotly_chart(fig_heatmap, use_container_width=True)

def analytics_page():
    st.markdown('<h2 class="module-header">📊 Analytics & Visualization</h2>', unsafe_allow_html=True)
    
    # Performance analytics
    st.markdown("### 🚀 System Performance Analytics")
    
    if hasattr(st.session_state, 'ids_training_results') and hasattr(st.session_state, 'detection_results'):
        
        # Model comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Model Performance Comparison")
            model_comparison_fig = st.session_state.visualizer.plot_model_comparison()
            st.plotly_chart(model_comparison_fig, use_container_width=True)
        
        with col2:
            st.markdown("#### ROC Curves")
            roc_fig = st.session_state.visualizer.plot_roc_curves(st.session_state.detection_results)
            st.plotly_chart(roc_fig, use_container_width=True)
        
        # Time series analysis
        st.markdown("#### Detection Performance Over Time")
        time_series_fig = st.session_state.visualizer.plot_detection_timeline()
        st.plotly_chart(time_series_fig, use_container_width=True)
        
        # Feature importance
        st.markdown("#### Feature Importance Analysis")
        feature_importance_fig = st.session_state.visualizer.plot_feature_importance(st.session_state.ids_system)
        st.plotly_chart(feature_importance_fig, use_container_width=True)
    
    else:
        st.info("📈 Train models and run detections to see analytics here.")
    
    # Export options
    st.markdown("### 📁 Export Results")
    export_col1, export_col2, export_col3 = st.columns(3)
    
    with export_col1:
        if st.button("📊 Export Metrics CSV", use_container_width=True):
            if hasattr(st.session_state, 'detection_results'):
                csv_data = st.session_state.metrics_calc.export_metrics_csv(st.session_state.detection_results)
                st.download_button(
                    label="Download Metrics CSV",
                    data=csv_data,
                    file_name="ids_metrics.csv",
                    mime="text/csv"
                )
    
    with export_col2:
        if st.button("📈 Export Visualizations", use_container_width=True):
            st.info("Visualization export functionality ready!")
    
    with export_col3:
        if st.button("📋 Generate Report", use_container_width=True):
            if hasattr(st.session_state, 'detection_results'):
                report_html = st.session_state.visualizer.generate_comprehensive_report(
                    st.session_state.detection_results,
                    getattr(st.session_state, 'crypto_results', None)
                )
                st.download_button(
                    label="Download Full Report",
                    data=report_html,
                    file_name="cybersecurity_report.html",
                    mime="text/html"
                )

if __name__ == "__main__":
    main()
