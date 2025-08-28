import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import io
import base64
import random
import math

class Visualizer:
    """
    Advanced visualization module for cybersecurity analytics and reporting.
    """
    
    def __init__(self):
        self.color_palette = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff9800',
            'info': '#17a2b8',
            'light': '#f8f9fa',
            'dark': '#343a40'
        }
        
    def plot_key_strength_analysis(self, crypto_results):
        """Plot cryptographic key strength analysis."""
        if 'keys' not in crypto_results:
            return self._create_empty_plot("No key data available")
        
        keys = crypto_results['keys']
        
        # Extract metrics
        entropies = [key.get('entropy', 0) for key in keys]
        attack_rates = [key.get('attack_success_rate', 0) for key in keys]
        key_lengths = [key.get('key_length_bits', 256) for key in keys]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Entropy Distribution', 'Attack Success Rate', 
                          'Key Length vs Entropy', 'Randomness Score'),
            specs=[[{"type": "histogram"}, {"type": "histogram"}],
                   [{"type": "scatter"}, {"type": "box"}]]
        )
        
        # Entropy histogram
        fig.add_trace(
            go.Histogram(x=entropies, nbinsx=20, name='Entropy', 
                        marker_color=self.color_palette['primary']),
            row=1, col=1
        )
        
        # Attack success rate histogram
        fig.add_trace(
            go.Histogram(x=attack_rates, nbinsx=20, name='Attack Success Rate', 
                        marker_color=self.color_palette['danger']),
            row=1, col=2
        )
        
        # Key length vs entropy scatter
        fig.add_trace(
            go.Scatter(x=key_lengths, y=entropies, mode='markers',
                      name='Length vs Entropy', 
                      marker=dict(color=self.color_palette['success'], size=8)),
            row=2, col=1
        )
        
        # Randomness score box plot
        randomness_scores = [key.get('randomness_score', 0.5) for key in keys]
        fig.add_trace(
            go.Box(y=randomness_scores, name='Randomness',
                   marker_color=self.color_palette['warning']),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Cryptographic Key Strength Analysis",
            showlegend=False,
            height=600
        )
        
        return fig
    
    def plot_crypto_performance(self, crypto_results):
        """Plot cryptographic performance metrics."""
        if 'keys' not in crypto_results:
            return self._create_empty_plot("No performance data available")
        
        keys = crypto_results['keys']
        
        # Extract performance metrics
        gen_times = [key.get('generation_time', 0) for key in keys]
        enc_times = [key.get('encoding_time', 0) for key in keys]
        dec_times = [key.get('decoding_time', 0) for key in keys]
        
        fig = go.Figure()
        
        # Performance comparison
        fig.add_trace(go.Bar(
            name='Generation Time',
            x=list(range(len(gen_times))),
            y=gen_times,
            marker_color=self.color_palette['primary']
        ))
        
        fig.add_trace(go.Bar(
            name='Encoding Time',
            x=list(range(len(enc_times))),
            y=enc_times,
            marker_color=self.color_palette['success']
        ))
        
        fig.add_trace(go.Bar(
            name='Decoding Time',
            x=list(range(len(dec_times))),
            y=dec_times,
            marker_color=self.color_palette['warning']
        ))
        
        fig.update_layout(
            title='Cryptographic Performance Metrics',
            xaxis_title='Key Index',
            yaxis_title='Time (ms)',
            barmode='group',
            height=400
        )
        
        return fig
    
    def plot_attack_detection_heatmap(self, attack_rates_df):
        """Plot heatmap of per-attack detection rates."""
        if attack_rates_df.empty:
            return self._create_empty_plot("No attack rate data available")
        
        # Prepare data for heatmap
        metrics = ['Recall', 'Precision', 'F1_Score']
        heatmap_data = attack_rates_df[metrics].values
        
        # Round values using pure Python
        rounded_data = [[round(val, 3) for val in row] for row in heatmap_data]
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=metrics,
            y=attack_rates_df['Attack_Type'].values,
            colorscale='RdYlBu_r',
            text=rounded_data,
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Attack Detection Performance Heatmap',
            xaxis_title='Metrics',
            yaxis_title='Attack Types',
            height=500
        )
        
        return fig
    
    def plot_model_comparison(self):
        """Plot comparison of different ML models."""
        # Generate sample model comparison data
        models = ['Random Forest', 'XGBoost', 'LightGBM', 'Neural Network', 'Ensemble']
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        
        # Simulate realistic performance data
        random.seed(42)
        performance_data = {}
        for model in models:
            base_performance = [random.uniform(0.85, 0.98) for _ in range(len(metrics))]
            # Ensemble should perform slightly better
            if model == 'Ensemble':
                base_performance = [p + 0.02 for p in base_performance]
            performance_data[model] = base_performance
        
        fig = go.Figure()
        
        for i, metric in enumerate(metrics):
            fig.add_trace(go.Scatter(
                x=models,
                y=[performance_data[model][i] for model in models],
                mode='lines+markers',
                name=metric,
                line=dict(width=3),
                marker=dict(size=8)
            ))
        
        fig.update_layout(
            title='Model Performance Comparison',
            xaxis_title='Models',
            yaxis_title='Performance Score',
            yaxis=dict(range=[0.8, 1.0]),
            height=400,
            hovermode='x unified'
        )
        
        return fig
    
    def plot_roc_curves(self, detection_results):
        """Plot ROC curves for different models."""
        fig = go.Figure()
        
        # Generate sample ROC curves
        models = ['Random Forest', 'XGBoost', 'Neural Network', 'Ensemble']
        colors = [self.color_palette['primary'], self.color_palette['success'], 
                 self.color_palette['warning'], self.color_palette['danger']]
        
        for i, model in enumerate(models):
            # Generate sample ROC curve data
            fpr = [j / 99 for j in range(100)]  # linspace equivalent
            # Different models have different shaped curves
            if model == 'Ensemble':
                tpr = [1 - (1 - f) ** 2.5 for f in fpr]  # Best performing
            elif model == 'Neural Network':
                tpr = [1 - (1 - f) ** 2.2 for f in fpr]
            elif model == 'Random Forest':
                tpr = [1 - (1 - f) ** 2.0 for f in fpr]
            else:
                tpr = [1 - (1 - f) ** 1.8 for f in fpr]
            
            # Add some noise
            tpr = [t + random.gauss(0, 0.02) for t in tpr]
            tpr = [max(0, min(1, t)) for t in tpr]  # clip equivalent
            
            # Simple trapezoidal integration
            auc_score = sum((tpr[i] + tpr[i-1]) * (fpr[i] - fpr[i-1]) / 2 for i in range(1, len(fpr)))
            
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'{model} (AUC = {auc_score:.3f})',
                line=dict(color=colors[i], width=2)
            ))
        
        # Add diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='black', dash='dash', width=1)
        ))
        
        fig.update_layout(
            title='ROC Curves - Model Comparison',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=500,
            showlegend=True
        )
        
        return fig
    
    def plot_detection_timeline(self):
        """Plot detection performance over time."""
        # Generate sample timeline data - 31 days * 24 hours = 744 hours
        start_date = datetime(2024, 1, 1)
        dates = [start_date + timedelta(hours=i) for i in range(744)]
        
        # Simulate detection metrics over time
        random.seed(42)
        base_accuracy = 0.95
        accuracy = [max(0.85, min(1.0, base_accuracy + random.gauss(0, 0.02))) for _ in range(len(dates))]
        
        base_throughput = 1000
        throughput = [max(100, base_throughput + random.gauss(0, 100)) for _ in range(len(dates))]
        
        base_latency = 2.5
        latency = [max(0.1, base_latency + random.gauss(0, 0.5)) for _ in range(len(dates))]
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Accuracy Over Time', 'Throughput Over Time', 'Detection Latency'),
            vertical_spacing=0.08
        )
        
        # Accuracy timeline
        fig.add_trace(
            go.Scatter(x=dates, y=accuracy, mode='lines', name='Accuracy',
                      line=dict(color=self.color_palette['success'], width=2)),
            row=1, col=1
        )
        
        # Throughput timeline
        fig.add_trace(
            go.Scatter(x=dates, y=throughput, mode='lines', name='Throughput',
                      line=dict(color=self.color_palette['primary'], width=2)),
            row=2, col=1
        )
        
        # Latency timeline
        fig.add_trace(
            go.Scatter(x=dates, y=latency, mode='lines', name='Latency',
                      line=dict(color=self.color_palette['warning'], width=2)),
            row=3, col=1
        )
        
        fig.update_layout(
            title='IDS Performance Timeline',
            height=800,
            showlegend=False
        )
        
        fig.update_xaxes(title_text="Time", row=3, col=1)
        fig.update_yaxes(title_text="Accuracy", row=1, col=1)
        fig.update_yaxes(title_text="Flows/sec", row=2, col=1)
        fig.update_yaxes(title_text="Latency (ms)", row=3, col=1)
        
        return fig
    
    def plot_feature_importance(self, ids_system):
        """Plot feature importance from IDS models."""
        if not hasattr(ids_system, 'models') or not ids_system.models:
            return self._create_empty_plot("No trained models available")
        
        # Get feature importance from available models
        importance_data = []
        
        for model_name in ['random_forest', 'xgboost', 'lightgbm']:
            if model_name in ids_system.models:
                feature_importance = ids_system.get_feature_importance(model_name)
                if not feature_importance.empty:
                    importance_data.append({
                        'model': model_name,
                        'features': feature_importance['feature'].values[:10],
                        'importance': feature_importance['importance'].values[:10]
                    })
                    break
        
        if not importance_data:
            # Generate sample feature importance
            features = [f'feature_{i}' for i in range(10)]
            # Exponential distribution approximation
            importance = [-0.1 * math.log(random.random()) for _ in range(10)]
            total_importance = sum(importance)
            importance = [i / total_importance for i in importance]  # normalize
            importance_data = [{
                'model': 'sample',
                'features': features,
                'importance': importance
            }]
        
        fig = go.Figure()
        
        model_data = importance_data[0]
        fig.add_trace(go.Bar(
            x=model_data['importance'],
            y=model_data['features'],
            orientation='h',
            marker_color=self.color_palette['primary']
        ))
        
        fig.update_layout(
            title='Top 10 Most Important Features',
            xaxis_title='Feature Importance',
            yaxis_title='Features',
            height=500,
            yaxis={'categoryorder':'total ascending'}
        )
        
        return fig
    
    def plot_threat_distribution(self, detection_results):
        """Plot distribution of detected threats."""
        if 'per_attack_rates' not in detection_results:
            return self._create_empty_plot("No threat data available")
        
        attack_rates_df = detection_results['per_attack_rates']
        
        # Create pie chart of attack types (simulated distribution)
        # Approximate multinomial distribution
        probs = [0.6, 0.1, 0.08, 0.06, 0.04, 0.05, 0.04, 0.03]
        attack_counts = [int(1000 * p) for p in probs]
        # Adjust for rounding
        attack_counts[0] += 1000 - sum(attack_counts)
        
        fig = go.Figure(data=[go.Pie(
            labels=attack_rates_df['Attack_Type'].values,
            values=attack_counts,
            hole=0.3,
            textinfo='label+percent',
            textposition='outside'
        )])
        
        fig.update_layout(
            title='Threat Distribution',
            height=500,
            showlegend=True
        )
        
        return fig
    
    def plot_confusion_matrix(self, y_true, y_pred, labels=None):
        """Plot confusion matrix heatmap using pure Python implementation."""
        # Pure Python confusion matrix implementation
        if labels is None:
            labels = ['Normal', 'Attack']
        
        # Create confusion matrix manually
        cm = [[0 for _ in range(len(labels))] for _ in range(len(labels))]
        
        for true_val, pred_val in zip(y_true, y_pred):
            # Ensure values are within bounds
            true_idx = min(max(int(true_val), 0), len(labels) - 1)
            pred_idx = min(max(int(pred_val), 0), len(labels) - 1)
            cm[true_idx][pred_idx] += 1
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale='Blues',
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 16},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Confusion Matrix',
            xaxis_title='Predicted',
            yaxis_title='Actual',
            height=400
        )
        
        return fig
    
    def plot_system_metrics_dashboard(self, system_metrics):
        """Create comprehensive system metrics dashboard."""
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('CPU Usage', 'Memory Usage', 'Network Throughput',
                          'Detection Latency', 'Accuracy Trend', 'Alert Volume'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "scatter"}, {"type": "bar"}]]
        )
        
        # Generate sample system metrics - 100 timestamps at 5-minute intervals
        start_time = datetime(2024, 1, 1)
        timestamps = [start_time + timedelta(minutes=i*5) for i in range(100)]
        
        # CPU Usage
        cpu_usage = [max(0, min(100, 20 + random.gauss(0, 5))) for _ in range(100)]
        fig.add_trace(
            go.Scatter(x=timestamps, y=cpu_usage, mode='lines', 
                      line=dict(color=self.color_palette['primary'])),
            row=1, col=1
        )
        
        # Memory Usage
        memory_usage = [max(0, min(100, 60 + random.gauss(0, 8))) for _ in range(100)]
        fig.add_trace(
            go.Scatter(x=timestamps, y=memory_usage, mode='lines',
                      line=dict(color=self.color_palette['success'])),
            row=1, col=2
        )
        
        # Network Throughput
        throughput = [max(0, 1000 + random.gauss(0, 200)) for _ in range(100)]
        fig.add_trace(
            go.Scatter(x=timestamps, y=throughput, mode='lines',
                      line=dict(color=self.color_palette['warning'])),
            row=1, col=3
        )
        
        # Detection Latency
        latency = [max(0, 2.5 + random.gauss(0, 0.5)) for _ in range(100)]
        fig.add_trace(
            go.Scatter(x=timestamps, y=latency, mode='lines',
                      line=dict(color=self.color_palette['danger'])),
            row=2, col=1
        )
        
        # Accuracy Trend
        accuracy = [max(0, min(1, 0.95 + random.gauss(0, 0.02))) for _ in range(100)]
        fig.add_trace(
            go.Scatter(x=timestamps, y=accuracy, mode='lines',
                      line=dict(color=self.color_palette['info'])),
            row=2, col=2
        )
        
        # Alert Volume
        alert_hours = [f"{i:02d}:00" for i in range(24)]
        # Approximate Poisson distribution
        alert_counts = [max(0, int(10 + random.gauss(0, math.sqrt(10)))) for _ in range(24)]
        fig.add_trace(
            go.Bar(x=alert_hours, y=alert_counts,
                   marker_color=self.color_palette['secondary']),
            row=2, col=3
        )
        
        fig.update_layout(
            title='System Performance Dashboard',
            height=800,
            showlegend=False
        )
        
        return fig
    
    def generate_comprehensive_report(self, detection_results, crypto_results=None):
        """Generate comprehensive HTML report."""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Cybersecurity Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ text-align: center; color: #1f77b4; margin-bottom: 30px; }}
                .section {{ margin-bottom: 30px; }}
                .metrics {{ display: flex; justify-content: space-around; margin: 20px 0; }}
                .metric {{ text-align: center; padding: 15px; background: #f8f9fa; border-radius: 8px; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #28a745; }}
                .metric-label {{ color: #6c757d; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .alert {{ padding: 15px; margin: 10px 0; border-radius: 4px; }}
                .alert-success {{ background-color: #d4edda; color: #155724; }}
                .alert-warning {{ background-color: #fff3cd; color: #856404; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🔐 ML-Enhanced Cybersecurity Analysis Report</h1>
                <p>Generated on {timestamp}</p>
            </div>
            
            <div class="section">
                <h2>📊 Detection Performance Summary</h2>
                <div class="metrics">
                    <div class="metric">
                        <div class="metric-value">{accuracy:.2%}</div>
                        <div class="metric-label">Accuracy</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{precision:.2%}</div>
                        <div class="metric-label">Precision</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{recall:.2%}</div>
                        <div class="metric-label">Recall</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{f1_score:.2%}</div>
                        <div class="metric-label">F1-Score</div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>🎯 Threat Detection Results</h2>
                <div class="alert alert-{alert_type}">
                    <strong>{threat_status}</strong> - {threat_count} threats detected out of {total_samples} samples
                </div>
                
                <h3>Per-Attack Type Performance</h3>
                {attack_table}
            </div>
            
            <div class="section">
                <h2>⚡ System Performance</h2>
                <div class="metrics">
                    <div class="metric">
                        <div class="metric-value">{throughput:.0f}</div>
                        <div class="metric-label">Flows/sec</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{latency:.1f}</div>
                        <div class="metric-label">Latency (ms)</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{cpu_usage:.1f}%</div>
                        <div class="metric-label">CPU Usage</div>
                    </div>
                </div>
            </div>
            
            {crypto_section}
            
            <div class="section">
                <h2>📈 Recommendations</h2>
                <ul>
                    {recommendations}
                </ul>
            </div>
        </body>
        </html>
        """
        
        # Extract metrics
        accuracy = detection_results.get('accuracy', 0)
        precision = detection_results.get('precision', 0)
        recall = detection_results.get('recall', 0)
        f1_score = detection_results.get('f1_score', 0)
        threat_count = detection_results.get('threat_count', 0)
        total_samples = detection_results.get('total_samples', 0)
        throughput = detection_results.get('throughput', 0)
        latency = detection_results.get('detection_latency', 0)
        cpu_usage = detection_results.get('cpu_usage', 0)
        
        # Determine alert type
        if threat_count > total_samples * 0.1:
            alert_type = "warning"
            threat_status = "High Alert"
        else:
            alert_type = "success"
            threat_status = "Normal Operation"
        
        # Generate attack performance table
        if 'per_attack_rates' in detection_results:
            attack_df = detection_results['per_attack_rates']
            attack_table = attack_df.to_html(classes='table', index=False, float_format='{:.3f}'.format)
        else:
            attack_table = "<p>No per-attack data available</p>"
        
        # Generate crypto section if available
        crypto_section = ""
        if crypto_results:
            crypto_section = f"""
            <div class="section">
                <h2>🔑 Cryptographic Key Generation</h2>
                <div class="metrics">
                    <div class="metric">
                        <div class="metric-value">{crypto_results.get('encoding_time', 0):.1f}</div>
                        <div class="metric-label">Encoding Time (ms)</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{crypto_results.get('entropy_score', 0):.3f}</div>
                        <div class="metric-label">Entropy Score</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{crypto_results.get('attack_success_rate', 0):.2f}%</div>
                        <div class="metric-label">Attack Success Rate</div>
                    </div>
                </div>
            </div>
            """
        
        # Generate recommendations
        recommendations = []
        if accuracy < 0.9:
            recommendations.append("Consider retraining models with more diverse data")
        if latency > 5.0:
            recommendations.append("Optimize detection algorithms for better performance")
        if cpu_usage > 80:
            recommendations.append("Scale system resources or optimize computational efficiency")
        if threat_count > total_samples * 0.05:
            recommendations.append("Increase monitoring and review security policies")
        
        recommendations_html = "".join([f"<li>{rec}</li>" for rec in recommendations])
        if not recommendations_html:
            recommendations_html = "<li>System is operating optimally</li>"
        
        # Format the HTML
        html_report = html_template.format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            alert_type=alert_type,
            threat_status=threat_status,
            threat_count=threat_count,
            total_samples=total_samples,
            attack_table=attack_table,
            throughput=throughput,
            latency=latency,
            cpu_usage=cpu_usage,
            crypto_section=crypto_section,
            recommendations=recommendations_html
        )
        
        return html_report
    
    def _create_empty_plot(self, message="No data available"):
        """Create an empty plot with a message."""
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5,
            text=message,
            showarrow=False,
            font=dict(size=16),
            xref="paper", yref="paper"
        )
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=400
        )
        return fig
