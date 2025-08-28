from datetime import datetime, timedelta
import random
import math
import json

class VisualizerFlask:
    """
    Visualization module for Flask application without numpy dependencies.
    Returns chart data as JSON for frontend rendering with Chart.js or similar.
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
    
    def get_key_strength_data(self, crypto_results):
        """Get cryptographic key strength data for charts."""
        if 'keys' not in crypto_results:
            return {'error': 'No key data available'}
        
        keys = crypto_results['keys']
        
        # Extract metrics
        entropies = [key.get('entropy', 0) for key in keys]
        attack_rates = [key.get('attack_success_rate', 0) for key in keys]
        key_lengths = [key.get('key_length_bits', 256) for key in keys]
        randomness_scores = [key.get('randomness_score', 0.5) for key in keys]
        
        return {
            'entropy_distribution': {
                'type': 'histogram',
                'data': entropies,
                'title': 'Entropy Distribution',
                'xlabel': 'Entropy',
                'ylabel': 'Frequency',
                'color': self.color_palette['primary']
            },
            'attack_rates': {
                'type': 'histogram',
                'data': attack_rates,
                'title': 'Attack Success Rate',
                'xlabel': 'Attack Success Rate (%)',
                'ylabel': 'Frequency',
                'color': self.color_palette['danger']
            },
            'length_vs_entropy': {
                'type': 'scatter',
                'x_data': key_lengths,
                'y_data': entropies,
                'title': 'Key Length vs Entropy',
                'xlabel': 'Key Length (bits)',
                'ylabel': 'Entropy',
                'color': self.color_palette['success']
            },
            'randomness_box': {
                'type': 'box',
                'data': randomness_scores,
                'title': 'Randomness Score Distribution',
                'ylabel': 'Randomness Score',
                'color': self.color_palette['warning']
            }
        }
    
    def get_crypto_performance_data(self, crypto_results):
        """Get cryptographic performance data for charts."""
        if 'keys' not in crypto_results:
            return {'error': 'No performance data available'}
        
        keys = crypto_results['keys']
        
        # Extract performance metrics
        gen_times = [key.get('generation_time', 0) for key in keys]
        enc_times = [key.get('encoding_time', 0) for key in keys]
        dec_times = [key.get('decoding_time', 0) for key in keys]
        
        return {
            'type': 'bar',
            'title': 'Cryptographic Performance Metrics',
            'datasets': [
                {
                    'label': 'Generation Time',
                    'data': gen_times,
                    'backgroundColor': self.color_palette['primary']
                },
                {
                    'label': 'Encoding Time',
                    'data': enc_times,
                    'backgroundColor': self.color_palette['success']
                },
                {
                    'label': 'Decoding Time',
                    'data': dec_times,
                    'backgroundColor': self.color_palette['warning']
                }
            ],
            'labels': [f'Key {i+1}' for i in range(len(gen_times))],
            'xlabel': 'Key Index',
            'ylabel': 'Time (ms)'
        }
    
    def get_attack_heatmap_data(self, detection_results):
        """Get attack detection heatmap data."""
        if 'per_attack_rates' not in detection_results:
            return {'error': 'No attack rate data available'}
        
        attack_rates_data = detection_results['per_attack_rates']
        
        # Prepare data for heatmap
        attack_types = [item['Attack_Type'] for item in attack_rates_data]
        metrics = ['Recall', 'Precision', 'F1_Score']
        
        heatmap_data = []
        for metric in metrics:
            metric_values = [item[metric] for item in attack_rates_data]
            heatmap_data.append(metric_values)
        
        return {
            'type': 'heatmap',
            'title': 'Attack Detection Performance Heatmap',
            'data': heatmap_data,
            'x_labels': attack_types,
            'y_labels': metrics,
            'colormap': 'RdYlBu_r'
        }
    
    def get_model_comparison_data(self):
        """Get model performance comparison data."""
        models = ['Random Forest', 'XGBoost', 'LightGBM', 'Neural Network', 'Ensemble']
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        
        # Simulate realistic performance data
        random.seed(42)
        performance_data = {}
        for model in models:
            base_performance = [random.uniform(0.85, 0.98) for _ in range(len(metrics))]
            # Ensemble should perform slightly better
            if model == 'Ensemble':
                base_performance = [min(1.0, p + 0.02) for p in base_performance]
            performance_data[model] = base_performance
        
        # Restructure for chart
        datasets = []
        colors = [self.color_palette['primary'], self.color_palette['success'], 
                 self.color_palette['warning'], self.color_palette['danger'], 
                 self.color_palette['info']]
        
        for i, metric in enumerate(metrics):
            datasets.append({
                'label': metric,
                'data': [performance_data[model][i] for model in models],
                'borderColor': colors[i % len(colors)],
                'backgroundColor': colors[i % len(colors)] + '33',  # Add transparency
                'fill': False
            })
        
        return {
            'type': 'line',
            'title': 'Model Performance Comparison',
            'datasets': datasets,
            'labels': models,
            'xlabel': 'Models',
            'ylabel': 'Performance Score'
        }
    
    def get_roc_curves_data(self):
        """Get ROC curves data."""
        models = ['Random Forest', 'XGBoost', 'Neural Network', 'Ensemble']
        colors = [self.color_palette['primary'], self.color_palette['success'], 
                 self.color_palette['warning'], self.color_palette['danger']]
        
        datasets = []
        
        for i, model in enumerate(models):
            # Generate sample ROC curve data
            fpr = [j / 99 for j in range(100)]  # 0 to 1 in steps of 0.01
            
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
            tpr = [min(1.0, max(0.0, t + random.gauss(0, 0.02))) for t in tpr]
            
            # Simple trapezoidal integration for AUC
            auc_score = sum((tpr[i] + tpr[i-1]) * (fpr[i] - fpr[i-1]) / 2 for i in range(1, len(fpr)))
            
            datasets.append({
                'label': f'{model} (AUC = {auc_score:.3f})',
                'data': [{'x': fpr[j], 'y': tpr[j]} for j in range(len(fpr))],
                'borderColor': colors[i],
                'backgroundColor': 'transparent',
                'fill': False
            })
        
        # Add diagonal line
        datasets.append({
            'label': 'Random Classifier',
            'data': [{'x': 0, 'y': 0}, {'x': 1, 'y': 1}],
            'borderColor': 'black',
            'borderDash': [5, 5],
            'backgroundColor': 'transparent',
            'fill': False
        })
        
        return {
            'type': 'scatter',
            'title': 'ROC Curves - Model Comparison',
            'datasets': datasets,
            'xlabel': 'False Positive Rate',
            'ylabel': 'True Positive Rate'
        }
    
    def get_detection_timeline_data(self):
        """Get detection performance timeline data."""
        # Generate sample timeline data - 31 days * 24 hours = 744 hours
        start_date = datetime(2024, 1, 1)
        dates = [(start_date + timedelta(hours=i)).isoformat() for i in range(744)]
        
        # Simulate detection metrics over time
        random.seed(42)
        base_accuracy = 0.95
        accuracy = [max(0.85, min(1.0, base_accuracy + random.gauss(0, 0.02))) for _ in range(len(dates))]
        
        base_throughput = 1000
        throughput = [max(100, base_throughput + random.gauss(0, 100)) for _ in range(len(dates))]
        
        base_latency = 2.5
        latency = [max(0.1, base_latency + random.gauss(0, 0.5)) for _ in range(len(dates))]
        
        return {
            'type': 'line',
            'title': 'IDS Performance Timeline',
            'datasets': [
                {
                    'label': 'Accuracy',
                    'data': accuracy,
                    'borderColor': self.color_palette['success'],
                    'backgroundColor': 'transparent',
                    'yAxisID': 'y1'
                },
                {
                    'label': 'Throughput (flows/s)',
                    'data': throughput,
                    'borderColor': self.color_palette['primary'],
                    'backgroundColor': 'transparent',
                    'yAxisID': 'y2'
                },
                {
                    'label': 'Latency (ms)',
                    'data': latency,
                    'borderColor': self.color_palette['warning'],
                    'backgroundColor': 'transparent',
                    'yAxisID': 'y3'
                }
            ],
            'labels': dates,
            'xlabel': 'Time'
        }
    
    def get_feature_importance_data(self):
        """Get feature importance data."""
        # Generate sample feature importance
        features = ['packet_size', 'flow_duration', 'bytes_sent', 'packets_per_second',
                   'port_number', 'protocol_type', 'entropy', 'connection_state',
                   'tcp_flags', 'payload_entropy']
        
        # Exponential distribution approximation
        random.seed(42)
        importance = [-0.1 * math.log(random.random()) for _ in range(len(features))]
        total_importance = sum(importance)
        importance = [i / total_importance for i in importance]  # normalize
        
        # Sort by importance
        feature_importance = list(zip(features, importance))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        features, importance = zip(*feature_importance)
        
        return {
            'type': 'horizontalBar',
            'title': 'Top 10 Most Important Features',
            'datasets': [{
                'label': 'Importance',
                'data': list(importance),
                'backgroundColor': self.color_palette['primary']
            }],
            'labels': list(features),
            'xlabel': 'Feature Importance',
            'ylabel': 'Features'
        }
    
    def get_threat_distribution_data(self, detection_results):
        """Get threat distribution data."""
        if 'per_attack_rates' not in detection_results:
            return {'error': 'No threat data available'}
        
        attack_rates_data = detection_results['per_attack_rates']
        attack_types = [item['Attack_Type'] for item in attack_rates_data]
        
        # Simulate distribution (approximating multinomial)
        probs = [0.6, 0.1, 0.08, 0.06, 0.04, 0.05, 0.04, 0.03]
        if len(probs) < len(attack_types):
            # Extend with uniform probability for additional types
            remaining_prob = 1.0 - sum(probs)
            additional_types = len(attack_types) - len(probs)
            probs.extend([remaining_prob / additional_types] * additional_types)
        
        attack_counts = [int(1000 * p) for p in probs[:len(attack_types)]]
        # Adjust for rounding
        if attack_counts:
            attack_counts[0] += 1000 - sum(attack_counts)
        
        return {
            'type': 'doughnut',
            'title': 'Threat Distribution',
            'datasets': [{
                'data': attack_counts,
                'backgroundColor': [
                    self.color_palette['primary'],
                    self.color_palette['danger'],
                    self.color_palette['warning'],
                    self.color_palette['success'],
                    self.color_palette['info'],
                    self.color_palette['secondary'],
                    '#6f42c1',
                    '#e83e8c'
                ][:len(attack_types)]
            }],
            'labels': attack_types
        }
    
    def get_confusion_matrix_data(self, y_true, y_pred, labels=None):
        """Get confusion matrix data."""
        if labels is None:
            labels = ['Normal', 'Attack']
        
        # Create confusion matrix manually
        cm = [[0 for _ in range(len(labels))] for _ in range(len(labels))]
        
        for true_val, pred_val in zip(y_true, y_pred):
            true_idx = min(max(int(true_val), 0), len(labels) - 1)
            pred_idx = min(max(int(pred_val), 0), len(labels) - 1)
            cm[true_idx][pred_idx] += 1
        
        return {
            'type': 'heatmap',
            'title': 'Confusion Matrix',
            'data': cm,
            'x_labels': labels,
            'y_labels': labels,
            'xlabel': 'Predicted',
            'ylabel': 'Actual'
        }
    
    def get_system_dashboard_data(self):
        """Get system metrics dashboard data."""
        # Generate sample system metrics - 100 timestamps
        start_time = datetime(2024, 1, 1)
        timestamps = [(start_time + timedelta(minutes=i*5)).isoformat() for i in range(100)]
        
        # Generate metrics
        random.seed(42)
        cpu_usage = [max(0, min(100, 20 + random.gauss(0, 5))) for _ in range(100)]
        memory_usage = [max(0, min(100, 60 + random.gauss(0, 8))) for _ in range(100)]
        throughput = [max(0, 1000 + random.gauss(0, 200)) for _ in range(100)]
        latency = [max(0, 2.5 + random.gauss(0, 0.5)) for _ in range(100)]
        accuracy = [max(0, min(1, 0.95 + random.gauss(0, 0.02))) for _ in range(100)]
        
        # Alert volume (24 hours)
        alert_hours = [f"{i:02d}:00" for i in range(24)]
        alert_counts = [max(0, int(10 + random.gauss(0, math.sqrt(10)))) for _ in range(24)]
        
        return {
            'cpu_usage': {
                'type': 'line',
                'title': 'CPU Usage',
                'data': cpu_usage,
                'labels': timestamps,
                'color': self.color_palette['primary']
            },
            'memory_usage': {
                'type': 'line',
                'title': 'Memory Usage',
                'data': memory_usage,
                'labels': timestamps,
                'color': self.color_palette['success']
            },
            'throughput': {
                'type': 'line',
                'title': 'Network Throughput',
                'data': throughput,
                'labels': timestamps,
                'color': self.color_palette['warning']
            },
            'latency': {
                'type': 'line',
                'title': 'Detection Latency',
                'data': latency,
                'labels': timestamps,
                'color': self.color_palette['danger']
            },
            'accuracy': {
                'type': 'line',
                'title': 'Accuracy Trend',
                'data': accuracy,
                'labels': timestamps,
                'color': self.color_palette['info']
            },
            'alerts': {
                'type': 'bar',
                'title': 'Alert Volume (24h)',
                'data': alert_counts,
                'labels': alert_hours,
                'color': self.color_palette['secondary']
            }
        }
    
    def generate_comprehensive_report(self, detection_results, crypto_results=None):
        """Generate comprehensive HTML report."""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Cybersecurity Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { text-align: center; color: #1f77b4; margin-bottom: 30px; }
                .section { margin-bottom: 30px; }
                .metrics { display: flex; justify-content: space-around; margin: 20px 0; }
                .metric { text-align: center; padding: 15px; background: #f8f9fa; border-radius: 8px; }
                .metric-value { font-size: 24px; font-weight: bold; color: #28a745; }
                .metric-label { color: #6c757d; }
                table { width: 100%; border-collapse: collapse; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
                th { background-color: #f2f2f2; }
                .alert { padding: 15px; margin: 10px 0; border-radius: 4px; }
                .alert-success { background-color: #d4edda; color: #155724; }
                .alert-warning { background-color: #fff3cd; color: #856404; }
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
            attack_data = detection_results['per_attack_rates']
            attack_table = "<table><tr><th>Attack Type</th><th>Recall</th><th>Precision</th><th>F1-Score</th></tr>"
            for item in attack_data:
                attack_table += f"<tr><td>{item['Attack_Type']}</td><td>{item['Recall']:.3f}</td><td>{item['Precision']:.3f}</td><td>{item['F1_Score']:.3f}</td></tr>"
            attack_table += "</table>"
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
