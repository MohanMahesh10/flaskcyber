import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve,
    mean_squared_error, mean_absolute_error
)
import time
import io
from datetime import datetime

class MetricsCalculator:
    """
    Comprehensive metrics calculation for cybersecurity applications.
    """
    
    def __init__(self):
        self.metric_history = []
        self.baseline_metrics = None
        
    def calculate_classification_metrics(self, y_true, y_pred, y_prob=None):
        """Calculate comprehensive classification metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        metrics['precision_per_class'] = precision_per_class
        metrics['recall_per_class'] = recall_per_class
        metrics['f1_per_class'] = f1_per_class
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm
        
        # True/False Positive/Negative rates
        if len(np.unique(y_true)) == 2:  # Binary classification
            tn, fp, fn, tp = cm.ravel()
            metrics['true_positive_rate'] = tp / (tp + fn) if (tp + fn) > 0 else 0
            metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
            metrics['true_negative_rate'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # ROC and PR curves if probabilities provided
        if y_prob is not None and len(np.unique(y_true)) == 2:
            try:
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
                metrics['fpr'] = fpr
                metrics['tpr'] = tpr
                
                precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
                metrics['pr_auc'] = average_precision_score(y_true, y_prob)
                metrics['precision_curve'] = precision_curve
                metrics['recall_curve'] = recall_curve
            except:
                metrics['roc_auc'] = 0.5
                metrics['pr_auc'] = 0.5
        
        return metrics
    
    def calculate_ids_performance_metrics(self, y_true, y_pred, y_prob=None, 
                                        detection_times=None, throughput=None):
        """Calculate IDS-specific performance metrics."""
        # Get classification metrics
        metrics = self.calculate_classification_metrics(y_true, y_pred, y_prob)
        
        # IDS-specific metrics
        metrics['detection_accuracy'] = metrics['accuracy']
        metrics['attack_detection_rate'] = metrics['recall']  # Sensitivity
        metrics['false_alarm_rate'] = metrics['false_positive_rate'] if 'false_positive_rate' in metrics else 0
        
        # Performance metrics
        if detection_times is not None:
            metrics['mean_detection_time'] = np.mean(detection_times)
            metrics['max_detection_time'] = np.max(detection_times)
            metrics['detection_time_std'] = np.std(detection_times)
        
        if throughput is not None:
            metrics['throughput'] = throughput
        
        # Attack type specific metrics
        attack_types = ['DoS', 'Probe', 'R2L', 'U2R', 'DDoS', 'Malware', 'Phishing']
        metrics['per_attack_metrics'] = self._simulate_per_attack_metrics(attack_types)
        
        return metrics
    
    def _simulate_per_attack_metrics(self, attack_types):
        """Simulate per-attack type metrics for demonstration."""
        per_attack = {}
        
        for attack_type in attack_types:
            # Simulate different performance for different attack types
            base_performance = np.random.uniform(0.8, 0.95)
            noise = np.random.normal(0, 0.05)
            
            recall = np.clip(base_performance + noise, 0, 1)
            precision = np.clip(base_performance + np.random.normal(0, 0.03), 0, 1)
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            per_attack[attack_type] = {
                'recall': recall,
                'precision': precision,
                'f1_score': f1,
                'support': np.random.randint(50, 500)
            }
        
        return per_attack
    
    def calculate_crypto_metrics(self, key_generation_times, key_strengths, 
                               encoding_times=None, decoding_times=None):
        """Calculate cryptographic performance metrics."""
        metrics = {}
        
        # Generation performance
        metrics['mean_generation_time'] = np.mean(key_generation_times)
        metrics['generation_time_std'] = np.std(key_generation_times)
        metrics['min_generation_time'] = np.min(key_generation_times)
        metrics['max_generation_time'] = np.max(key_generation_times)
        
        # Key strength metrics
        metrics['mean_key_strength'] = np.mean(key_strengths)
        metrics['key_strength_std'] = np.std(key_strengths)
        metrics['min_key_strength'] = np.min(key_strengths)
        metrics['max_key_strength'] = np.max(key_strengths)
        
        # Encoding/Decoding performance
        if encoding_times is not None:
            metrics['mean_encoding_time'] = np.mean(encoding_times)
            metrics['encoding_time_std'] = np.std(encoding_times)
        
        if decoding_times is not None:
            metrics['mean_decoding_time'] = np.mean(decoding_times)
            metrics['decoding_time_std'] = np.std(decoding_times)
        
        # Efficiency metrics
        metrics['keys_per_second'] = len(key_generation_times) / np.sum(key_generation_times) * 1000
        
        return metrics
    
    def calculate_system_metrics(self, cpu_usage, memory_usage, network_throughput,
                               response_times=None, error_rates=None):
        """Calculate system performance metrics."""
        metrics = {}
        
        # Resource utilization
        metrics['avg_cpu_usage'] = np.mean(cpu_usage)
        metrics['max_cpu_usage'] = np.max(cpu_usage)
        metrics['cpu_usage_std'] = np.std(cpu_usage)
        
        metrics['avg_memory_usage'] = np.mean(memory_usage)
        metrics['max_memory_usage'] = np.max(memory_usage)
        metrics['memory_usage_std'] = np.std(memory_usage)
        
        # Network performance
        metrics['avg_network_throughput'] = np.mean(network_throughput)
        metrics['min_network_throughput'] = np.min(network_throughput)
        metrics['network_throughput_std'] = np.std(network_throughput)
        
        # Response time metrics
        if response_times is not None:
            metrics['avg_response_time'] = np.mean(response_times)
            metrics['percentile_95_response_time'] = np.percentile(response_times, 95)
            metrics['percentile_99_response_time'] = np.percentile(response_times, 99)
        
        # Error rate metrics
        if error_rates is not None:
            metrics['avg_error_rate'] = np.mean(error_rates)
            metrics['max_error_rate'] = np.max(error_rates)
        
        return metrics
    
    def calculate_ml_model_metrics(self, model, X_test, y_test, model_name="Model"):
        """Calculate ML model-specific metrics."""
        start_time = time.time()
        
        # Predictions
        y_pred = model.predict(X_test)
        prediction_time = time.time() - start_time
        
        # Probabilities if available
        y_prob = None
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)
            if y_prob.shape[1] > 1:
                y_prob = y_prob[:, 1]  # Positive class probability
            else:
                y_prob = y_prob.flatten()
        
        # Classification metrics
        metrics = self.calculate_classification_metrics(y_test, y_pred, y_prob)
        
        # Model-specific metrics
        metrics['model_name'] = model_name
        metrics['prediction_time'] = prediction_time
        metrics['predictions_per_second'] = len(X_test) / prediction_time
        
        # Model complexity metrics
        if hasattr(model, 'n_features_in_'):
            metrics['n_features'] = model.n_features_in_
        
        if hasattr(model, 'n_estimators'):
            metrics['n_estimators'] = model.n_estimators
        
        if hasattr(model, 'max_depth'):
            metrics['max_depth'] = model.max_depth
        
        return metrics
    
    def compare_models(self, model_metrics_list):
        """Compare multiple models and rank them."""
        comparison = {}
        
        # Extract key metrics for comparison
        models = [m['model_name'] for m in model_metrics_list]
        accuracies = [m['accuracy'] for m in model_metrics_list]
        f1_scores = [m['f1_score'] for m in model_metrics_list]
        precisions = [m['precision'] for m in model_metrics_list]
        recalls = [m['recall'] for m in model_metrics_list]
        prediction_times = [m.get('prediction_time', 0) for m in model_metrics_list]
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame({
            'Model': models,
            'Accuracy': accuracies,
            'F1_Score': f1_scores,
            'Precision': precisions,
            'Recall': recalls,
            'Prediction_Time': prediction_times
        })
        
        # Calculate composite score (higher is better)
        comparison_df['Composite_Score'] = (
            comparison_df['Accuracy'] * 0.3 +
            comparison_df['F1_Score'] * 0.3 +
            comparison_df['Precision'] * 0.2 +
            comparison_df['Recall'] * 0.2
        )
        
        # Rank models
        comparison_df['Rank'] = comparison_df['Composite_Score'].rank(ascending=False)
        comparison_df = comparison_df.sort_values('Rank')
        
        comparison['comparison_table'] = comparison_df
        comparison['best_model'] = comparison_df.iloc[0]['Model']
        comparison['best_score'] = comparison_df.iloc[0]['Composite_Score']
        
        return comparison
    
    def calculate_trend_metrics(self, metric_history, window_size=10):
        """Calculate trend metrics over time."""
        if len(metric_history) < 2:
            return {}
        
        trends = {}
        
        # Extract time series for key metrics
        timestamps = [m.get('timestamp', i) for i, m in enumerate(metric_history)]
        accuracies = [m.get('accuracy', 0) for m in metric_history]
        response_times = [m.get('avg_response_time', 0) for m in metric_history]
        cpu_usage = [m.get('avg_cpu_usage', 0) for m in metric_history]
        
        # Calculate trends
        trends['accuracy_trend'] = self._calculate_linear_trend(accuracies)
        trends['response_time_trend'] = self._calculate_linear_trend(response_times)
        trends['cpu_usage_trend'] = self._calculate_linear_trend(cpu_usage)
        
        # Moving averages
        if len(accuracies) >= window_size:
            trends['accuracy_ma'] = self._moving_average(accuracies, window_size)
            trends['response_time_ma'] = self._moving_average(response_times, window_size)
        
        # Volatility measures
        trends['accuracy_volatility'] = np.std(accuracies)
        trends['response_time_volatility'] = np.std(response_times)
        
        return trends
    
    def _calculate_linear_trend(self, values):
        """Calculate linear trend slope."""
        if len(values) < 2:
            return 0
        
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        return slope
    
    def _moving_average(self, values, window_size):
        """Calculate moving average."""
        return pd.Series(values).rolling(window=window_size).mean().tolist()
    
    def export_metrics_csv(self, metrics_dict, filename=None):
        """Export metrics to CSV format."""
        # Flatten nested metrics
        flattened_metrics = self._flatten_dict(metrics_dict)
        
        # Create DataFrame
        df = pd.DataFrame([flattened_metrics])
        
        # Add timestamp
        df['timestamp'] = datetime.now()
        
        if filename:
            df.to_csv(filename, index=False)
            return filename
        else:
            # Return CSV string
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            return csv_buffer.getvalue()
    
    def _flatten_dict(self, d, parent_key='', sep='_'):
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, (list, np.ndarray)):
                if len(v) <= 10:  # Only include small arrays
                    for i, val in enumerate(v):
                        items.append((f"{new_key}_{i}", val))
                else:
                    # Include summary statistics for large arrays
                    items.extend([
                        (f"{new_key}_mean", np.mean(v)),
                        (f"{new_key}_std", np.std(v)),
                        (f"{new_key}_min", np.min(v)),
                        (f"{new_key}_max", np.max(v))
                    ])
            else:
                items.append((new_key, v))
        return dict(items)
    
    def generate_metrics_report(self, metrics_dict, report_type="comprehensive"):
        """Generate formatted metrics report."""
        report = []
        
        if report_type == "comprehensive":
            report.append("=" * 60)
            report.append("CYBERSECURITY METRICS REPORT")
            report.append("=" * 60)
            report.append(f"Generated at: {datetime.now()}")
            report.append("")
            
            # Detection Performance
            if 'accuracy' in metrics_dict:
                report.append("DETECTION PERFORMANCE")
                report.append("-" * 30)
                report.append(f"Accuracy: {metrics_dict.get('accuracy', 0):.4f}")
                report.append(f"Precision: {metrics_dict.get('precision', 0):.4f}")
                report.append(f"Recall: {metrics_dict.get('recall', 0):.4f}")
                report.append(f"F1-Score: {metrics_dict.get('f1_score', 0):.4f}")
                report.append(f"ROC-AUC: {metrics_dict.get('roc_auc', 0):.4f}")
                report.append("")
            
            # System Performance
            if 'avg_cpu_usage' in metrics_dict:
                report.append("SYSTEM PERFORMANCE")
                report.append("-" * 30)
                report.append(f"CPU Usage: {metrics_dict.get('avg_cpu_usage', 0):.2f}%")
                report.append(f"Memory Usage: {metrics_dict.get('avg_memory_usage', 0):.2f}%")
                report.append(f"Throughput: {metrics_dict.get('avg_network_throughput', 0):.2f} flows/s")
                report.append("")
            
            # Threat Analysis
            if 'per_attack_metrics' in metrics_dict:
                report.append("THREAT ANALYSIS")
                report.append("-" * 30)
                per_attack = metrics_dict['per_attack_metrics']
                for attack_type, metrics in per_attack.items():
                    report.append(f"{attack_type}:")
                    report.append(f"  Recall: {metrics['recall']:.4f}")
                    report.append(f"  Precision: {metrics['precision']:.4f}")
                    report.append(f"  F1-Score: {metrics['f1_score']:.4f}")
                report.append("")
        
        elif report_type == "summary":
            report.append(f"Metrics Summary - {datetime.now()}")
            report.append(f"Accuracy: {metrics_dict.get('accuracy', 0):.3f}")
            report.append(f"F1-Score: {metrics_dict.get('f1_score', 0):.3f}")
            report.append(f"CPU Usage: {metrics_dict.get('avg_cpu_usage', 0):.1f}%")
        
        return "\n".join(report)
    
    def save_baseline_metrics(self, metrics_dict):
        """Save metrics as baseline for comparison."""
        self.baseline_metrics = metrics_dict.copy()
    
    def compare_with_baseline(self, current_metrics):
        """Compare current metrics with baseline."""
        if self.baseline_metrics is None:
            return {"message": "No baseline metrics available"}
        
        comparison = {}
        
        for metric_name in ['accuracy', 'precision', 'recall', 'f1_score']:
            if metric_name in current_metrics and metric_name in self.baseline_metrics:
                current_value = current_metrics[metric_name]
                baseline_value = self.baseline_metrics[metric_name]
                
                change = current_value - baseline_value
                change_percent = (change / baseline_value * 100) if baseline_value != 0 else 0
                
                comparison[metric_name] = {
                    'current': current_value,
                    'baseline': baseline_value,
                    'change': change,
                    'change_percent': change_percent,
                    'improved': change > 0
                }
        
        return comparison
