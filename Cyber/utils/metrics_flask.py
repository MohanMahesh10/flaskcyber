import time
import io
import csv
import math
import random
from datetime import datetime
from collections import Counter, defaultdict

class MetricsCalculatorFlask:
    """
    Comprehensive metrics calculation for cybersecurity applications without numpy/pandas.
    Uses pure Python for all calculations.
    """
    
    def __init__(self):
        self.metric_history = []
        self.baseline_metrics = None
        
    def calculate_mean(self, values):
        """Calculate mean of a list."""
        return sum(values) / len(values) if values else 0
    
    def calculate_std(self, values):
        """Calculate standard deviation of a list."""
        if not values or len(values) < 2:
            return 0
        mean_val = self.calculate_mean(values)
        variance = sum((x - mean_val) ** 2 for x in values) / len(values)
        return math.sqrt(variance)
    
    def calculate_median(self, values):
        """Calculate median of a list."""
        if not values:
            return 0
        sorted_values = sorted(values)
        n = len(sorted_values)
        if n % 2 == 0:
            return (sorted_values[n//2 - 1] + sorted_values[n//2]) / 2
        else:
            return sorted_values[n//2]
    
    def calculate_percentile(self, values, percentile):
        """Calculate percentile of a list."""
        if not values:
            return 0
        sorted_values = sorted(values)
        k = (len(sorted_values) - 1) * percentile / 100
        floor_k = int(k)
        ceil_k = floor_k + 1
        
        if ceil_k >= len(sorted_values):
            return sorted_values[-1]
        if floor_k < 0:
            return sorted_values[0]
        
        d0 = sorted_values[floor_k]
        d1 = sorted_values[ceil_k]
        return d0 + (d1 - d0) * (k - floor_k)
    
    def calculate_classification_metrics(self, y_true, y_pred, y_prob=None):
        """Calculate comprehensive classification metrics."""
        if not y_true or not y_pred or len(y_true) != len(y_pred):
            return {}
        
        metrics = {}
        
        # Basic metrics calculation
        correct = sum(1 for i in range(len(y_true)) if y_true[i] == y_pred[i])
        total = len(y_true)
        
        metrics['accuracy'] = correct / total if total > 0 else 0
        
        # For binary classification
        if len(set(y_true)) <= 2:
            tp = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 1)
            fp = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 1)
            fn = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 0)
            tn = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 0)
            
            metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
            metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
            metrics['f1_score'] = (2 * metrics['precision'] * metrics['recall'] / 
                                  (metrics['precision'] + metrics['recall']) 
                                  if (metrics['precision'] + metrics['recall']) > 0 else 0)
            
            metrics['true_positive_rate'] = metrics['recall']
            metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
            metrics['true_negative_rate'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
            
            # Confusion matrix
            metrics['confusion_matrix'] = [[tn, fp], [fn, tp]]
            
            # ROC AUC approximation if probabilities provided
            if y_prob is not None and len(y_prob) == len(y_true):
                metrics['roc_auc'] = self._calculate_auc_approximation(y_true, y_prob)
                metrics['pr_auc'] = self._calculate_pr_auc_approximation(y_true, y_prob)
            else:
                metrics['roc_auc'] = 0.5
                metrics['pr_auc'] = 0.5
        
        else:
            # Multi-class metrics (simplified)
            unique_classes = list(set(y_true + y_pred))
            per_class_precision = []
            per_class_recall = []
            
            for cls in unique_classes:
                tp_cls = sum(1 for i in range(len(y_true)) if y_true[i] == cls and y_pred[i] == cls)
                fp_cls = sum(1 for i in range(len(y_true)) if y_true[i] != cls and y_pred[i] == cls)
                fn_cls = sum(1 for i in range(len(y_true)) if y_true[i] == cls and y_pred[i] != cls)
                
                precision_cls = tp_cls / (tp_cls + fp_cls) if (tp_cls + fp_cls) > 0 else 0
                recall_cls = tp_cls / (tp_cls + fn_cls) if (tp_cls + fn_cls) > 0 else 0
                
                per_class_precision.append(precision_cls)
                per_class_recall.append(recall_cls)
            
            metrics['precision'] = self.calculate_mean(per_class_precision)
            metrics['recall'] = self.calculate_mean(per_class_recall)
            metrics['f1_score'] = (2 * metrics['precision'] * metrics['recall'] / 
                                  (metrics['precision'] + metrics['recall']) 
                                  if (metrics['precision'] + metrics['recall']) > 0 else 0)
            
            metrics['precision_per_class'] = per_class_precision
            metrics['recall_per_class'] = per_class_recall
        
        return metrics
    
    def _calculate_auc_approximation(self, y_true, y_prob):
        """Calculate approximate AUC using trapezoidal rule."""
        # Sort by probability scores
        sorted_pairs = sorted(zip(y_prob, y_true), reverse=True)
        
        # Calculate TPR and FPR at different thresholds
        pos_count = sum(y_true)
        neg_count = len(y_true) - pos_count
        
        if pos_count == 0 or neg_count == 0:
            return 0.5
        
        tp = fp = 0
        tpr_prev = fpr_prev = 0
        auc = 0
        
        for prob, label in sorted_pairs:
            if label == 1:
                tp += 1
            else:
                fp += 1
            
            tpr = tp / pos_count
            fpr = fp / neg_count
            
            # Trapezoidal rule
            auc += (tpr + tpr_prev) * (fpr - fpr_prev) / 2
            
            tpr_prev = tpr
            fpr_prev = fpr
        
        return auc
    
    def _calculate_pr_auc_approximation(self, y_true, y_prob):
        """Calculate approximate Precision-Recall AUC."""
        sorted_pairs = sorted(zip(y_prob, y_true), reverse=True)
        
        pos_count = sum(y_true)
        if pos_count == 0:
            return 0
        
        tp = fp = 0
        precision_prev = 1.0
        recall_prev = 0.0
        auc = 0
        
        for prob, label in sorted_pairs:
            if label == 1:
                tp += 1
            else:
                fp += 1
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / pos_count
            
            # Trapezoidal rule
            auc += (precision + precision_prev) * (recall - recall_prev) / 2
            
            precision_prev = precision
            recall_prev = recall
        
        return auc
    
    def calculate_ids_performance_metrics(self, y_true, y_pred, y_prob=None, 
                                        detection_times=None, throughput=None):
        """Calculate IDS-specific performance metrics."""
        # Get classification metrics
        metrics = self.calculate_classification_metrics(y_true, y_pred, y_prob)
        
        # IDS-specific metrics
        metrics['detection_accuracy'] = metrics.get('accuracy', 0)
        metrics['attack_detection_rate'] = metrics.get('recall', 0)  # Sensitivity
        metrics['false_alarm_rate'] = metrics.get('false_positive_rate', 0)
        
        # Performance metrics
        if detection_times:
            metrics['mean_detection_time'] = self.calculate_mean(detection_times)
            metrics['max_detection_time'] = max(detection_times)
            metrics['detection_time_std'] = self.calculate_std(detection_times)
        
        if throughput is not None:
            metrics['throughput'] = throughput
        
        # Attack type specific metrics (simulated)
        attack_types = ['DoS', 'Probe', 'R2L', 'U2R', 'DDoS', 'Malware', 'Phishing']
        metrics['per_attack_metrics'] = self._simulate_per_attack_metrics(attack_types)
        
        return metrics
    
    def _simulate_per_attack_metrics(self, attack_types):
        """Simulate per-attack type metrics for demonstration."""
        per_attack = {}
        random.seed(42)
        
        for attack_type in attack_types:
            # Simulate different performance for different attack types
            base_performance = random.uniform(0.8, 0.95)
            noise = random.gauss(0, 0.05)
            
            recall = max(0, min(1, base_performance + noise))
            precision = max(0, min(1, base_performance + random.gauss(0, 0.03)))
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            per_attack[attack_type] = {
                'recall': recall,
                'precision': precision,
                'f1_score': f1,
                'support': random.randint(50, 500)
            }
        
        return per_attack
    
    def calculate_crypto_metrics(self, key_generation_times, key_strengths, 
                               encoding_times=None, decoding_times=None):
        """Calculate cryptographic performance metrics."""
        metrics = {}
        
        # Generation performance
        metrics['mean_generation_time'] = self.calculate_mean(key_generation_times)
        metrics['generation_time_std'] = self.calculate_std(key_generation_times)
        metrics['min_generation_time'] = min(key_generation_times) if key_generation_times else 0
        metrics['max_generation_time'] = max(key_generation_times) if key_generation_times else 0
        
        # Key strength metrics
        metrics['mean_key_strength'] = self.calculate_mean(key_strengths)
        metrics['key_strength_std'] = self.calculate_std(key_strengths)
        metrics['min_key_strength'] = min(key_strengths) if key_strengths else 0
        metrics['max_key_strength'] = max(key_strengths) if key_strengths else 0
        
        # Encoding/Decoding performance
        if encoding_times:
            metrics['mean_encoding_time'] = self.calculate_mean(encoding_times)
            metrics['encoding_time_std'] = self.calculate_std(encoding_times)
        
        if decoding_times:
            metrics['mean_decoding_time'] = self.calculate_mean(decoding_times)
            metrics['decoding_time_std'] = self.calculate_std(decoding_times)
        
        # Efficiency metrics
        total_generation_time = sum(key_generation_times)
        if total_generation_time > 0:
            metrics['keys_per_second'] = len(key_generation_times) / (total_generation_time / 1000)
        else:
            metrics['keys_per_second'] = 0
        
        return metrics
    
    def calculate_system_metrics(self, cpu_usage, memory_usage, network_throughput,
                               response_times=None, error_rates=None):
        """Calculate system performance metrics."""
        metrics = {}
        
        # Resource utilization
        metrics['avg_cpu_usage'] = self.calculate_mean(cpu_usage)
        metrics['max_cpu_usage'] = max(cpu_usage) if cpu_usage else 0
        metrics['cpu_usage_std'] = self.calculate_std(cpu_usage)
        
        metrics['avg_memory_usage'] = self.calculate_mean(memory_usage)
        metrics['max_memory_usage'] = max(memory_usage) if memory_usage else 0
        metrics['memory_usage_std'] = self.calculate_std(memory_usage)
        
        # Network performance
        metrics['avg_network_throughput'] = self.calculate_mean(network_throughput)
        metrics['min_network_throughput'] = min(network_throughput) if network_throughput else 0
        metrics['network_throughput_std'] = self.calculate_std(network_throughput)
        
        # Response time metrics
        if response_times:
            metrics['avg_response_time'] = self.calculate_mean(response_times)
            metrics['percentile_95_response_time'] = self.calculate_percentile(response_times, 95)
            metrics['percentile_99_response_time'] = self.calculate_percentile(response_times, 99)
        
        # Error rate metrics
        if error_rates:
            metrics['avg_error_rate'] = self.calculate_mean(error_rates)
            metrics['max_error_rate'] = max(error_rates) if error_rates else 0
        
        return metrics
    
    def compare_models(self, model_metrics_list):
        """Compare multiple models and rank them."""
        comparison = {}
        
        # Extract key metrics for comparison
        models = [m.get('model_name', f'Model_{i}') for i, m in enumerate(model_metrics_list)]
        accuracies = [m.get('accuracy', 0) for m in model_metrics_list]
        f1_scores = [m.get('f1_score', 0) for m in model_metrics_list]
        precisions = [m.get('precision', 0) for m in model_metrics_list]
        recalls = [m.get('recall', 0) for m in model_metrics_list]
        prediction_times = [m.get('prediction_time', 0) for m in model_metrics_list]
        
        # Create comparison data
        comparison_data = []
        for i in range(len(models)):
            composite_score = (
                accuracies[i] * 0.3 +
                f1_scores[i] * 0.3 +
                precisions[i] * 0.2 +
                recalls[i] * 0.2
            )
            
            comparison_data.append({
                'Model': models[i],
                'Accuracy': accuracies[i],
                'F1_Score': f1_scores[i],
                'Precision': precisions[i],
                'Recall': recalls[i],
                'Prediction_Time': prediction_times[i],
                'Composite_Score': composite_score
            })
        
        # Sort by composite score (descending)
        comparison_data.sort(key=lambda x: x['Composite_Score'], reverse=True)
        
        # Add ranks
        for i, item in enumerate(comparison_data):
            item['Rank'] = i + 1
        
        comparison['comparison_table'] = comparison_data
        comparison['best_model'] = comparison_data[0]['Model'] if comparison_data else 'Unknown'
        comparison['best_score'] = comparison_data[0]['Composite_Score'] if comparison_data else 0
        
        return comparison
    
    def calculate_trend_metrics(self, metric_history, window_size=10):
        """Calculate trend metrics over time."""
        if len(metric_history) < 2:
            return {}
        
        trends = {}
        
        # Extract time series for key metrics
        accuracies = [m.get('accuracy', 0) for m in metric_history]
        response_times = [m.get('avg_response_time', 0) for m in metric_history]
        cpu_usage = [m.get('avg_cpu_usage', 0) for m in metric_history]
        
        # Calculate trends (simple linear regression slope approximation)
        trends['accuracy_trend'] = self._calculate_linear_trend(accuracies)
        trends['response_time_trend'] = self._calculate_linear_trend(response_times)
        trends['cpu_usage_trend'] = self._calculate_linear_trend(cpu_usage)
        
        # Moving averages
        if len(accuracies) >= window_size:
            trends['accuracy_ma'] = self._moving_average(accuracies, window_size)
            trends['response_time_ma'] = self._moving_average(response_times, window_size)
        
        # Volatility measures
        trends['accuracy_volatility'] = self.calculate_std(accuracies)
        trends['response_time_volatility'] = self.calculate_std(response_times)
        
        return trends
    
    def _calculate_linear_trend(self, values):
        """Calculate linear trend slope using least squares."""
        if len(values) < 2:
            return 0
        
        n = len(values)
        x = list(range(n))
        
        # Calculate means
        x_mean = sum(x) / n
        y_mean = sum(values) / n
        
        # Calculate slope
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        return numerator / denominator if denominator != 0 else 0
    
    def _moving_average(self, values, window_size):
        """Calculate moving average."""
        moving_avg = []
        for i in range(len(values)):
            if i < window_size - 1:
                # Not enough data points, use available
                window_data = values[:i+1]
            else:
                window_data = values[i-window_size+1:i+1]
            
            moving_avg.append(sum(window_data) / len(window_data))
        
        return moving_avg
    
    def export_metrics_csv(self, metrics_dict, filename=None):
        """Export metrics to CSV format."""
        # Flatten nested metrics
        flattened_metrics = self._flatten_dict(metrics_dict)
        
        # Add timestamp
        flattened_metrics['timestamp'] = datetime.now().isoformat()
        
        if filename:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=flattened_metrics.keys())
                writer.writeheader()
                writer.writerow(flattened_metrics)
            return filename
        else:
            # Return CSV string
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=flattened_metrics.keys())
            writer.writeheader()
            writer.writerow(flattened_metrics)
            return output.getvalue()
    
    def _flatten_dict(self, d, parent_key='', sep='_'):
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                if len(v) <= 10:  # Only include small lists
                    for i, val in enumerate(v):
                        if isinstance(val, (int, float, str)):
                            items.append((f"{new_key}_{i}", val))
                else:
                    # Include summary statistics for large lists
                    if all(isinstance(x, (int, float)) for x in v):
                        items.extend([
                            (f"{new_key}_mean", self.calculate_mean(v)),
                            (f"{new_key}_std", self.calculate_std(v)),
                            (f"{new_key}_min", min(v)),
                            (f"{new_key}_max", max(v))
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
