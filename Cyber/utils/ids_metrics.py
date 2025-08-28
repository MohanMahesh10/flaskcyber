#!/usr/bin/env python3
"""
Advanced IDS Metrics Collector
Comprehensive ML performance tracking for Intrusion Detection Systems
"""

import time
import psutil
import numpy as np
from datetime import datetime
from collections import defaultdict, Counter
import json
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, confusion_matrix,
    classification_report
)
import threading
import queue

class IDSMetricsCollector:
    """Comprehensive ML metrics collector for IDS systems."""
    
    def __init__(self):
        self.detection_history = []
        self.real_time_metrics = {
            'total_predictions': 0,
            'threat_detections': 0,
            'false_positives': 0,
            'true_positives': 0,
            'false_negatives': 0,
            'true_negatives': 0,
            'detection_times': [],
            'cpu_usage_samples': [],
            'memory_usage_samples': []
        }
        self.attack_type_metrics = defaultdict(lambda: {
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'true_negatives': 0,
            'detection_times': [],
            'prediction_scores': []
        })
        self.performance_monitor = None
        self.start_monitoring()
    
    def start_monitoring(self):
        """Start real-time performance monitoring."""
        self.performance_monitor = threading.Thread(target=self._monitor_system_resources, daemon=True)
        self.performance_monitor.start()
    
    def _monitor_system_resources(self):
        """Monitor system resources in real-time."""
        while True:
            try:
                cpu_percent = psutil.cpu_percent(interval=1)
                memory_info = psutil.virtual_memory()
                
                self.real_time_metrics['cpu_usage_samples'].append({
                    'timestamp': datetime.now().isoformat(),
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory_info.percent,
                    'memory_mb': memory_info.used / 1024 / 1024
                })
                
                # Keep only last 100 samples to prevent memory bloat
                if len(self.real_time_metrics['cpu_usage_samples']) > 100:
                    self.real_time_metrics['cpu_usage_samples'] = \
                        self.real_time_metrics['cpu_usage_samples'][-100:]
                
                time.sleep(1)
            except Exception as e:
                print(f"Resource monitoring error: {e}")
                break
    
    def record_detection(self, true_labels, predicted_labels, prediction_scores=None, 
                        attack_types=None, detection_time_ms=None):
        """
        Record detection results for comprehensive metrics calculation.
        
        Args:
            true_labels: Ground truth labels (0=benign, 1=malicious)
            predicted_labels: Predicted labels (0=benign, 1=malicious)
            prediction_scores: Prediction confidence scores (optional)
            attack_types: Specific attack types for each sample (optional)
            detection_time_ms: Detection time in milliseconds (optional)
        """
        timestamp = datetime.now().isoformat()
        
        # Convert to numpy arrays for easier processing
        true_labels = np.array(true_labels)
        predicted_labels = np.array(predicted_labels)
        
        if prediction_scores is not None:
            prediction_scores = np.array(prediction_scores)
        
        # Calculate basic metrics
        metrics = self._calculate_comprehensive_metrics(
            true_labels, predicted_labels, prediction_scores
        )
        
        # Add timing and system info
        metrics.update({
            'timestamp': timestamp,
            'sample_count': len(true_labels),
            'detection_time_ms': detection_time_ms or 0,
            'throughput_samples_per_sec': len(true_labels) / (detection_time_ms / 1000) if detection_time_ms else 0
        })
        
        # Record per-attack-type metrics
        if attack_types is not None:
            attack_metrics = self._calculate_per_attack_metrics(
                true_labels, predicted_labels, attack_types, prediction_scores
            )
            metrics['per_attack_metrics'] = attack_metrics
            
            # Update running attack-specific counters
            self._update_attack_type_counters(
                true_labels, predicted_labels, attack_types, 
                prediction_scores, detection_time_ms
            )
        
        # Update real-time counters
        self._update_real_time_counters(true_labels, predicted_labels, detection_time_ms)
        
        # Store complete metrics
        self.detection_history.append(metrics)
        
        # Keep only last 1000 records to prevent memory issues
        if len(self.detection_history) > 1000:
            self.detection_history = self.detection_history[-1000:]
        
        return metrics
    
    def _calculate_comprehensive_metrics(self, true_labels, predicted_labels, prediction_scores=None):
        """Calculate comprehensive ML metrics."""
        metrics = {}
        
        try:
            # Basic classification metrics
            metrics['accuracy'] = accuracy_score(true_labels, predicted_labels)
            metrics['precision'] = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(true_labels, predicted_labels, average='weighted', zero_division=0)
            metrics['f1_score'] = f1_score(true_labels, predicted_labels, average='weighted', zero_division=0)
            
            # Confusion matrix
            cm = confusion_matrix(true_labels, predicted_labels)
            if cm.shape == (2, 2):  # Binary classification
                tn, fp, fn, tp = cm.ravel()
                metrics['true_negatives'] = int(tn)
                metrics['false_positives'] = int(fp)
                metrics['false_negatives'] = int(fn)
                metrics['true_positives'] = int(tp)
                
                # Calculate FPR (False Positive Rate)
                metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
                
                # Calculate TPR (True Positive Rate / Sensitivity)
                metrics['true_positive_rate'] = tp / (tp + fn) if (tp + fn) > 0 else 0
                
                # Calculate TNR (True Negative Rate / Specificity)
                metrics['true_negative_rate'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            # ROC-AUC and PR-AUC (if prediction scores available)
            if prediction_scores is not None and len(np.unique(true_labels)) > 1:
                try:
                    metrics['roc_auc'] = roc_auc_score(true_labels, prediction_scores)
                    
                    # Calculate PR-AUC
                    precision_curve, recall_curve, _ = precision_recall_curve(true_labels, prediction_scores)
                    metrics['pr_auc'] = auc(recall_curve, precision_curve)
                except Exception as e:
                    metrics['roc_auc'] = 0.0
                    metrics['pr_auc'] = 0.0
                    print(f"Error calculating AUC metrics: {e}")
            else:
                metrics['roc_auc'] = 0.0
                metrics['pr_auc'] = 0.0
            
            # Additional metrics
            metrics['support'] = len(true_labels)
            metrics['malicious_samples'] = int(np.sum(true_labels))
            metrics['benign_samples'] = int(len(true_labels) - np.sum(true_labels))
            
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            # Return default metrics in case of error
            metrics = {
                'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0,
                'roc_auc': 0.0, 'pr_auc': 0.0, 'false_positive_rate': 0.0,
                'support': len(true_labels)
            }
        
        return metrics
    
    def _calculate_per_attack_metrics(self, true_labels, predicted_labels, attack_types, prediction_scores=None):
        """Calculate metrics per attack type."""
        attack_metrics = {}
        unique_attacks = list(set(attack_types))
        
        for attack_type in unique_attacks:
            # Get indices for this attack type
            attack_indices = np.array([i for i, at in enumerate(attack_types) if at == attack_type])
            
            if len(attack_indices) == 0:
                continue
            
            attack_true = true_labels[attack_indices]
            attack_pred = predicted_labels[attack_indices]
            attack_scores = prediction_scores[attack_indices] if prediction_scores is not None else None
            
            # Calculate metrics for this attack type
            try:
                attack_metrics[attack_type] = {
                    'accuracy': accuracy_score(attack_true, attack_pred),
                    'precision': precision_score(attack_true, attack_pred, average='weighted', zero_division=0),
                    'recall': recall_score(attack_true, attack_pred, average='weighted', zero_division=0),
                    'f1_score': f1_score(attack_true, attack_pred, average='weighted', zero_division=0),
                    'support': len(attack_indices),
                    'samples': int(len(attack_indices))
                }
                
                # Confusion matrix for this attack type
                if len(np.unique(attack_true)) > 1 and len(np.unique(attack_pred)) > 1:
                    cm = confusion_matrix(attack_true, attack_pred)
                    if cm.shape == (2, 2):
                        tn, fp, fn, tp = cm.ravel()
                        attack_metrics[attack_type].update({
                            'true_positives': int(tp),
                            'false_positives': int(fp),
                            'false_negatives': int(fn),
                            'true_negatives': int(tn),
                            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0
                        })
                
            except Exception as e:
                attack_metrics[attack_type] = {
                    'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0,
                    'support': len(attack_indices), 'error': str(e)
                }
        
        return attack_metrics
    
    def _update_attack_type_counters(self, true_labels, predicted_labels, attack_types, 
                                    prediction_scores=None, detection_time_ms=None):
        """Update running counters for each attack type."""
        for i, attack_type in enumerate(attack_types):
            true_label = true_labels[i]
            pred_label = predicted_labels[i]
            
            # Update confusion matrix counters
            if true_label == 1 and pred_label == 1:
                self.attack_type_metrics[attack_type]['true_positives'] += 1
            elif true_label == 0 and pred_label == 1:
                self.attack_type_metrics[attack_type]['false_positives'] += 1
            elif true_label == 1 and pred_label == 0:
                self.attack_type_metrics[attack_type]['false_negatives'] += 1
            else:  # true_label == 0 and pred_label == 0
                self.attack_type_metrics[attack_type]['true_negatives'] += 1
            
            # Update timing info
            if detection_time_ms:
                self.attack_type_metrics[attack_type]['detection_times'].append(detection_time_ms)
            
            # Update prediction scores
            if prediction_scores is not None:
                self.attack_type_metrics[attack_type]['prediction_scores'].append(prediction_scores[i])
    
    def _update_real_time_counters(self, true_labels, predicted_labels, detection_time_ms=None):
        """Update real-time performance counters."""
        self.real_time_metrics['total_predictions'] += len(true_labels)
        self.real_time_metrics['threat_detections'] += np.sum(predicted_labels)
        
        # Update confusion matrix counters
        for true_label, pred_label in zip(true_labels, predicted_labels):
            if true_label == 1 and pred_label == 1:
                self.real_time_metrics['true_positives'] += 1
            elif true_label == 0 and pred_label == 1:
                self.real_time_metrics['false_positives'] += 1
            elif true_label == 1 and pred_label == 0:
                self.real_time_metrics['false_negatives'] += 1
            else:  # true_label == 0 and pred_label == 0
                self.real_time_metrics['true_negatives'] += 1
        
        # Update timing info
        if detection_time_ms:
            self.real_time_metrics['detection_times'].append(detection_time_ms)
            # Keep only last 1000 detection times
            if len(self.real_time_metrics['detection_times']) > 1000:
                self.real_time_metrics['detection_times'] = self.real_time_metrics['detection_times'][-1000:]
    
    def get_real_time_metrics(self):
        """Get current real-time metrics."""
        metrics = self.real_time_metrics.copy()
        
        # Calculate derived metrics
        total = metrics['total_predictions']
        if total > 0:
            metrics['accuracy'] = (metrics['true_positives'] + metrics['true_negatives']) / total
            metrics['precision'] = metrics['true_positives'] / (metrics['true_positives'] + metrics['false_positives']) if (metrics['true_positives'] + metrics['false_positives']) > 0 else 0
            metrics['recall'] = metrics['true_positives'] / (metrics['true_positives'] + metrics['false_negatives']) if (metrics['true_positives'] + metrics['false_negatives']) > 0 else 0
            metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall']) if (metrics['precision'] + metrics['recall']) > 0 else 0
            metrics['false_positive_rate'] = metrics['false_positives'] / (metrics['false_positives'] + metrics['true_negatives']) if (metrics['false_positives'] + metrics['true_negatives']) > 0 else 0
        
        # Calculate timing metrics
        if metrics['detection_times']:
            metrics['mean_detection_latency_ms'] = np.mean(metrics['detection_times'])
            metrics['median_detection_latency_ms'] = np.median(metrics['detection_times'])
            metrics['min_detection_latency_ms'] = np.min(metrics['detection_times'])
            metrics['max_detection_latency_ms'] = np.max(metrics['detection_times'])
            metrics['std_detection_latency_ms'] = np.std(metrics['detection_times'])
        
        # Calculate throughput
        if metrics['cpu_usage_samples']:
            recent_samples = metrics['cpu_usage_samples'][-10:]  # Last 10 samples
            metrics['current_cpu_usage'] = np.mean([s['cpu_percent'] for s in recent_samples])
            metrics['current_memory_usage'] = np.mean([s['memory_percent'] for s in recent_samples])
            metrics['current_memory_mb'] = np.mean([s['memory_mb'] for s in recent_samples])
        
        # Add timestamp
        metrics['timestamp'] = datetime.now().isoformat()
        
        return metrics
    
    def get_per_attack_metrics_matrix(self):
        """Get comprehensive per-attack metrics matrix."""
        matrix = {}
        
        for attack_type, counters in self.attack_type_metrics.items():
            tp = counters['true_positives']
            fp = counters['false_positives']
            fn = counters['false_negatives']
            tn = counters['true_negatives']
            
            total = tp + fp + fn + tn
            if total > 0:
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                accuracy = (tp + tn) / total
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                
                matrix[attack_type] = {
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'accuracy': accuracy,
                    'false_positive_rate': fpr,
                    'support': total,
                    'true_positives': tp,
                    'false_positives': fp,
                    'false_negatives': fn,
                    'true_negatives': tn
                }
                
                # Add timing metrics
                if counters['detection_times']:
                    matrix[attack_type]['mean_detection_time_ms'] = np.mean(counters['detection_times'])
                    matrix[attack_type]['median_detection_time_ms'] = np.median(counters['detection_times'])
                
                # Add prediction score statistics
                if counters['prediction_scores']:
                    matrix[attack_type]['mean_confidence_score'] = np.mean(counters['prediction_scores'])
                    matrix[attack_type]['std_confidence_score'] = np.std(counters['prediction_scores'])
        
        return matrix
    
    def get_comprehensive_report(self):
        """Generate comprehensive performance report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'real_time_metrics': self.get_real_time_metrics(),
            'per_attack_metrics': self.get_per_attack_metrics_matrix(),
            'historical_summary': self._get_historical_summary(),
            'performance_trends': self._analyze_performance_trends(),
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _get_historical_summary(self):
        """Get summary of historical performance."""
        if not self.detection_history:
            return {'message': 'No historical data available'}
        
        # Calculate averages across all detection runs
        avg_metrics = {}
        metric_keys = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc', 'false_positive_rate']
        
        for key in metric_keys:
            values = [h.get(key, 0) for h in self.detection_history if key in h]
            if values:
                avg_metrics[f'avg_{key}'] = np.mean(values)
                avg_metrics[f'std_{key}'] = np.std(values)
                avg_metrics[f'min_{key}'] = np.min(values)
                avg_metrics[f'max_{key}'] = np.max(values)
        
        # Calculate throughput statistics
        throughput_values = [h.get('throughput_samples_per_sec', 0) for h in self.detection_history]
        if throughput_values:
            avg_metrics['avg_throughput_samples_per_sec'] = np.mean(throughput_values)
            avg_metrics['max_throughput_samples_per_sec'] = np.max(throughput_values)
        
        return {
            'total_detection_runs': len(self.detection_history),
            'total_samples_processed': sum(h.get('sample_count', 0) for h in self.detection_history),
            'average_metrics': avg_metrics,
            'time_period': {
                'start': self.detection_history[0]['timestamp'],
                'end': self.detection_history[-1]['timestamp']
            }
        }
    
    def _analyze_performance_trends(self):
        """Analyze performance trends over time."""
        if len(self.detection_history) < 5:
            return {'message': 'Insufficient data for trend analysis'}
        
        recent_runs = self.detection_history[-10:]  # Last 10 runs
        
        trends = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            values = [r.get(metric, 0) for r in recent_runs]
            if len(values) >= 2:
                # Simple linear trend
                x = np.arange(len(values))
                coeffs = np.polyfit(x, values, 1)
                slope = coeffs[0]
                
                if slope > 0.01:
                    trends[metric] = 'improving'
                elif slope < -0.01:
                    trends[metric] = 'declining'
                else:
                    trends[metric] = 'stable'
        
        return trends
    
    def _generate_recommendations(self):
        """Generate performance recommendations."""
        recommendations = []
        real_time = self.get_real_time_metrics()
        
        # Check accuracy
        if real_time.get('accuracy', 0) < 0.85:
            recommendations.append("Low accuracy detected. Consider retraining the model with more diverse data.")
        
        # Check false positive rate
        if real_time.get('false_positive_rate', 0) > 0.1:
            recommendations.append("High false positive rate. Consider adjusting decision threshold or feature engineering.")
        
        # Check recall
        if real_time.get('recall', 0) < 0.8:
            recommendations.append("Low recall rate. The model may be missing threats. Consider improving sensitivity.")
        
        # Check detection latency
        if real_time.get('mean_detection_latency_ms', 0) > 1000:
            recommendations.append("High detection latency. Consider model optimization or hardware upgrades.")
        
        # Check CPU usage
        if real_time.get('current_cpu_usage', 0) > 80:
            recommendations.append("High CPU usage detected. Consider load balancing or performance optimization.")
        
        return recommendations
    
    def export_metrics(self, format='json'):
        """Export all collected metrics."""
        data = {
            'real_time_metrics': self.get_real_time_metrics(),
            'per_attack_metrics': self.get_per_attack_metrics_matrix(),
            'detection_history': self.detection_history[-100:],  # Last 100 records
            'comprehensive_report': self.get_comprehensive_report()
        }
        
        if format == 'json':
            return json.dumps(data, indent=2, default=str)
        else:
            return data
    
    def reset_metrics(self):
        """Reset all metrics (useful for testing)."""
        self.detection_history.clear()
        self.real_time_metrics = {
            'total_predictions': 0,
            'threat_detections': 0,
            'false_positives': 0,
            'true_positives': 0,
            'false_negatives': 0,
            'true_negatives': 0,
            'detection_times': [],
            'cpu_usage_samples': [],
            'memory_usage_samples': []
        }
        self.attack_type_metrics.clear()
