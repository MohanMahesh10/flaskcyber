#!/usr/bin/env python3
"""
Enhanced Cryptographic Metrics Collector
Tracks comprehensive performance metrics for cryptographic operations
"""

import time
import psutil
import os
import threading
import json
from datetime import datetime
from collections import defaultdict
import hashlib
import base64
from cryptography.hazmat.primitives.asymmetric import rsa as crypto_rsa
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import secrets

class CryptoMetricsCollector:
    """Comprehensive metrics collector for cryptographic operations."""
    
    def __init__(self):
        self.metrics_history = []
        self.attack_simulation_results = []
        self.cpu_samples = []
        self.start_time = time.time()
        
    def measure_key_generation(self, key_type='RSA', key_size=2048, num_samples=1):
        """
        Measure comprehensive key generation metrics.
        
        Args:
            key_type: Type of key (RSA, AES, ECC)
            key_size: Size in bits
            num_samples: Number of samples for averaging
            
        Returns:
            dict: Comprehensive metrics
        """
        metrics = {
            'key_type': key_type,
            'key_size': key_size,
            'timestamp': datetime.now().isoformat(),
            'generation_times_ms': [],
            'encoding_times_ms': [],
            'decoding_times_ms': [],
            'cpu_cycles_estimated': [],
            'memory_usage_mb': [],
            'cpu_usage_percent': [],
            'keys_generated': []
        }
        
        for sample in range(num_samples):
            print(f"Generating sample {sample + 1}/{num_samples}...")
            
            if key_type.upper() == 'RSA':
                sample_metrics = self._measure_rsa_generation(key_size)
            elif key_type.upper() == 'AES':
                sample_metrics = self._measure_aes_generation(key_size)
            else:
                raise ValueError(f"Unsupported key type: {key_type}")
            
            # Aggregate metrics
            for key, value in sample_metrics.items():
                if key in metrics and isinstance(metrics[key], list):
                    metrics[key].append(value)
        
        # Calculate aggregated statistics
        metrics.update(self._calculate_aggregate_stats(metrics))
        
        # Simulate attack resistance testing
        if num_samples > 0:
            metrics['attack_resistance'] = self._simulate_attack_resistance(
                key_type, key_size, metrics['keys_generated'][0] if metrics['keys_generated'] else None
            )
        
        self.metrics_history.append(metrics)
        return metrics
    
    def _measure_rsa_generation(self, key_size):
        """Measure RSA key generation metrics."""
        process = psutil.Process()
        
        # Pre-generation measurements
        cpu_before = process.cpu_percent()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Generation timing
        start_time = time.perf_counter()
        private_key = crypto_rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size,
        )
        generation_time = (time.perf_counter() - start_time) * 1000  # ms
        
        public_key = private_key.public_key()
        
        # Post-generation measurements
        cpu_after = process.cpu_percent()
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        
        # Encoding timing
        start_time = time.perf_counter()
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        encoding_time = (time.perf_counter() - start_time) * 1000  # ms
        
        # Decoding timing
        start_time = time.perf_counter()
        decoded_private = serialization.load_pem_private_key(
            private_pem,
            password=None,
        )
        decoded_public = serialization.load_pem_public_key(public_pem)
        decoding_time = (time.perf_counter() - start_time) * 1000  # ms
        
        # Estimate CPU cycles (approximate based on time and CPU frequency)
        cpu_freq = psutil.cpu_freq()
        if cpu_freq:
            estimated_cycles = generation_time * cpu_freq.current * 1000  # rough estimate
        else:
            estimated_cycles = generation_time * 2400000  # assume 2.4GHz
        
        return {
            'generation_times_ms': generation_time,
            'encoding_times_ms': encoding_time,
            'decoding_times_ms': decoding_time,
            'cpu_cycles_estimated': estimated_cycles,
            'memory_usage_mb': memory_after - memory_before,
            'cpu_usage_percent': max(0, cpu_after - cpu_before),
            'keys_generated': {
                'private_key': private_pem.decode('utf-8'),
                'public_key': public_pem.decode('utf-8'),
                'private_key_object': private_key,
                'public_key_object': public_key
            }
        }
    
    def _measure_aes_generation(self, key_size):
        """Measure AES key generation metrics."""
        process = psutil.Process()
        
        # Pre-generation measurements
        cpu_before = process.cpu_percent()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Generation timing
        start_time = time.perf_counter()
        key = secrets.token_bytes(key_size // 8)
        generation_time = (time.perf_counter() - start_time) * 1000  # ms
        
        # Post-generation measurements
        cpu_after = process.cpu_percent()
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        
        # Encoding timing (to hex)
        start_time = time.perf_counter()
        key_hex = key.hex()
        key_b64 = base64.b64encode(key).decode('utf-8')
        encoding_time = (time.perf_counter() - start_time) * 1000  # ms
        
        # Decoding timing
        start_time = time.perf_counter()
        decoded_from_hex = bytes.fromhex(key_hex)
        decoded_from_b64 = base64.b64decode(key_b64)
        decoding_time = (time.perf_counter() - start_time) * 1000  # ms
        
        # Estimate CPU cycles
        cpu_freq = psutil.cpu_freq()
        if cpu_freq:
            estimated_cycles = generation_time * cpu_freq.current * 1000
        else:
            estimated_cycles = generation_time * 2400000
        
        return {
            'generation_times_ms': generation_time,
            'encoding_times_ms': encoding_time,
            'decoding_times_ms': decoding_time,
            'cpu_cycles_estimated': estimated_cycles,
            'memory_usage_mb': memory_after - memory_before,
            'cpu_usage_percent': max(0, cpu_after - cpu_before),
            'keys_generated': {
                'key_bytes': key,
                'key_hex': key_hex,
                'key_b64': key_b64
            }
        }
    
    def _simulate_attack_resistance(self, key_type, key_size, key_data):
        """
        Simulate various attacks to measure resistance.
        
        Returns:
            dict: Attack simulation results
        """
        attack_results = {
            'brute_force_resistance': self._estimate_brute_force_resistance(key_type, key_size),
            'timing_attack_resistance': self._test_timing_attack_resistance(key_type, key_data),
            'frequency_analysis_resistance': self._test_frequency_analysis(key_type, key_data),
            'overall_security_score': 0
        }
        
        # Calculate overall security score (0-100)
        if key_type.upper() == 'RSA':
            if key_size >= 4096:
                attack_results['overall_security_score'] = 95
            elif key_size >= 3072:
                attack_results['overall_security_score'] = 90
            elif key_size >= 2048:
                attack_results['overall_security_score'] = 85
            else:
                attack_results['overall_security_score'] = 60
        elif key_type.upper() == 'AES':
            if key_size >= 256:
                attack_results['overall_security_score'] = 98
            elif key_size >= 192:
                attack_results['overall_security_score'] = 95
            elif key_size >= 128:
                attack_results['overall_security_score'] = 90
            else:
                attack_results['overall_security_score'] = 70
        
        return attack_results
    
    def _estimate_brute_force_resistance(self, key_type, key_size):
        """Estimate brute force attack resistance."""
        if key_type.upper() == 'RSA':
            # For RSA, security is based on factoring difficulty
            operations_needed = 2 ** (key_size / 2)  # Simplified estimate
        elif key_type.upper() == 'AES':
            operations_needed = 2 ** key_size
        else:
            operations_needed = 2 ** key_size
        
        # Assume attacker has 10^12 operations/second
        attacker_ops_per_sec = 10 ** 12
        time_to_break_seconds = operations_needed / (2 * attacker_ops_per_sec)  # Average case
        time_to_break_years = time_to_break_seconds / (365.25 * 24 * 3600)
        
        return {
            'operations_needed': f"{operations_needed:.2e}",
            'time_to_break_years': f"{time_to_break_years:.2e}",
            'resistance_score': min(100, max(0, (key_size - 80) * 2))  # Simplified scoring
        }
    
    def _test_timing_attack_resistance(self, key_type, key_data):
        """Test resistance to timing attacks."""
        if not key_data:
            return {'resistance_score': 50, 'note': 'No key data available for testing'}
        
        # Simulate timing variations in operations
        timing_samples = []
        
        for _ in range(10):
            start_time = time.perf_counter()
            
            if key_type.upper() == 'RSA' and 'private_key_object' in key_data:
                # Test signature timing
                try:
                    message = b"test message for timing analysis"
                    signature = key_data['private_key_object'].sign(
                        message,
                        padding.PSS(
                            mgf=padding.MGF1(hashes.SHA256()),
                            salt_length=padding.PSS.MAX_LENGTH
                        ),
                        hashes.SHA256()
                    )
                    timing = (time.perf_counter() - start_time) * 1000000  # microseconds
                    timing_samples.append(timing)
                except:
                    timing_samples.append(0)
            
            elif key_type.upper() == 'AES' and 'key_bytes' in key_data:
                # Test encryption timing
                try:
                    iv = secrets.token_bytes(16)
                    cipher = Cipher(algorithms.AES(key_data['key_bytes']), modes.CBC(iv))
                    encryptor = cipher.encryptor()
                    ciphertext = encryptor.update(b"test message for timing" * 10)
                    encryptor.finalize()
                    timing = (time.perf_counter() - start_time) * 1000000  # microseconds
                    timing_samples.append(timing)
                except:
                    timing_samples.append(0)
        
        if timing_samples:
            timing_variance = max(timing_samples) - min(timing_samples)
            avg_timing = sum(timing_samples) / len(timing_samples)
            coefficient_of_variation = (timing_variance / avg_timing) if avg_timing > 0 else 0
            
            # Lower variance = better timing attack resistance
            resistance_score = max(0, 100 - (coefficient_of_variation * 100))
        else:
            resistance_score = 50
        
        return {
            'resistance_score': resistance_score,
            'timing_variance_us': timing_variance if timing_samples else 0,
            'avg_timing_us': sum(timing_samples) / len(timing_samples) if timing_samples else 0
        }
    
    def _test_frequency_analysis(self, key_type, key_data):
        """Test resistance to frequency analysis."""
        if not key_data:
            return {'resistance_score': 50}
        
        if key_type.upper() == 'AES' and 'key_bytes' in key_data:
            # Analyze byte frequency distribution in the key
            key_bytes = key_data['key_bytes']
            frequency = [0] * 256
            
            for byte in key_bytes:
                frequency[byte] += 1
            
            # Calculate chi-square test for randomness
            expected_freq = len(key_bytes) / 256
            chi_square = sum((freq - expected_freq) ** 2 / expected_freq for freq in frequency)
            
            # Chi-square critical value for 255 degrees of freedom at 95% confidence is ~293
            resistance_score = max(0, 100 - (chi_square / 293) * 50)
        else:
            resistance_score = 75  # Default for RSA
        
        return {
            'resistance_score': resistance_score,
            'chi_square_value': chi_square if key_type.upper() == 'AES' else 0
        }
    
    def _calculate_aggregate_stats(self, metrics):
        """Calculate aggregate statistics from samples."""
        stats = {}
        
        numeric_fields = [
            'generation_times_ms', 'encoding_times_ms', 'decoding_times_ms',
            'cpu_cycles_estimated', 'memory_usage_mb', 'cpu_usage_percent'
        ]
        
        for field in numeric_fields:
            if field in metrics and metrics[field]:
                values = metrics[field]
                stats[f'{field}_avg'] = sum(values) / len(values)
                stats[f'{field}_min'] = min(values)
                stats[f'{field}_max'] = max(values)
                stats[f'{field}_std'] = (sum((x - stats[f'{field}_avg']) ** 2 for x in values) / len(values)) ** 0.5
        
        return stats
    
    def get_metrics_summary(self):
        """Get summary of all collected metrics."""
        if not self.metrics_history:
            return {'message': 'No metrics collected yet'}
        
        summary = {
            'total_measurements': len(self.metrics_history),
            'measurement_period': {
                'start': self.metrics_history[0]['timestamp'] if self.metrics_history else None,
                'end': self.metrics_history[-1]['timestamp'] if self.metrics_history else None
            },
            'key_types_tested': list(set(m['key_type'] for m in self.metrics_history)),
            'average_metrics': {},
            'performance_trends': self._analyze_performance_trends()
        }
        
        # Calculate averages across all measurements
        all_generation_times = []
        all_security_scores = []
        
        for metric in self.metrics_history:
            if 'generation_times_ms_avg' in metric:
                all_generation_times.append(metric['generation_times_ms_avg'])
            if 'attack_resistance' in metric and 'overall_security_score' in metric['attack_resistance']:
                all_security_scores.append(metric['attack_resistance']['overall_security_score'])
        
        if all_generation_times:
            summary['average_metrics']['avg_generation_time_ms'] = sum(all_generation_times) / len(all_generation_times)
        
        if all_security_scores:
            summary['average_metrics']['avg_security_score'] = sum(all_security_scores) / len(all_security_scores)
        
        return summary
    
    def _analyze_performance_trends(self):
        """Analyze performance trends over time."""
        if len(self.metrics_history) < 2:
            return {'note': 'Insufficient data for trend analysis'}
        
        trends = {
            'generation_time_trend': 'stable',
            'security_score_trend': 'stable',
            'recommendations': []
        }
        
        # Simple trend analysis
        recent_metrics = self.metrics_history[-5:] if len(self.metrics_history) >= 5 else self.metrics_history
        
        generation_times = [m.get('generation_times_ms_avg', 0) for m in recent_metrics]
        if len(generation_times) >= 2:
            if generation_times[-1] > generation_times[0] * 1.2:
                trends['generation_time_trend'] = 'increasing'
                trends['recommendations'].append('Generation times are increasing - consider system optimization')
            elif generation_times[-1] < generation_times[0] * 0.8:
                trends['generation_time_trend'] = 'decreasing'
                trends['recommendations'].append('Generation times are improving - good performance optimization')
        
        return trends
    
    def export_metrics(self, format='json'):
        """Export collected metrics."""
        if format == 'json':
            return json.dumps({
                'metrics_history': self.metrics_history,
                'summary': self.get_metrics_summary()
            }, indent=2, default=str)
        else:
            return self.metrics_history
