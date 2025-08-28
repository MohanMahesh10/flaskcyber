import json
import csv
import io
import base64
import random
import math
import statistics
from collections import Counter, defaultdict

class DataProcessor:
    """
    Comprehensive data processing module for cybersecurity applications.
    Handles CSV, JSON inputs for both crypto and IDS modules without numpy/pandas.
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.feature_columns = []
        self.feature_stats = {}  # Store mean/std for normalization
        
    def generate_ids_training_data(self, n_samples=10000, n_features=30):
        """Generate realistic IDS training data with multiple attack types."""
        
        # Generate base features using pure Python
        X = []
        for _ in range(n_samples):
            # Generate random features with some correlation
            sample = []
            for j in range(n_features):
                # Create features with different distributions
                if j % 3 == 0:
                    # Normal distribution approximation
                    val = sum([random.uniform(-1, 1) for _ in range(12)]) / 2
                elif j % 3 == 1:
                    # Exponential-like distribution
                    val = -math.log(random.random())
                else:
                    # Uniform distribution
                    val = random.uniform(-2, 2)
                sample.append(val)
            X.append(sample)
        
        # Create multi-class labels for different attack types
        attack_types = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R', 'DDoS', 'Malware', 'Phishing']
        y_multiclass = [random.choices(range(len(attack_types)), 
                                     weights=[0.6, 0.1, 0.08, 0.06, 0.04, 0.05, 0.04, 0.03])[0] 
                       for _ in range(n_samples)]
        
        # Add realistic network features
        additional_features = self._generate_network_features(n_samples)
        
        # Combine features
        X_enhanced = [X[i] + additional_features[i] for i in range(n_samples)]
        
        # Create feature names
        base_features = [f'feature_{i}' for i in range(n_features)]
        network_features = [
            'packet_size', 'flow_duration', 'bytes_sent', 'bytes_received',
            'packets_sent', 'packets_received', 'tcp_flags', 'port_number',
            'protocol_type', 'service_type', 'connection_state', 'payload_entropy'
        ]
        
        feature_names = base_features + network_features
        self.feature_columns = feature_names
        
        # Convert to list of dictionaries format
        data = []
        for i in range(n_samples):
            row = {}
            for j, feature in enumerate(feature_names):
                row[feature] = X_enhanced[i][j]
            row['attack_type'] = attack_types[y_multiclass[i]]
            row['is_attack'] = 1 if y_multiclass[i] > 0 else 0
            data.append(row)
        
        return data, [row['is_attack'] for row in data]
    
    def _generate_network_features(self, n_samples):
        """Generate realistic network traffic features."""
        features = []
        
        for _ in range(n_samples):
            # Packet size (bytes) - using exponential distribution approximation
            packet_size = -100 * math.log(random.random()) + 64
            
            # Flow duration (seconds) - exponential distribution
            flow_duration = -10 * math.log(random.random())
            
            # Bytes sent/received - exponential distribution
            bytes_sent = -1000 * math.log(random.random())
            bytes_received = -800 * math.log(random.random())
            
            # Packets sent/received
            packets_sent = max(1, int(bytes_sent / packet_size))
            packets_received = max(1, int(bytes_received / packet_size))
            
            # TCP flags (encoded as integer)
            tcp_flags = random.randint(0, 255)
            
            # Port number
            common_ports = [80, 443, 22, 21, 25, 53, 110, 143, 993, 995]
            port_number = random.choice(common_ports + [random.randint(1024, 65535)])
            
            # Protocol type (0=TCP, 1=UDP, 2=ICMP)
            protocol_type = random.choices([0, 1, 2], weights=[0.7, 0.25, 0.05])[0]
            
            # Service type (0=HTTP, 1=HTTPS, 2=SSH, etc.)
            service_type = random.randint(0, 9)
            
            # Connection state (0=established, 1=failed, 2=timeout, etc.)
            connection_state = random.choices([0, 1, 2, 3], weights=[0.6, 0.2, 0.1, 0.1])[0]
            
            # Payload entropy (measure of randomness in data)
            payload_entropy = random.uniform(0, 8)
            
            features.append([
                packet_size, flow_duration, bytes_sent, bytes_received,
                packets_sent, packets_received, tcp_flags, port_number,
                protocol_type, service_type, connection_state, payload_entropy
            ])
        
        return features
    
    def generate_sample_network_data(self, n_samples=1000):
        """Generate sample network data for testing."""
        return self.generate_ids_training_data(n_samples)[0]
    
    def process_csv_file(self, file_path_or_buffer, target_column=None):
        """Process CSV file for IDS training/testing."""
        try:
            data = []
            
            if hasattr(file_path_or_buffer, 'read'):
                # File-like object
                csv_content = file_path_or_buffer.read()
                if isinstance(csv_content, bytes):
                    csv_content = csv_content.decode('utf-8')
                csv_reader = csv.DictReader(io.StringIO(csv_content))
            else:
                # File path
                with open(file_path_or_buffer, 'r', encoding='utf-8') as f:
                    csv_reader = csv.DictReader(f)
                    data = list(csv_reader)
            
            if hasattr(file_path_or_buffer, 'read'):
                data = list(csv_reader)
            
            if not data:
                return None, None
            
            # Basic preprocessing
            # Handle missing values and convert types
            processed_data = []
            for row in data:
                processed_row = {}
                for key, value in row.items():
                    if value is None or value == '' or value.lower() == 'nan':
                        # Use median for numeric columns, mode for categorical
                        processed_row[key] = self._get_default_value(data, key)
                    else:
                        # Try to convert to number, otherwise keep as string
                        try:
                            processed_row[key] = float(value)
                        except ValueError:
                            processed_row[key] = str(value)
                processed_data.append(processed_row)
            
            # Encode categorical variables
            self._encode_categorical_features(processed_data)
            
            # Separate features and target
            if target_column and target_column in processed_data[0]:
                X = [{k: v for k, v in row.items() if k != target_column} for row in processed_data]
                y = [row[target_column] for row in processed_data]
                
                # Encode target if it's categorical
                if isinstance(y[0], str):
                    y = self._encode_target(y)
                
                return X, y
            else:
                return processed_data, None
            
        except Exception as e:
            print(f"Error processing CSV file: {str(e)}")
            return None, None
    
    def _get_default_value(self, data, column):
        """Get default value for missing data."""
        values = [row.get(column) for row in data if row.get(column) and row[column] != '']
        if not values:
            return 0
        
        # Try numeric
        try:
            numeric_values = [float(v) for v in values]
            return statistics.median(numeric_values)
        except:
            # Use most common value
            counter = Counter(values)
            return counter.most_common(1)[0][0] if counter else ''
    
    def _encode_categorical_features(self, data):
        """Encode categorical features in place."""
        if not data:
            return
        
        # Identify categorical columns
        categorical_cols = []
        for key, value in data[0].items():
            if isinstance(value, str):
                categorical_cols.append(key)
        
        # Encode each categorical column
        for col in categorical_cols:
            if col not in self.label_encoders:
                # Create encoder
                unique_values = list(set(row[col] for row in data if col in row))
                self.label_encoders[col] = {val: idx for idx, val in enumerate(unique_values)}
            
            # Apply encoding
            for row in data:
                if col in row:
                    row[col] = self.label_encoders[col].get(row[col], 0)
    
    def _encode_target(self, y):
        """Encode target variable."""
        if 'target' not in self.label_encoders:
            unique_values = list(set(y))
            self.label_encoders['target'] = {val: idx for idx, val in enumerate(unique_values)}
        
        return [self.label_encoders['target'].get(val, 0) for val in y]
    
    def process_json_file(self, file_path_or_buffer, flatten=True):
        """Process JSON file for network data."""
        try:
            if hasattr(file_path_or_buffer, 'read'):
                # File-like object
                data = json.load(file_path_or_buffer)
            else:
                # File path
                with open(file_path_or_buffer, 'r') as f:
                    data = json.load(f)
            
            # Convert to list of dictionaries
            if isinstance(data, list):
                processed_data = data if all(isinstance(item, dict) for item in data) else [{"value": item} for item in data]
            elif isinstance(data, dict):
                if flatten:
                    processed_data = [self._flatten_dict(data)]
                else:
                    processed_data = [data]
            else:
                processed_data = [{"value": data}]
            
            # Basic preprocessing - fill missing values
            for row in processed_data:
                for key, value in row.items():
                    if value is None:
                        row[key] = 0
            
            # Convert and encode
            self._convert_to_numeric(processed_data)
            self._encode_categorical_features(processed_data)
            
            return processed_data, None
            
        except Exception as e:
            print(f"Error processing JSON file: {str(e)}")
            return None, None
    
    def _flatten_dict(self, d, parent_key='', sep='_'):
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def _convert_to_numeric(self, data):
        """Convert string values to numeric where possible."""
        for row in data:
            for key, value in row.items():
                if isinstance(value, str):
                    try:
                        row[key] = float(value)
                    except ValueError:
                        pass
    
    def process_uploaded_data(self, uploaded_file, target_column=None):
        """Process uploaded file (CSV or JSON)."""
        if uploaded_file is None:
            return self.generate_sample_network_data(1000), [random.choice([0, 1]) for _ in range(1000)]
        
        try:
            # Determine file type
            file_name = uploaded_file.name if hasattr(uploaded_file, 'name') else str(uploaded_file)
            
            if file_name.endswith('.csv'):
                return self.process_csv_file(uploaded_file, target_column)
            elif file_name.endswith('.json'):
                return self.process_json_file(uploaded_file)
            else:
                # Try to read as CSV first, then JSON
                try:
                    return self.process_csv_file(uploaded_file, target_column)
                except:
                    return self.process_json_file(uploaded_file)
                    
        except Exception as e:
            print(f"Error processing uploaded file: {str(e)}")
            # Return sample data as fallback
            return self.generate_sample_network_data(1000), [random.choice([0, 1]) for _ in range(1000)]
    
    def process_image_data(self, images, extract_features=True):
        """Process network visualization images - simplified version without cv2/numpy."""
        # Skip image processing to avoid cv2 and numpy dependencies
        # Return empty list or basic feature extraction if needed
        print("Image processing skipped to avoid cv2/numpy dependencies")
        return []
    
    # Image processing methods removed to avoid cv2/numpy dependencies
    
    def preprocess_for_crypto(self, data):
        """Preprocess data for cryptographic key generation."""
        if isinstance(data, str):
            return data.encode('utf-8')
        elif isinstance(data, list):
            # Convert list of dicts to string
            data_str = json.dumps(data, sort_keys=True)
            return data_str.encode('utf-8')
        elif isinstance(data, dict):
            data_str = json.dumps(data, sort_keys=True)
            return data_str.encode('utf-8')
        else:
            return str(data).encode('utf-8')
    
    def validate_data(self, data, expected_features=None):
        """Validate input data format and quality."""
        validation_results = {
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'data_shape': None,
            'missing_values': 0,
            'data_types': {}
        }
        
        try:
            if isinstance(data, list) and data:
                if isinstance(data[0], dict):
                    # List of dictionaries
                    validation_results['data_shape'] = (len(data), len(data[0]))
                    
                    # Count missing values
                    missing_count = 0
                    for row in data:
                        for value in row.values():
                            if value is None or (isinstance(value, str) and value.lower() in ['nan', 'null', '']):
                                missing_count += 1
                    validation_results['missing_values'] = missing_count
                    
                    # Check data types
                    if data:
                        for key, value in data[0].items():
                            validation_results['data_types'][key] = type(value).__name__
                    
                    # Check minimum samples
                    if len(data) < 10:
                        validation_results['issues'].append("Insufficient data samples (minimum 10 required)")
                        validation_results['is_valid'] = False
                    
                    # Check expected features
                    if expected_features and data:
                        available_features = set(data[0].keys())
                        missing_features = set(expected_features) - available_features
                        if missing_features:
                            validation_results['warnings'].append(f"Missing expected features: {missing_features}")
                    
                    # Check for high missing value percentage
                    total_values = len(data) * len(data[0]) if data else 0
                    if total_values > 0 and missing_count > total_values * 0.5:
                        validation_results['issues'].append("Too many missing values (>50%)")
                        validation_results['is_valid'] = False
                else:
                    validation_results['data_shape'] = (len(data),)
            
            elif not data:
                validation_results['issues'].append("Empty data")
                validation_results['is_valid'] = False
            else:
                validation_results['warnings'].append("Unrecognized data type")
                
        except Exception as e:
            validation_results['issues'].append(f"Validation error: {str(e)}")
            validation_results['is_valid'] = False
        
        return validation_results
    
    def normalize_features(self, data, fit_scaler=True):
        """Normalize numerical features."""
        if not isinstance(data, list) or not data or not isinstance(data[0], dict):
            return data
        
        # Identify numeric columns
        numeric_cols = []
        for key, value in data[0].items():
            if isinstance(value, (int, float)):
                numeric_cols.append(key)
        
        if not numeric_cols:
            return data
        
        if fit_scaler:
            # Calculate mean and std for each numeric column
            for col in numeric_cols:
                values = [row[col] for row in data if col in row and isinstance(row[col], (int, float))]
                if values:
                    mean_val = statistics.mean(values)
                    std_val = statistics.stdev(values) if len(values) > 1 else 1.0
                    self.feature_stats[col] = {'mean': mean_val, 'std': std_val}
        
        # Apply normalization
        normalized_data = []
        for row in data:
            normalized_row = row.copy()
            for col in numeric_cols:
                if col in self.feature_stats and col in row:
                    stats = self.feature_stats[col]
                    normalized_row[col] = (row[col] - stats['mean']) / stats['std'] if stats['std'] > 0 else 0
            normalized_data.append(normalized_row)
        
        return normalized_data
    
    def export_processed_data(self, data, file_path, format='csv'):
        """Export processed data to file."""
        try:
            if format.lower() == 'csv':
                if isinstance(data, list) and data and isinstance(data[0], dict):
                    with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                        fieldnames = data[0].keys()
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(data)
                else:
                    raise ValueError("Data must be a list of dictionaries for CSV export")
            
            elif format.lower() == 'json':
                with open(file_path, 'w', encoding='utf-8') as jsonfile:
                    json.dump(data, jsonfile, indent=2)
            
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            print(f"Data exported to {file_path}")
            return True
        except Exception as e:
            print(f"Error exporting data: {str(e)}")
            return False
