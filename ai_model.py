import numpy as np
import joblib
import os
import random
import time
import math
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import SelectFromModel

class NetworkAttackDetector:
    """
    A class for detecting network attacks using machine learning.
    This is a simplified implementation for demonstration purposes.
    In a real-world scenario, this would be trained on actual network traffic data.
    """
    
    def __init__(self):
        self.model = None
        self.model_path = os.path.join('model', 'attack_detection_model.joblib')
        self.history_path = os.path.join('model', 'detection_history.joblib')
        self.feature_names = [
            # Basic packet features
            'packet_size', 'packet_rate', 'source_port', 'destination_port',
            'tcp_flags', 'udp_length', 'icmp_type', 'connection_duration',
            'bytes_in', 'bytes_out', 'packets_in', 'packets_out',
            
            # Advanced features
            'packet_interval', 'packet_size_variance', 'entropy',
            'flag_distribution', 'port_distribution', 'payload_pattern',
            'time_of_day', 'day_of_week', 'protocol_type',
            'ttl_value', 'window_size', 'urgent_pointer'
        ]
        
        # Detection history for anomaly detection
        self.detection_history = []
        self.anomaly_threshold = 0.85
        self.max_history_size = 1000
        
        # Create model directory if it doesn't exist
        os.makedirs('model', exist_ok=True)
        
        # Load detection history if exists
        if os.path.exists(self.history_path):
            try:
                self.detection_history = joblib.load(self.history_path)
                print(f"Loaded {len(self.detection_history)} historical detection records")
            except Exception as e:
                print(f"Error loading detection history: {e}")
                self.detection_history = []
        
        # Load or train model
        if os.path.exists(self.model_path):
            self.load_model()
        else:
            self.train_model()
    
    def load_model(self):
        """Load the trained model from disk."""
        try:
            self.model = joblib.load(self.model_path)
            print(f"Model loaded from {self.model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.train_model()
    
    def train_model(self):
        """
        Train a new model for attack detection.
        In a real application, this would use actual network traffic data.
        """
        print("Training new attack detection model...")
        
        # Generate synthetic training data
        X, y = self._generate_synthetic_data(n_samples=20000)
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Feature selection to identify most important features
        print("Performing feature selection...")
        selector = SelectFromModel(
            GradientBoostingClassifier(n_estimators=100, random_state=42),
            threshold='median'
        )
        selector.fit(X_train, y_train)
        X_train_selected = selector.transform(X_train)
        X_test_selected = selector.transform(X_test)
        
        # Selected feature indices
        selected_features = selector.get_support()
        print(f"Selected {sum(selected_features)} out of {len(selected_features)} features")
        
        # Grid search for hyperparameter tuning
        print("Performing hyperparameter tuning...")
        param_grid = {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [None, 10, 20],
            'classifier__min_samples_split': [2, 5, 10]
        }
        
        # Create a pipeline with preprocessing and model
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(random_state=42))
        ])
        
        # Use grid search to find the best parameters
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=3, n_jobs=-1, verbose=1, scoring='f1_weighted'
        )
        
        # Train the model with grid search
        grid_search.fit(X_train, y_train)
        
        # Get the best model
        best_model = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
        
        # Evaluate the model
        y_pred = best_model.predict(X_test)
        accuracy = best_model.score(X_test, y_test)
        print(f"Model trained with accuracy: {accuracy:.2f}")
        print("Classification report:")
        print(classification_report(y_test, y_pred))
        
        # Save the model
        self.model = best_model
        joblib.dump(self.model, self.model_path)
        print(f"Model saved to {self.model_path}")
    
    def _generate_synthetic_data(self, n_samples=1000):
        """
        Generate synthetic network traffic data for training.
        In a real application, this would be replaced with actual network data.
        
        Returns:
            X (numpy.ndarray): Feature matrix
            y (numpy.ndarray): Target vector (0: normal, 1: attack)
        """
        # Initialize feature matrix and target vector
        X = np.zeros((n_samples, len(self.feature_names)))
        y = np.zeros(n_samples)
        
        # Calculate how many features we have now
        n_features = len(self.feature_names)
        
        # Check if we have the expected number of features
        if X.shape[1] != n_features:
            print(f"Warning: Feature matrix has {X.shape[1]} columns but {n_features} feature names")
        
        # Generate normal traffic (70% of samples)
        n_normal = int(0.7 * n_samples)
        
        # Normal traffic patterns
        X[:n_normal, 0] = np.random.normal(500, 200, n_normal)  # packet_size
        X[:n_normal, 1] = np.random.normal(10, 5, n_normal)     # packet_rate
        X[:n_normal, 2] = np.random.randint(1024, 65535, n_normal)  # source_port
        X[:n_normal, 3] = np.random.choice([80, 443, 22, 25, 53], n_normal)  # destination_port
        X[:n_normal, 4] = np.random.randint(0, 64, n_normal)    # tcp_flags
        X[:n_normal, 5] = np.random.normal(100, 50, n_normal)   # udp_length
        X[:n_normal, 6] = np.random.randint(0, 10, n_normal)    # icmp_type
        X[:n_normal, 7] = np.random.normal(30, 20, n_normal)    # connection_duration
        X[:n_normal, 8] = np.random.normal(2000, 1000, n_normal)  # bytes_in
        X[:n_normal, 9] = np.random.normal(1000, 500, n_normal)   # bytes_out
        X[:n_normal, 10] = np.random.normal(20, 10, n_normal)     # packets_in
        X[:n_normal, 11] = np.random.normal(15, 8, n_normal)      # packets_out
        
        # Generate attack traffic (30% of samples)
        n_attack = n_samples - n_normal
        
        # DDoS attack patterns
        n_ddos = int(0.3 * n_attack)
        start_idx = n_normal
        end_idx = start_idx + n_ddos
        
        X[start_idx:end_idx, 0] = np.random.normal(300, 100, n_ddos)  # packet_size
        X[start_idx:end_idx, 1] = np.random.normal(100, 30, n_ddos)   # packet_rate (high)
        X[start_idx:end_idx, 2] = np.random.randint(1024, 65535, n_ddos)  # source_port
        X[start_idx:end_idx, 3] = np.random.choice([80, 443], n_ddos)  # destination_port
        X[start_idx:end_idx, 4] = np.random.randint(0, 64, n_ddos)    # tcp_flags
        X[start_idx:end_idx, 5] = np.random.normal(50, 20, n_ddos)    # udp_length
        X[start_idx:end_idx, 6] = np.random.randint(0, 10, n_ddos)    # icmp_type
        X[start_idx:end_idx, 7] = np.random.normal(5, 3, n_ddos)      # connection_duration (short)
        X[start_idx:end_idx, 8] = np.random.normal(500, 200, n_ddos)  # bytes_in (low)
        X[start_idx:end_idx, 9] = np.random.normal(5000, 2000, n_ddos)  # bytes_out (high)
        X[start_idx:end_idx, 10] = np.random.normal(5, 3, n_ddos)     # packets_in (low)
        X[start_idx:end_idx, 11] = np.random.normal(100, 50, n_ddos)  # packets_out (high)
        
        # Port scanning attack patterns
        n_scan = int(0.3 * n_attack)
        start_idx = n_normal + n_ddos
        end_idx = start_idx + n_scan
        
        X[start_idx:end_idx, 0] = np.random.normal(100, 50, n_scan)   # packet_size (small)
        X[start_idx:end_idx, 1] = np.random.normal(50, 20, n_scan)    # packet_rate (medium)
        X[start_idx:end_idx, 2] = np.random.randint(1024, 65535, n_scan)  # source_port
        X[start_idx:end_idx, 3] = np.random.randint(1, 1024, n_scan)  # destination_port (scanning)
        X[start_idx:end_idx, 4] = np.random.choice([2, 18], n_scan)   # tcp_flags (SYN, SYN-ACK)
        X[start_idx:end_idx, 5] = np.random.normal(0, 0, n_scan)      # udp_length (mostly 0)
        X[start_idx:end_idx, 6] = np.random.normal(0, 0, n_scan)      # icmp_type (mostly 0)
        X[start_idx:end_idx, 7] = np.random.normal(1, 0.5, n_scan)    # connection_duration (very short)
        X[start_idx:end_idx, 8] = np.random.normal(100, 50, n_scan)   # bytes_in (very low)
        X[start_idx:end_idx, 9] = np.random.normal(100, 50, n_scan)   # bytes_out (very low)
        X[start_idx:end_idx, 10] = np.random.normal(1, 0.5, n_scan)   # packets_in (very low)
        X[start_idx:end_idx, 11] = np.random.normal(1, 0.5, n_scan)   # packets_out (very low)
        
        # Other attack patterns (SQL injection, brute force, etc.)
        n_other = n_attack - n_ddos - n_scan
        start_idx = n_normal + n_ddos + n_scan
        end_idx = n_samples
        
        X[start_idx:end_idx, 0] = np.random.normal(800, 300, n_other)  # packet_size (large)
        X[start_idx:end_idx, 1] = np.random.normal(5, 3, n_other)      # packet_rate (low)
        X[start_idx:end_idx, 2] = np.random.randint(1024, 65535, n_other)  # source_port
        X[start_idx:end_idx, 3] = np.random.choice([80, 443, 8080, 3306], n_other)  # destination_port
        X[start_idx:end_idx, 4] = np.random.randint(0, 64, n_other)    # tcp_flags
        X[start_idx:end_idx, 5] = np.random.normal(200, 100, n_other)  # udp_length
        X[start_idx:end_idx, 6] = np.random.normal(0, 0, n_other)      # icmp_type (mostly 0)
        X[start_idx:end_idx, 7] = np.random.normal(60, 30, n_other)    # connection_duration (long)
        X[start_idx:end_idx, 8] = np.random.normal(3000, 1500, n_other)  # bytes_in (high)
        X[start_idx:end_idx, 9] = np.random.normal(1000, 500, n_other)   # bytes_out
        X[start_idx:end_idx, 10] = np.random.normal(30, 15, n_other)     # packets_in (high)
        X[start_idx:end_idx, 11] = np.random.normal(20, 10, n_other)     # packets_out
        
        # Set target values (0 for normal, 1 for attack)
        y[n_normal:] = 1
        
        # Add some noise to make it more realistic
        X += np.random.normal(0, 0.1, X.shape)
        
        return X, y
    
    def predict(self, packet_data):
        """
        Predict if a network packet is an attack.
        
        Args:
            packet_data (dict): Dictionary containing packet features
            
        Returns:
            dict: Prediction results including attack type and confidence
        """
        if self.model is None:
            self.load_model()
        
        # Extract features from packet data
        features = self._extract_features(packet_data)
        
        # Make prediction
        features_array = np.array([features])
        is_attack = self.model.predict(features_array)[0]
        confidence = np.max(self.model.predict_proba(features_array)[0])
        
        # Determine attack type based on features
        attack_type = self._determine_attack_type(features) if is_attack else None
        
        # Check for anomalies in the traffic pattern
        anomaly_score = self._check_for_anomalies(features)
        
        # If anomaly score is high but model didn't detect an attack, flag it as potential zero-day
        if not is_attack and anomaly_score > self.anomaly_threshold:
            is_attack = True
            attack_type = "Unknown (Potential Zero-Day)"
            confidence = anomaly_score
        
        # Store this detection in history for future anomaly detection
        self._update_detection_history(features, is_attack, attack_type, confidence)
        
        return {
            'is_attack': bool(is_attack),
            'attack_type': attack_type,
            'confidence': float(confidence),
            'severity': self._determine_severity(confidence, attack_type),
            'anomaly_score': float(anomaly_score) if anomaly_score else 0.0,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def _extract_features(self, packet_data):
        """
        Extract features from packet data.
        
        Args:
            packet_data (dict): Dictionary containing packet information
            
        Returns:
            list: List of features in the correct order
        """
        # In a real application, this would extract actual features from network packets
        # For demonstration, we'll use the provided packet data or generate random values
        
        features = []
        for feature in self.feature_names:
            if feature in packet_data:
                features.append(packet_data[feature])
            else:
                # Use a reasonable default or random value based on feature type
                if feature == 'packet_size':
                    features.append(random.randint(100, 1500))
                elif feature == 'packet_rate':
                    features.append(random.randint(1, 100))
                elif feature == 'source_port':
                    features.append(random.randint(1024, 65535))
                elif feature == 'destination_port':
                    features.append(random.choice([80, 443, 22, 25, 53, 8080]))
                elif feature == 'tcp_flags':
                    features.append(random.randint(0, 63))
                elif feature == 'udp_length':
                    features.append(random.randint(0, 500))
                elif feature == 'icmp_type':
                    features.append(random.randint(0, 10))
                elif feature == 'connection_duration':
                    features.append(random.randint(1, 100))
                elif feature == 'bytes_in':
                    features.append(random.randint(100, 5000))
                elif feature == 'bytes_out':
                    features.append(random.randint(100, 5000))
                elif feature == 'packets_in':
                    features.append(random.randint(1, 50))
                elif feature == 'packets_out':
                    features.append(random.randint(1, 50))
                
                # Advanced features
                elif feature == 'packet_interval':
                    features.append(random.uniform(0.001, 0.5))
                elif feature == 'packet_size_variance':
                    features.append(random.uniform(10, 500))
                elif feature == 'entropy':
                    features.append(random.uniform(0, 8))
                elif feature == 'flag_distribution':
                    features.append(random.uniform(0, 1))
                elif feature == 'port_distribution':
                    features.append(random.uniform(0, 1))
                elif feature == 'payload_pattern':
                    features.append(random.randint(0, 10))
                elif feature == 'time_of_day':
                    features.append(datetime.now().hour)
                elif feature == 'day_of_week':
                    features.append(datetime.now().weekday())
                elif feature == 'protocol_type':
                    features.append(random.choice([1, 6, 17]))  # ICMP, TCP, UDP
                elif feature == 'ttl_value':
                    features.append(random.randint(32, 255))
                elif feature == 'window_size':
                    features.append(random.randint(1024, 65535))
                elif feature == 'urgent_pointer':
                    features.append(random.randint(0, 1))
                else:
                    features.append(0)
        
        return features
    
    def _determine_attack_type(self, features):
        """
        Determine the type of attack based on features.
        
        Args:
            features (list): List of packet features
            
        Returns:
            str: Attack type
        """
        # In a real application, this would use a more sophisticated approach
        # For demonstration, we'll use simple heuristics based on the features
        
        packet_size = features[0]
        packet_rate = features[1]
        source_port = features[2]
        dest_port = features[3]
        tcp_flags = features[4]
        udp_length = features[5]
        icmp_type = features[6]
        conn_duration = features[7]
        bytes_in = features[8]
        bytes_out = features[9]
        packets_in = features[10]
        packets_out = features[11]
        
        # DDoS attack: high packet rate, high outgoing traffic
        if packet_rate > 50 and packets_out > 50:
            return "DDoS"
            
        # DoS attack: high packet rate to a single destination
        if packet_rate > 40 and packets_out > 40 and conn_duration < 10:
            return "DoS"
        
        # Port scanning: many different destination ports, short connections
        if conn_duration < 5 and dest_port < 1024:
            return "Port Scanning"
            
        # SYN Flood: many SYN packets (TCP flag 2)
        if tcp_flags == 2 and packet_rate > 30:
            return "SYN Flood"
            
        # UDP Flood: high UDP traffic
        if udp_length > 0 and packet_rate > 30:
            return "UDP Flood"
            
        # ICMP Flood (Ping Flood): high ICMP traffic
        if icmp_type > 0 and packet_rate > 20:
            return "Ping Flood"
            
        # TCP Reset Attack: many RST packets (TCP flag 4)
        if tcp_flags == 4 and packet_rate > 10:
            return "TCP Reset Attack"
            
        # ARP Spoofing: specific packet size and high rate
        if packet_size < 100 and packet_rate > 5:
            return "ARP Spoofing"
            
        # DNS Spoofing: specific port (53) and unusual traffic
        if dest_port == 53 and bytes_out > bytes_in * 2:
            return "DNS Spoofing"
            
        # IP Spoofing: unusual source/destination patterns
        if source_port < 1024 and packet_size > 1000:
            return "IP Spoofing"
            
        # MAC Spoofing: specific packet characteristics
        if packet_size < 80 and conn_duration < 2:
            return "MAC Spoofing"
            
        # DHCP Spoofing: specific port (67/68)
        if dest_port in [67, 68] and bytes_out > 500:
            return "DHCP Spoofing"
            
        # Session Hijacking: unusual traffic patterns
        if conn_duration > 30 and bytes_in > 3000 and bytes_out < 500:
            return "Session Hijacking"
            
        # Replay Attack: repeated identical packets
        if packet_rate > 10 and bytes_in == bytes_out and packets_in == packets_out:
            return "Replay Attack"
            
        # Smurf Attack: broadcast ICMP
        if icmp_type > 0 and packet_size < 100 and packet_rate > 15:
            return "Smurf Attack"
            
        # ICMP Redirect: specific ICMP type (5)
        if icmp_type == 5:
            return "ICMP Redirect Attack"
            
        # Deauth Attack: specific characteristics (wireless)
        if packet_size < 50 and packet_rate > 8:
            return "Deauth Attack"
            
        # NetBIOS Enumeration: specific ports (137-139)
        if dest_port in [137, 138, 139] and conn_duration > 10:
            return "NetBIOS Enumeration"
            
        # Network Worm: rapidly increasing connections
        if packet_rate > 20 and conn_duration < 1 and dest_port > 1024:
            return "Network Worm"
        
        # Default to MITM if no specific pattern is matched
        return "Man-in-the-Middle"
    
    def _determine_severity(self, confidence, attack_type):
        """
        Determine the severity of an attack based on confidence and type.
        
        Args:
            confidence (float): Prediction confidence
            attack_type (str): Type of attack
            
        Returns:
            str: Severity level (Low, Medium, High, Critical)
        """
        # Base severity on confidence
        if confidence < 0.6:
            base_severity = "Low"
        elif confidence < 0.75:
            base_severity = "Medium"
        elif confidence < 0.9:
            base_severity = "High"
        else:
            base_severity = "Critical"
        
        # Adjust based on attack type
        if attack_type == "DDoS" and base_severity in ["High", "Critical"]:
            return "Critical"
        elif attack_type == "SQL Injection" and base_severity in ["Medium", "High"]:
            return "High"
        
        return base_severity


    def _check_for_anomalies(self, features):
        """
        Check if the current packet features represent an anomaly compared to historical data.
        
        Args:
            features (list): List of packet features
            
        Returns:
            float: Anomaly score between 0 and 1, higher means more anomalous
        """
        if not self.detection_history:
            return None
        
        # Calculate distance to known patterns
        distances = []
        for historical_features, _, _, _ in self.detection_history:
            # Calculate Euclidean distance between current features and historical features
            distance = np.sqrt(np.sum([(a - b) ** 2 for a, b in zip(features, historical_features)]))
            distances.append(distance)
        
        # Normalize distances to [0, 1] range
        if distances:
            max_distance = max(distances)
            min_distance = min(distances)
            if max_distance > min_distance:
                normalized_distances = [(d - min_distance) / (max_distance - min_distance) for d in distances]
                # Average distance as anomaly score
                return sum(normalized_distances) / len(normalized_distances)
        
        return 0.5  # Default if we can't calculate
    
    def _update_detection_history(self, features, is_attack, attack_type, confidence):
        """
        Update the detection history with the current packet information.
        
        Args:
            features (list): List of packet features
            is_attack (bool): Whether the packet was classified as an attack
            attack_type (str): Type of attack detected
            confidence (float): Confidence of the prediction
        """
        # Add current detection to history
        self.detection_history.append((features, is_attack, attack_type, confidence))
        
        # Limit history size
        if len(self.detection_history) > self.max_history_size:
            self.detection_history = self.detection_history[-self.max_history_size:]
        
        # Save history periodically (every 100 detections)
        if len(self.detection_history) % 100 == 0:
            try:
                joblib.dump(self.detection_history, self.history_path)
            except Exception as e:
                print(f"Error saving detection history: {e}")

# Singleton instance
detector = NetworkAttackDetector()

def analyze_packet(packet_data):
    """
    Analyze a network packet to detect if it's an attack.
    
    Args:
        packet_data (dict): Dictionary containing packet information
        
    Returns:
        dict: Prediction results
    """
    return detector.predict(packet_data)
