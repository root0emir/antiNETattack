from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import random
import time
import threading
import json
import os
import datetime
from collections import deque
from ai_model import analyze_packet

# Copyright (c) 2025 root0emir. All rights reserved.

# antiNETattack is a tool developed by root0emir and is released under the MIT License.
# This software can be used, copied, modified, and distributed under the terms of the MIT License.
# However, the name of the software, logos, and other promotional elements are the exclusive property of the developer.

# The use of the software for commercial purposes or the unauthorized use of the project name/logos is prohibited.
# The license applies only to the source code and documentation of the software.

# This software is provided "as is" without any warranties. Users use it at their own risk.

#############################################################################
#  Securonis GNU/Linux Network and System Technologies Research Laboratory  #
#############################################################################


# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variables
monitoring = False
attack_data = {
    'total_packets': 0,
    'attacks_detected': 0,
    'attack_types': {},
    'recent_attacks': [],
    'anomalies_detected': 0,
    'detection_history': [],
    'traffic_patterns': {}
}
monitor_thread = None

# Store last 1000 packets for analysis
packet_history = deque(maxlen=1000)

# Store detection logs
log_directory = 'logs'
os.makedirs(log_directory, exist_ok=True)
log_file = os.path.join(log_directory, f'detection_log_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.json')

# Load AI model
def load_or_create_model():
    # The model is loaded automatically when the ai_model module is imported
    return {'name': 'Network Attack Detection Model', 'version': '1.0'}

lock = threading.Lock()

def simulate_network_traffic():
    """
    Simulate network traffic for demonstration purposes.
    In a real application, this would be replaced with actual packet capture.
    """
    global attack_data, monitoring, packet_history
    
    # Common IP ranges for legitimate traffic
    internal_ips = [f'192.168.1.{i}' for i in range(1, 20)]  # Internal network
    common_external_ips = [
        '8.8.8.8', '8.8.4.4',  # Google DNS
        '1.1.1.1', '1.0.0.1',  # Cloudflare DNS
        '208.67.222.222',      # OpenDNS
        '172.217.169.78',      # Google
        '31.13.72.36',         # Facebook
        '13.107.42.14',        # Microsoft
    ]
    
    # Common ports for legitimate traffic
    common_ports = [80, 443, 53, 22, 25, 143, 993, 110, 995, 587]
    
    while monitoring:
        # Determine if this is normal traffic (80%) or potentially suspicious (20%)
        is_normal = random.random() > 0.20
        
        # Simulate a network packet with more realistic features
        current_time = time.time()
        protocol = random.choice(['TCP', 'UDP', 'ICMP']) if is_normal else random.choice(['TCP', 'UDP', 'ICMP', 'TCP', 'TCP'])
        
        # Source and destination IPs
        if is_normal:
            source_ip = random.choice(internal_ips)
            dest_ip = random.choice(common_external_ips)
        else:
            source_ip = f'{random.randint(1, 223)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}'
            dest_ip = random.choice(internal_ips)
        
        # Create more realistic packet data based on protocol
        packet = {
            'timestamp': current_time,
            'source_ip': source_ip,
            'destination_ip': dest_ip,
            'source_port': random.randint(1024, 65535) if is_normal else random.randint(1, 65535),
            'destination_port': random.choice(common_ports) if is_normal else random.choice([80, 443, 22, 25, 53, 8080, 3389, 445, 139, 21, 23, 3306, 5432, 27017, 6379]),
            'protocol': protocol,
            'packet_size': random.randint(64, 1500) if is_normal else random.randint(40, 9000),
            'packet_rate': random.randint(1, 20) if is_normal else random.randint(10, 100),
        }
        
        # Add protocol-specific fields
        if protocol == 'TCP':
            packet['tcp_flags'] = random.randint(0, 63)
            packet['window_size'] = random.randint(1024, 65535)
            packet['urgent_pointer'] = random.randint(0, 1)
            packet['flags'] = random.choice(['SYN', 'ACK', 'SYN-ACK', 'FIN', 'RST', 'PSH'])
        elif protocol == 'UDP':
            packet['udp_length'] = random.randint(8, 1472)
            packet['flags'] = ''
        elif protocol == 'ICMP':
            packet['icmp_type'] = random.randint(0, 10)
            packet['flags'] = ''
        
        # Add connection-related fields
        packet['connection_duration'] = random.randint(1, 100)
        packet['bytes_in'] = random.randint(100, 5000)
        packet['bytes_out'] = random.randint(100, 5000)
        packet['packets_in'] = random.randint(1, 50)
        packet['packets_out'] = random.randint(1, 50)
        
        # Add advanced features
        packet['packet_interval'] = random.uniform(0.001, 0.5)
        packet['packet_size_variance'] = random.uniform(10, 500)
        packet['entropy'] = random.uniform(0, 8)
        packet['flag_distribution'] = random.uniform(0, 1)
        packet['port_distribution'] = random.uniform(0, 1)
        packet['payload_pattern'] = random.randint(0, 10)
        packet['time_of_day'] = datetime.datetime.now().hour
        packet['day_of_week'] = datetime.datetime.now().weekday()
        packet['protocol_type'] = 6 if protocol == 'TCP' else (17 if protocol == 'UDP' else 1)  # TCP=6, UDP=17, ICMP=1
        packet['ttl_value'] = random.randint(32, 255)
        
        # Add to packet history
        packet_history.append(packet)
        
        # Analyze the packet
        result = analyze_packet(packet)
        
        # Update statistics
        with lock:
            attack_data['total_packets'] += 1
            
            # Update traffic patterns
            hour = datetime.datetime.now().hour
            if hour not in attack_data['traffic_patterns']:
                attack_data['traffic_patterns'][hour] = {'total': 0, 'attacks': 0}
            attack_data['traffic_patterns'][hour]['total'] += 1
            
            if result['is_attack']:
                attack_data['attacks_detected'] += 1
                attack_data['traffic_patterns'][hour]['attacks'] += 1
                
                # Check if it's an anomaly/zero-day
                if 'Unknown' in result['attack_type']:
                    attack_data['anomalies_detected'] += 1
                
                # Update attack type statistics
                attack_type = result['attack_type']
                if attack_type in attack_data['attack_types']:
                    attack_data['attack_types'][attack_type] += 1
                else:
                    attack_data['attack_types'][attack_type] = 1
                
                # Add to recent attacks with more details
                attack_info = {
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'source_ip': packet['source_ip'],
                    'destination_ip': packet['destination_ip'],
                    'source_port': packet['source_port'],
                    'destination_port': packet['destination_port'],
                    'protocol': packet['protocol'],
                    'attack_type': attack_type,
                    'confidence': result['confidence'],
                    'severity': result['severity'],
                    'anomaly_score': result.get('anomaly_score', 0),
                    'packet_size': packet['packet_size'],
                    'flags': packet.get('flags', '')
                }
                attack_data['recent_attacks'].insert(0, attack_info)
                
                # Keep only the most recent 100 attacks
                if len(attack_data['recent_attacks']) > 100:
                    attack_data['recent_attacks'] = attack_data['recent_attacks'][:100]
                
                # Log attack to file
                with open(log_file, 'a') as f:
                    f.write(json.dumps(attack_info) + '\n')
            
            # Add to detection history
            detection_entry = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'is_attack': result['is_attack'],
                'attack_type': result['attack_type'] if result['is_attack'] else None,
                'confidence': result['confidence'],
                'source_ip': packet['source_ip'],
                'destination_ip': packet['destination_ip']
            }
            attack_data['detection_history'].append(detection_entry)
            if len(attack_data['detection_history']) > 1000:
                attack_data['detection_history'] = attack_data['detection_history'][-1000:]
        
        # More realistic timing
        # Normal traffic is more spaced out, potential attacks might be more rapid
        if is_normal:
            time.sleep(random.uniform(0.3, 0.8))
        else:
            time.sleep(random.uniform(0.05, 0.2))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/toggle-monitoring', methods=['POST'])
def toggle_monitoring():
    """
    Start or stop network monitoring.
    """
    global monitoring, monitor_thread
    
    monitoring = not monitoring
    
    if monitoring:
        # Start monitoring in a new thread
        monitor_thread = threading.Thread(target=simulate_network_traffic)
        monitor_thread.daemon = True
        monitor_thread.start()
        message = "Monitoring started"
        print(f"[INFO] Network monitoring started at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        message = "Monitoring stopped"
        print(f"[INFO] Network monitoring stopped at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return jsonify({'status': 'success', 'monitoring': monitoring, 'message': message})

@app.route('/api/stats')
def get_stats():
    """
    Get current attack statistics.
    """
    global attack_data
    
    with lock:
        # Calculate additional statistics
        stats = dict(attack_data)
        
        # Calculate attack rate (percentage of packets that are attacks)
        if stats['total_packets'] > 0:
            stats['attack_rate'] = (stats['attacks_detected'] / stats['total_packets']) * 100
        else:
            stats['attack_rate'] = 0
            
        # Calculate anomaly rate
        if stats['attacks_detected'] > 0:
            stats['anomaly_rate'] = (stats['anomalies_detected'] / stats['attacks_detected']) * 100
        else:
            stats['anomaly_rate'] = 0
            
        # Get top attack sources
        source_ips = {}
        for attack in stats['recent_attacks']:
            ip = attack['source_ip']
            if ip in source_ips:
                source_ips[ip] += 1
            else:
                source_ips[ip] = 1
        
        stats['top_sources'] = sorted(source_ips.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return jsonify(stats)

@app.route('/api/monitoring-status')
def monitoring_status():
    """
    Get current monitoring status.
    """
    global monitoring, attack_data
    
    with lock:
        status = {
            'monitoring': monitoring,
            'total_packets': attack_data['total_packets'],
            'attacks_detected': attack_data['attacks_detected'],
            'anomalies_detected': attack_data.get('anomalies_detected', 0)
        }
    
    return jsonify(status)

@app.route('/api/packet-analysis')
def packet_analysis():
    """
    Get detailed packet analysis data for advanced visualizations.
    """
    global packet_history
    
    # Extract useful data from packet history
    protocols = {}
    ports = {}
    packet_sizes = []
    source_ips = {}
    
    for packet in packet_history:
        # Count protocols
        proto = packet['protocol']
        protocols[proto] = protocols.get(proto, 0) + 1
        
        # Count destination ports
        port = packet['destination_port']
        ports[port] = ports.get(port, 0) + 1
        
        # Collect packet sizes
        packet_sizes.append(packet['packet_size'])
        
        # Count source IPs
        ip = packet['source_ip']
        source_ips[ip] = source_ips.get(ip, 0) + 1
    
    # Calculate packet size distribution
    size_ranges = {'0-100': 0, '101-500': 0, '501-1000': 0, '1001-1500': 0}
    for size in packet_sizes:
        if size <= 100:
            size_ranges['0-100'] += 1
        elif size <= 500:
            size_ranges['101-500'] += 1
        elif size <= 1000:
            size_ranges['501-1000'] += 1
        else:
            size_ranges['1001-1500'] += 1
    
    return jsonify({
        'protocols': protocols,
        'top_ports': dict(sorted(ports.items(), key=lambda x: x[1], reverse=True)[:10]),
        'packet_size_distribution': size_ranges,
        'top_source_ips': dict(sorted(source_ips.items(), key=lambda x: x[1], reverse=True)[:10]),
        'packet_count': len(packet_history)
    })

@app.route('/api/reset')
def reset_stats():
    """
    Reset attack statistics.
    """
    global attack_data, packet_history
    
    with lock:
        attack_data = {
            'total_packets': 0,
            'attacks_detected': 0,
            'attack_types': {},
            'recent_attacks': [],
            'anomalies_detected': 0,
            'detection_history': [],
            'traffic_patterns': {}
        }
        packet_history.clear()
    
    # Create a new log file
    global log_file
    log_file = os.path.join(log_directory, f'detection_log_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    
    return jsonify({'status': 'success', 'message': 'Statistics reset successfully'})

if __name__ == '__main__':
    # Load the model
    model = load_or_create_model()
    
    # Create necessary directories
    os.makedirs('static', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)
