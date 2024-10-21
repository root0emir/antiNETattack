# antiNETattack

Author
antiNETattack is created by "root0emir", a professional security researcher and ethical hacker.


antiNETattack: Advanced Network Intrusion Detection & Prevention Tool
Overview
antiNETattack is a professional, real-time network intrusion detection and prevention tool built for monitoring network traffic and detecting various types of network-based attacks. It automatically detects and blocks malicious activity such as port scanning, DoS attacks, ARP spoofing, DNS spoofing, and malicious HTTP uploads.

Features
Real-Time Detection: Monitors live network traffic for suspicious activities.
Auto-blocking: Automatically blocks suspicious IP addresses using iptables.
Attack Detection Modules:
Port Scanning
DoS (SYN Flood)
ARP Spoofing
DNS Spoofing
ICMP Flood
Malicious HTTP .exe Uploads
Real-time Dashboard: Continuously updated attack statistics on the terminal.
Alerting System: Sends real-time alerts via Slack and Email.
Report Generation: Exports attack statistics as CSV files for further analysis.
User-Friendly Interface: Menu-driven CLI with options for whitelist/blacklist management and reporting.

Installation 

git clone https://github.com/yourusername/antiNETattack.git
cd antiNETattack
pip install -r requirements.txt

Usage

python3 antiNETattack.py --help

To Start The Tool

python3 antiNETattack.py --start

To monitor a specific interface and add whitelist/blacklist entries:

python3 antiNETattack.py --interface eth0 --whitelist 192.168.1.10 192.168.1.20 --start
