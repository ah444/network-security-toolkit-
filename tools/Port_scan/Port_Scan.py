import os
import socket
import ipaddress
import ssl
import subprocess
import re
import platform
import urllib.parse
import requests
import json
import csv
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count

# Logs progress 
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Determine the directory of the current file and set the path to api_keys.txt
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
API_KEYS_PATH = os.path.join(BASE_DIR, "api_keys.txt")

# Read API Keys from the determined path
def read_api_keys(filename=API_KEYS_PATH):
    keys = {}
    try:
        with open(filename, "r") as f:
            for line in f:
                line = line.strip()
                if not line or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                keys[key.strip()] = value.strip()
    except Exception as e:
        logging.error("Error reading API keys from file: %s", e)
    return keys

api_keys = read_api_keys()

# Setup API Keys
OPENROUTER_API_KEY = api_keys.get("openrouter", "")
ABUSEIPDB_API_KEY = api_keys.get("abuseipdb", "")

# OpenRouter/Deepseek 
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_ENGINE = "deepseek/deepseek-chat:free"
TEMPERATURE = 0.5
TOKEN_LIMIT = 2048

# HTTP session
session = requests.Session()

# Network Recon section

def get_vulnerability_data_from_ai(banner):
    """
    Call the Deepseek API to retrieve a vulnerability report based on the provided banner.
    """
    if not banner:
        return ""
    prompt = f"""
You are a cybersecurity expert specializing in penetration testing. After performing a port scan, you received the following service banner:
**Service Banner:** {banner}

Generate an organized report including all associated CVE identifiers (if available), vulnerabilities related to this service and version, their severity, and recommendations to mitigate them.
Do not include any signature in your output.
"""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": MODEL_ENGINE,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "max_tokens": TOKEN_LIMIT
    }
    try:
        response = session.post(
            OPENROUTER_API_URL,
            headers=headers,
            data=json.dumps(data),
            timeout=10
        )
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            return f"Error: {response.status_code}"
    except Exception as e:
        logging.error("Deepseek API error: %s", e)
        return "Deepseek API error"

cve_cache = {}  # Cache stores CVE results

# Search for CVE according to version with MITRE CVE
def search_cves(vendor_version):
    base_url = "https://cve.mitre.org/cgi-bin/cvekey.cgi?keyword="
    query = urllib.parse.quote(vendor_version)
    url = base_url + query
    try:
        response = session.get(url, timeout=5)
        if response.status_code == 200:
            page_text = response.text
            cve_list = re.findall(r"CVE-\d{4}-\d{4,7}", page_text)
            return sorted(set(cve_list))
        else:
            return []
    except Exception as e:
        logging.error("search_cves error for '%s': %s", vendor_version, e)
        return []

def search_cves_cash(vendor_version):
    # Caching 
    if vendor_version in cve_cache:
        return cve_cache[vendor_version]
    cve_list = search_cves(vendor_version)
    cve_cache[vendor_version] = cve_list
    return cve_list

def get_version_from_banner(banner):
    # Get version from banner
    lower_banner = banner.lower()
    if "protocol mismatch" in lower_banner:
        banner = banner.split("protocol mismatch")[0].strip()
    return banner

def clean_version(version):
    # Removing extra whitespace.
    return " ".join(version.split())

def is_valid_version_banner(banner):
    # Checking if it's a valid version banner according to the regex provided
    if not banner:
        return False
    if not banner[0].isalnum():
        return False
    return bool(re.search(r'\d+\.\d+', banner))

def check_abuse_ipdb(ip_address, max_age=90):
    # Check IP history if it is public IP
    url = "https://api.abuseipdb.com/api/v2/check"
    querystring = {
        "ipAddress": ip_address,
        "maxAgeInDays": str(max_age)
    }
    headers = {
        "Accept": "application/json",
        "Key": ABUSEIPDB_API_KEY
    }
    try:
        response = requests.get(url, headers=headers, params=querystring, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            logging.error("AbuseIPDB API error: %s", response.text)
            return {}
    except Exception as e:
        logging.error("AbuseIPDB API exception: %s", e)
        return {}

def print_abuseipdb_result(result):
    # Print in organized way result
    data = result.get("data", {})
    keys_order = [
        "ipAddress", "ipVersion", "isPublic", "countryCode",
        "domain", "hostnames", "abuseConfidenceScore", "totalReports",
        "numDistinctUsers", "isTor", "isWhitelisted", "usageType",
        "isp", "lastReportedAt"
    ]
    for key in keys_order:
        value = data.get(key, "N/A")
        if isinstance(value, list):
            value = ", ".join(value) if value else "N/A"
        print(f"{key} is {value}")

def resolve_domain(domain):
    # Resolve domain with IP
    try:
        return socket.gethostbyname(domain)
    except Exception as e:
        logging.error("Domain resolution error for %s: %s", domain, e)
        return None

def is_public_ip(ip):
    # Checker if this IP is public IP or not
    try:
        ip_obj = ipaddress.ip_address(ip)
        return ip_obj.is_global
    except Exception as e:
        logging.error("Error checking if IP is public for %s: %s", ip, e)
        return False

def get_banner_tcp(host, port, timeout=1):
    # Gets the banner for various common ports with specific methods for each service type
    try:
        # HTTP/HTTPS (80, 443, 8080, 8443)
        if port in (80, 443, 8080, 8443):
            request = f"GET / HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n"
            if port in (443, 8443):  # HTTPS ports
                context = ssl.create_default_context()
                s = socket.create_connection((host, port), timeout=timeout)
                s = context.wrap_socket(s, server_hostname=host)
            else:
                s = socket.create_connection((host, port), timeout=timeout)
            s.send(request.encode())
            banner_bytes = b""
            while True:
                data = s.recv(4096)
                if not data:
                    break
                banner_bytes += data
            s.close()
            banner = banner_bytes.decode('utf-8', errors='ignore')
            for line in banner.splitlines():
                if line.lower().startswith("server:"):
                    return line.split(":", 1)[1].strip()
            return banner.strip()
        
        # FTP (21)
        elif port == 21:
            s = socket.create_connection((host, port), timeout=timeout)
            banner_bytes = s.recv(1024)  # Get initial greeting
            s.close()
            return banner_bytes.decode('utf-8', errors='ignore').strip()
        
        # SSH (22)
        elif port == 22:
            s = socket.create_connection((host, port), timeout=timeout)
            banner_bytes = s.recv(1024)  # SSH sends banner immediately
            s.close()
            return banner_bytes.decode('utf-8', errors='ignore').strip()
        
        # SMTP (25, 587)
        elif port in (25, 587):
            s = socket.create_connection((host, port), timeout=timeout)
            banner_bytes = s.recv(1024)  # Get greeting
            # Send EHLO to get more info
            s.send(b'EHLO example.com\r\n')
            response = s.recv(1024)
            banner_bytes += response
            s.send(b'QUIT\r\n')
            s.close()
            return banner_bytes.decode('utf-8', errors='ignore').strip()
        
        # DNS (53)
        elif port == 53:
            try:
                # Try to get DNS server version using a dig-like query
                # This is a simple version query packet
                query = bytes.fromhex('00ff0100000100000000000007version04bind0000100003')
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.settimeout(timeout)
                s.sendto(query, (host, port))
                response = s.recv(1024)
                s.close()
                return f"DNS Server detected on port {port}"
            except Exception:
                return f"DNS Server detected on port {port}"
        
        # POP3 (110, 995)
        elif port in (110, 995):
            if port == 995:  # POP3S
                context = ssl.create_default_context()
                s = socket.create_connection((host, port), timeout=timeout)
                s = context.wrap_socket(s, server_hostname=host)
            else:
                s = socket.create_connection((host, port), timeout=timeout)
            banner_bytes = s.recv(1024)  # Get greeting
            s.send(b'QUIT\r\n')
            s.close()
            return banner_bytes.decode('utf-8', errors='ignore').strip()
        
        # IMAP (143, 993)
        elif port in (143, 993):
            if port == 993:  # IMAPS
                context = ssl.create_default_context()
                s = socket.create_connection((host, port), timeout=timeout)
                s = context.wrap_socket(s, server_hostname=host)
            else:
                s = socket.create_connection((host, port), timeout=timeout)
            banner_bytes = s.recv(1024)  # Get greeting
            s.send(b'a001 LOGOUT\r\n')
            s.close()
            return banner_bytes.decode('utf-8', errors='ignore').strip()
        
        # NTP (123)
        elif port == 123:
            # NTP v3 request in mode 6 (control message)
            ntp_query = bytes.fromhex('1b0304fa0001000000000000')
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(timeout)
            s.sendto(ntp_query, (host, port))
            try:
                response = s.recv(1024)
                s.close()
                return f"NTP Server detected on port {port}"
            except socket.timeout:
                s.close()
                return f"NTP Server detected on port {port} (no response)"
        
        # SNMP (161)
        elif port == 161:
            # SNMP v1 Get request for system description (1.3.6.1.2.1.1.1.0)
            snmp_query = bytes.fromhex('302602010004067075626c6963a0190201000201000201003011300f060b2b060102010101000500')
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(timeout)
            s.sendto(snmp_query, (host, port))
            try:
                response = s.recv(1024)
                s.close()
                return f"SNMP Server detected on port {port}"
            except socket.timeout:
                s.close()
                return f"SNMP Server detected on port {port} (no response)"
        
        # LDAP (389, 636)
        elif port in (389, 636):
            if port == 636:  # LDAPS
                context = ssl.create_default_context()
                s = socket.create_connection((host, port), timeout=timeout)
                s = context.wrap_socket(s, server_hostname=host)
            else:
                s = socket.create_connection((host, port), timeout=timeout)
            # Simple LDAP bind request
            ldap_bind = bytes.fromhex('300c0201016007020103040080')
            s.send(ldap_bind)
            try:
                response = s.recv(1024)
                s.close()
                return f"LDAP Server detected on port {port}"
            except Exception:
                s.close()
                return f"LDAP Server detected on port {port}"
        
        # MySQL (3306)
        elif port == 3306:
            s = socket.create_connection((host, port), timeout=timeout)
            banner_bytes = s.recv(1024)  # Get greeting
            s.close()
            return banner_bytes.decode('utf-8', errors='ignore').strip()
        
        # PostgreSQL (5432)
        elif port == 5432:
            s = socket.create_connection((host, port), timeout=timeout)
            # PostgreSQL startup message
            startup = bytes.fromhex('00000008000003e0')
            s.send(startup)
            try:
                response = s.recv(1024)
                s.close()
                return f"PostgreSQL Server detected on port {port}"
            except Exception:
                s.close()
                return f"PostgreSQL Server detected on port {port}"
        
        # Redis (6379)
        elif port == 6379:
            s = socket.create_connection((host, port), timeout=timeout)
            # Send INFO command
            s.send(b'INFO\r\n')
            try:
                response = s.recv(1024)
                s.close()
                return response.decode('utf-8', errors='ignore').strip()
            except Exception:
                s.close()
                return f"Redis Server detected on port {port}"
        
        # MongoDB (27017)
        elif port == 27017:
            s = socket.create_connection((host, port), timeout=timeout)
            # MongoDB ismaster command
            ismaster = bytes.fromhex('3a000000ffffffff0000000000000000000f69736d6173746572000100000000')
            try:
                s.send(ismaster)
                response = s.recv(1024)
                s.close()
                return f"MongoDB Server detected on port {port}"
            except Exception:
                s.close()
                return f"MongoDB Server detected on port {port}"
        
        # MSSQL (1433)
        elif port == 1433:
            s = socket.create_connection((host, port), timeout=timeout)
            # Send a TDS pre-login packet
            prelogin = bytes.fromhex('1201002f000000001a000006010000001b000102001a0002000000000000000000')
            s.send(prelogin)
            banner_bytes = s.recv(1024)
            s.close()
            return f"MSSQL Server detected on port {port}"
        
        # RDP (3389)
        elif port == 3389:
            s = socket.create_connection((host, port), timeout=timeout)
            # Just detect the service, don't try to negotiate
            s.close()
            return f"RDP Server detected on port {port}"
        
        # Elasticsearch (9200, 9300)
        elif port in (9200, 9300):
            if port == 9200:  # REST API
                request = f"GET / HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n"
                s = socket.create_connection((host, port), timeout=timeout)
                s.send(request.encode())
                banner_bytes = b""
                while True:
                    data = s.recv(4096)
                    if not data:
                        break
                    banner_bytes += data
                s.close()
                return f"Elasticsearch REST API detected on port {port}"
            else:  # Transport protocol
                s = socket.create_connection((host, port), timeout=timeout)
                s.close()
                return f"Elasticsearch Transport Protocol detected on port {port}"
        
        # Memcached (11211)
        elif port == 11211:
            s = socket.create_connection((host, port), timeout=timeout)
            # Send version command
            s.send(b'version\r\n')
            try:
                response = s.recv(1024)
                s.close()
                return response.decode('utf-8', errors='ignore').strip()
            except Exception:
                s.close()
                return f"Memcached Server detected on port {port}"
        
        # Default approach for other ports
        else:
            s = socket.create_connection((host, port), timeout=timeout)
            s.send(b'\r\n')
            banner_bytes = s.recv(1024)
            s.close()
            return banner_bytes.decode('utf-8', errors='ignore').strip()
    except Exception as e:
        logging.debug("TCP banner retrieval failed for %s:%s - %s", host, port, e)
        return ""

def scan_tcp_port(host, port, timeout=1):
    # Scanning ports and determine whether it's open or closed (TCP)
    try:
        s = socket.create_connection((host, port), timeout=timeout)
        banner = get_banner_tcp(host, port, timeout)
        try:
            service = socket.getservbyport(port)
        except Exception:
            service = ""
        s.close()
        return port, "open", service, banner
    except Exception:
        return port, "closed", "", ""

def scan_udp_port(host, port, timeout=1):
    # Scan UDP ports 
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(timeout)
        s.sendto(b'\r\n', (host, port))
        try:
            data, _ = s.recvfrom(1024)
            banner = data.decode('utf-8', errors='ignore').strip()
            s.close()
            return port, "open", "udp", banner
        except socket.timeout:
            s.close()
            return port, "closed", "udp", ""
    except Exception as e:
        logging.debug("UDP scan failed for %s:%s - %s", host, port, e)
        return port, "closed", "udp", ""

def is_host_live(host, timeout=1):
    # Checker to determine if the host is live 
    try:
        param = "-n" if platform.system().lower() == "windows" else "-c"
        command = ["ping", param, "1", host]
        subprocess.check_output(command, stderr=subprocess.STDOUT, universal_newlines=True, timeout=timeout)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False

def detect_os_by_ping(host):
    # Detect operating system according to TTL
    try:
        param = "-n" if platform.system().lower() == "windows" else "-c"
        command = ["ping", param, "1", host]
        ping_output = subprocess.check_output(command, stderr=subprocess.STDOUT, universal_newlines=True)
    except subprocess.CalledProcessError:
        logging.warning("Ping failed for host %s", host)
        return

    match = re.search(r"TTL=(\d+)", ping_output, re.IGNORECASE)
    if match:
        ttl_value = int(match.group(1))
        if ttl_value == 128:
            print("Running OS: Windows")
        elif ttl_value == 64:
            print("Running OS: Linux/FreeBSD/OSX/Juniper/HP-UX")
        elif ttl_value == 255:
            print("IP belongs to a Cisco device")
        elif ttl_value == 254:
            print("Running OS: Solaris/AIX")
        elif ttl_value == 252:
            print("Running OS: Windows Server 2003/XP")
        elif ttl_value == 240:
            print("Running OS: Novell")
        elif ttl_value == 200:
            print("Running OS: HP-UX")
        elif ttl_value in (190, 127):
            print("Running OS: MacOS")
        elif ttl_value == 100:
            print("Running OS: IBM OS/2")
        elif ttl_value == 60:
            print("Running OS: AIX")
        elif ttl_value == 50:
            print("Running OS: Windows 95/98/ME")
        elif ttl_value == 48:
            print("Running OS: BSDI")
        elif ttl_value == 30:
            print("Running OS: SunOS")
        else:
            print("Unknown OS or device")
    else:
        print("Cannot detect OS (TTL not found)")

def print_table(results, title):
    # The representation of table to the user
    print(f"\n{title}")
    header = f"| {'PORT':^9} | {'STATE':^7} | {'SERVICE':^10} | {'BANNER':^40} |"
    separator = "+" + "-" * (len(header) - 2) + "+"
    print(separator)
    print(header)
    print(separator)
    for port, state, service, banner in results:
        port_field = f"{port}/tcp" if service != "udp" else f"{port}/udp"
        banner_field = banner if len(banner) <= 40 else banner[:37] + "..."
        print(f"| {port_field:^9} | {state:^7} | {service:^10} | {banner_field:^40} |")
    print(separator)


# Thread-safe progress counter class to avoid race conditions
class ProgressCounter:
    """Thread-safe counter for tracking scan progress"""
    def __init__(self):
        self._count = 0
        self._lock = threading.Lock()
    
    def increment(self):
        with self._lock:
            self._count += 1
    
    def get(self):
        with self._lock:
            return self._count
    
    def reset(self):
        with self._lock:
            self._count = 0


# Global progress counter instance
progress_counter = ProgressCounter()


def scan_wrapper(scan_func, *args):
    """Wrapper function that increments progress after scan completes"""
    result = scan_func(*args)
    progress_counter.increment()
    return result


def verbose_progress(total_tasks):
    """Verbose progress indicator that shows scan progress"""
    while True:
        count = progress_counter.get()
        progress = (count / total_tasks) * 100
        remaining = total_tasks - count
        sys.stdout.write(f"\rProgress: {progress:.2f}% complete | {remaining} tasks remaining")
        sys.stdout.flush()
        if count >= total_tasks:
            break
        time.sleep(2)
    print("\nProgress: 100.00% complete")

def export_results_to_json(results, filename="scan_results.json"):
    try:
        with open(filename, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Results exported to {filename}")
    except Exception as e:
        logging.error("Error exporting to JSON: %s", e)

def export_results_to_csv(results, filename="scan_results.csv"):
    try:
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Port", "Service", "Banner", "Version", "MITRE_CVEs", "Deepseek_Info"])
            for entry in results:
                writer.writerow([
                    entry.get("port"),
                    entry.get("service"),
                    entry.get("banner"),
                    entry.get("version"),
                    ", ".join(entry.get("mitre_cves", [])),
                    entry.get("deepseek_info")
                ])
        print(f"Results exported to {filename}")
    except Exception as e:
        logging.error("Error exporting to CSV: %s", e)

def export_results_to_txt(results, filename="scan_results.txt"):
    try:
        with open(filename, "w") as f:
            for entry in results:
                f.write(f"Port: {entry.get('port')}\n")
                f.write(f"Service: {entry.get('service')}\n")
                f.write(f"Banner: {entry.get('banner')}\n")
                f.write(f"Version: {entry.get('version')}\n")
                f.write(f"MITRE CVEs: {', '.join(entry.get('mitre_cves', []))}\n")
                f.write(f"Deepseek Info: {entry.get('deepseek_info')}\n")
                f.write("-" * 40 + "\n")
        print(f"Results exported to {filename}")
    except Exception as e:
        logging.error("Error exporting to TXT: %s", e)

# The main or first representation for the user

def main():
    # Directly starting with port scan
    start_time = time.time()
    target_input = input("Enter target IP or domain (e.g., 192.168.1.0/24 or example.com): ").strip()
    if "/" in target_input:
        resolved_ip = target_input
    else:
        try:
            ipaddress.ip_address(target_input)
            resolved_ip = target_input
        except ValueError:
            resolved_ip = resolve_domain(target_input)
        if resolved_ip:
            print(f"Domain {target_input} resolved to IP: {resolved_ip}")
        else:
            print("Failed to resolve domain. Exiting.")
            return

    try:
        start_port = int(input("Enter start port: "))
        end_port = int(input("Enter end port: "))
    except ValueError:
        print("Invalid port values.")
        return

    targets = []
    if "/" in target_input:
        try:
            network_obj = ipaddress.ip_network(target_input, strict=False)
            targets = [str(ip) for ip in network_obj.hosts()]
        except Exception as e:
            logging.error("Error parsing network: %s", e)
            return
    else:
        targets = [resolved_ip]

    protocol_choice = input("Select protocol for scanning (TCP/UDP/both): ").strip().lower()
    verbose_mode = input("Enable verbose mode? (Y/N): ").strip().lower() == "y"

    all_export_data = []

    for target in targets:
        target_ip = target
        if is_public_ip(target_ip):
            abuse_choice = input(f"Do you want to check the reputation of public IP {target_ip} using AbuseIPDB? (Y/N): ").strip().lower()
            if abuse_choice == "y":
                abuse_result = check_abuse_ipdb(target_ip)
                print(f"\nAbuseIPDB reputation info for {target_ip}:")
                print_abuseipdb_result(abuse_result)
        else:
            print(f"{target_ip} is not a public IP; skipping AbuseIPDB check.")

        if not is_host_live(target_ip):
            print(f"Host {target_ip} did not respond to ping.")
            if input("Do you want to scan this host anyway? (Y/N): ").strip().lower() != "y":
                continue

        print(f"\nScanning target: {target_ip}")
        print("Detecting OS via ping...")
        detect_os_by_ping(target_ip)

        scan_tasks = []
        for port in range(start_port, end_port + 1):
            if protocol_choice in ("tcp", "both"):
                scan_tasks.append(("tcp", port, scan_tcp_port, target_ip, port))
            if protocol_choice in ("udp", "both"):
                scan_tasks.append(("udp", port, scan_udp_port, target_ip, port))
        total_tasks = len(scan_tasks)
        
        # Reset the progress counter for this scan (thread-safe)
        progress_counter.reset()

        if verbose_mode:
            progress_thread = threading.Thread(target=verbose_progress, args=(total_tasks,), daemon=True)
            progress_thread.start()

        results = []
        max_workers = min(100, total_tasks, cpu_count() * 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(scan_wrapper, task[2], task[3], task[4], 1): task for task in scan_tasks
            }
            for future in as_completed(future_to_task):
                result = future.result()
                results.append(result)
        if verbose_mode:
            progress_thread.join()

        open_ports = [r for r in results if r[1] == "open"]

        if not open_ports:
            print(f"No open ports found on {target_ip}")
        else:
            print_table(open_ports, "OPEN PORTS")
            print("\nVulnerability Details per Open Port:")
            for port, state, service, banner in open_ports:
                if banner:
                    version = clean_version(get_version_from_banner(banner))
                    mitre_cves = search_cves_cash(version)
                    deepseek_info = get_vulnerability_data_from_ai(banner)
                else:
                    version = ""
                    mitre_cves = []
                    deepseek_info = ""
                print(f"\nPort: {port}")
                print(f"Service: {service}")
                print(f"Banner: {banner}")
                print(f"Version: {version}")
                print(f"MITRE CVEs: {', '.join(mitre_cves) if mitre_cves else 'None'}")
                print(f"Deepseek Info: {deepseek_info}")
                export_entry = {
                    "port": port,
                    "service": service,
                    "banner": banner,
                    "version": version,
                    "mitre_cves": mitre_cves,
                    "deepseek_info": deepseek_info
                }
                all_export_data.append(export_entry)

        print("=" * 60)

    end_time = time.time()
    elapsed = end_time - start_time
    print("\nScan Summary Report")
    print("-------------------")
    print(f"Total targets scanned: {len(targets)}")
    print(f"Total open ports found: {len(all_export_data)}")
    print(f"Elapsed time: {elapsed:.2f} seconds")
    
    if all_export_data:
        export_choice = input("Do you want to export the results to a file? (Y/N): ").strip().lower()
        if export_choice == "y":
            file_ext = input("Enter desired file extension (txt, json, csv): ").strip().lower()
            if file_ext == "json":
                export_results_to_json(all_export_data, filename="scan_results.json")
            elif file_ext == "csv":
                export_results_to_csv(all_export_data, filename="scan_results.csv")
            elif file_ext == "txt":
                export_results_to_txt(all_export_data, filename="scan_results.txt")
            else:
                print("Unknown extension. No export performed.")
    else:
        print("No open port results to export.")

if __name__ == "__main__":
    main()
