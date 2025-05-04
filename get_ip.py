import netifaces

def get_local_ips():
    ips = []
    for interface in netifaces.interfaces():
        addrs = netifaces.ifaddresses(interface)
        # 获取 IPv4 地址
        if netifaces.AF_INET in addrs:
            for addr in addrs[netifaces.AF_INET]:
                ip = addr["addr"]
                if ip != "127.0.0.1":
                    ips.append(ip)
    return ips

print("本地 IP 列表:", get_local_ips())