#!/bin/bash
echo "=== 详细网络诊断 ==="

# 1. 获取 Windows IP
WIN_IP=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}')
echo "Windows IP: $WIN_IP"

# 2. 测试基础连通性
echo -n "Ping 测试: "
ping -c 2 -W 1 $WIN_IP >/dev/null 2>&1 && echo "✓ 成功" || echo "✗ 失败"

# 3. 测试端口扫描
echo -n "端口 3306: "
nc -zv $WIN_IP 3306 2>&1 | grep -q "succeeded" && echo "✓ 开放" || echo "✗ 关闭"

# 4. 检查其他常见端口
for port in 22 80 443 3306; do
    echo -n "端口 $port: "
    nc -zv $WIN_IP $port 2>&1 | grep -q "succeeded" && echo "开放" || echo "关闭"
done

# 5. 使用 telnet 测试
echo "Telnet 测试:"
timeout 2 telnet $WIN_IP 3306 2>&1 | head -5