---
title: Linux下MongoDB的安装与使用
tags: ["MongoDB", "服务器", "Linux","安装","使用","Debug", "数据存储"]
article_header:
  type: cover
  image:
    src:
---

# 前言

大多数情况下是用不到MongoDB的，直接存储在csv或者json即可。以下是比较适合用MongoDB存储的情况

- 抓数据
- 数据规模极大，起码百万+
- 频繁查找字段
- 文本格式的数据
- 字典格式的数据

不适用于

- 图像、视频、音频数据存储。这种建议采用字段索引的方式，将key和数据分离存储，将数据存储到单独存储到一个文件夹，其中文件名作为MongoDB的字段访问。



# Linux下MongoDB的安装与使用

## 安装

直接跟着以下教程一步一步安装就行，亲测有效

[在Ubuntu22.04中安装MongoDB6.0（2024年1月版）_厦大数据库实验室博客 (xmu.edu.cn)](https://dblab.xmu.edu.cn/blog/4594/)

## 常用命令

```
# 查看状态 
sudo systemctl status mongod
# 设置开机自启动 
sudo systemctl enable 服务名 
# 设置开机不启动 
sudo systemctl disable 服务名

# 重启，需要注意，每次重启后都会花几分钟，不会立马就能使用
sudo systemctl restart mongod
sudo systemctl stop mongod
sudo systemctl start mongod

```

## 设置公开访问

进入conf文件设置IP为 0.0.0.0



## 出现问题之后如何debug

使用以下命令查看日志信息

```
sudo journalctl -u mongod --no-pager -n 100 # 100是输出最近的100行日志
```

### 报错 "msg":"Failed to unlink socket file",

The logs indicate that MongoDB is failing to start due to an issue with the socket file `/tmp/mongodb-27017.sock`. Specifically, MongoDB is unable to unlink (remove) the socket file, which results in a fatal assertion and termination of the process.

```
{"t":{"$date":"2024-10-27T18:29:20.286+08:00"},"s":"E",  "c":"NETWORK",  "id":23024,   "ctx":"initandlisten","msg":"Failed to unlink socket file","attr":{"path":"/tmp/mongodb-27017.sock","error":"Operation not permitted"}}
```

**如何解决**

1. **Socket File is Locked or In Use:**

   - The socket file `/tmp/mongodb-27017.sock` might still be in use by another process or might be locked.
   - Ensure that no other MongoDB process is running that might be using this socket file.

   bash

   复制

   ```
   sudo lsof /tmp/mongodb-27017.sock
   ```

   - If another process is using the socket file, you may need to stop that process or resolve the conflict.

2. **Permissions Issue:**

   - The MongoDB process might not have the necessary permissions to remove the socket file.
   - Ensure that the MongoDB process has the necessary permissions to access and remove the socket file.

   bash

   复制

   ```
   sudo chown mongodb:mongodb /tmp/mongodb-27017.sock
   ```

3. **Manual Removal of Socket File:**

   - Manually remove the socket file if it exists and is not in use.

   ```
   sudo rm /tmp/mongodb-27017.sock
   ```



### 如何避免 Out of Memory杀死进程 问题

一方面是设置内存限制, 另一方面是设置自动重启

**1. 创建正确的override文件**：

```
sudo mkdir -p /etc/systemd/system/mongod.service.d
sudo tee /etc/systemd/system/mongod.service.d/override.conf <<'EOF'
[Service]
# 重启策略
Restart=on-failure
RestartSec=10s
StartLimitInterval=60s
StartLimitBurst=5

# 内存限制（核心配置）
MemoryLimit=32G
LimitAS=32G
LimitMEMLOCK=32G
OOMScoreAdjust=-200

# CPU限制
CPUQuota=80%

# 其他限制
LimitNOFILE=64000
LimitNPROC=64000
EOF
```

2.**强制重载systemd配置**：

```
sudo systemctl daemon-reexec
sudo systemctl daemon-reload
sudo systemctl restart mongod
```

**3. 验证配置生效**：

```
# 检查内存限制
cat /proc/$(pgrep mongod)/limits | grep -E 'Max address space|Max locked memory|memory limit'

# 检查重启策略
systemctl show mongod --property=Restart,RestartSec,StartLimitInterval,StartLimitBurst
```

**关键配置说明**

| 参数             | 值   | 作用                             |
| :--------------- | :--- | :------------------------------- |
| `LimitAS`        | 32G  | 虚拟内存硬限制（关键防护）       |
| `MemoryLimit`    | 32G  | cgroups内存限制                  |
| `OOMScoreAdjust` | -200 | 使MongoDB成为最后被OOM杀死的进程 |
| `RestartSec`     | 10s  | 避免频繁重启导致系统负载         |

4. **测试自动重启功能**



```
# 测试1：模拟崩溃
sudo kill -9 $(pgrep mongod) && \
sleep 15 && \
systemctl status mongod

# 测试2：查看重启日志
journalctl -u mongod --since "5 minutes ago" | grep -i 'Scheduled restart'
```

**哪些情况会重启**：

| 场景                 | 是否重启   | 日志特征                             |
| :------------------- | :--------- | :----------------------------------- |
| 被 OOM Killer 杀死   | ✅ 会重启   | `Main process killed (signal=KILL)`  |
| 手动 `kill -9`       | ✅ 会重启   | `Process exited, code=killed`        |
| `systemctl stop`     | ❌ 不重启   | `Stopped MongoDB...`                 |
| 程序崩溃             | ✅ 会重启   | `Segmentation fault`                 |
| 达到 StartLimitBurst | ❌ 停止重启 | `Start request repeated too quickly` |
