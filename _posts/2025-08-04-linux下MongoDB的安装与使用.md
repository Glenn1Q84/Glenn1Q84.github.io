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



## 导出与备份数据

#### 导出为mongodb格式

```
# 导出数据
mongodump --uri="mongodb://localhost:27017/test" --out="/mnt/sdb1/liuhu/mongodb_output"
# 恢复数据
mongorestore --uri="mongodb://localhost:27017" --dir="/mnt/sdb1/liuhu/mongodb_output/"

```

大批量文件分批导出

```
mongorestore --uri="mongodb://localhost:27017" \
             --dir="/mnt/sdb1/mongodb_output/" \
             --batchSize=100
```

### 导出为json



#### pymongo实现 批量导出并删除：无需认证

```
#!/usr/bin/env python3
import os
import pymongo
from pymongo import MongoClient
import json
import sys
from bson import json_util






def export_and_delete_collections(DB_NAME,HOST,OUTPUT_DIR,max_collections) :
    try :
        # 创建输出目录
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        # 连接 MongoDB
        client = MongoClient(HOST)
        db = client[DB_NAME]

        # 获取所有集合名
        collection_names = db.list_collection_names()

        # 限制只处理前5000个集合
        counter = 0


        print(f"找到 {len(collection_names)} 个集合，开始导出并删除前 {max_collections} 个集合...")

        for collection_name in collection_names :
            if counter >= max_collections :
                break

            print(f"正在处理 ({counter + 1}/{max_collections}): {collection_name}")

            try :
                collection = db[collection_name]

                # 检查集合是否为空
                if collection.count_documents({}) == 0 :
                    print(f"集合 {collection_name} 为空，直接删除")
                    db[collection_name].drop()
                    counter += 1
                    continue

                # 导出集合数据到JSON文件
                output_file = os.path.join(OUTPUT_DIR, f"{collection_name}.json")
                success = False

                with open(output_file, 'w', encoding='utf-8') as f :
                    # 使用批量查询提高效率
                    cursor = collection.find()
                    doc_count = 0

                    for doc in cursor :
                        # 使用BSON工具处理MongoDB特殊类型
                        json_str = json_util.dumps(doc)
                        f.write(json_str + '\n')
                        doc_count += 1

                print(f"导出成功: {doc_count} 个文档 -> {output_file}")

                # 验证文件是否创建且非空
                if os.path.exists(output_file) and os.path.getsize(output_file) > 0 :
                    # 删除集合
                    db[collection_name].drop()
                    print(f"成功删除集合: {collection_name}")
                    success = True
                else :
                    print(f"警告: 导出文件可能为空，跳过删除集合 {collection_name}")

            except Exception as e :
                print(f"错误: 处理集合 {collection_name} 时发生异常: {str(e)}")
                continue

            if success :
                counter += 1

            print("----------------------------------------")

        print(f"操作完成！导出的文件保存在: {OUTPUT_DIR}")
        print(f"成功处理 {counter} 个集合")
        return True

    except Exception as e :
        print(f"连接或处理过程中发生错误: {str(e)}")
        return False
    finally :
        # 关闭连接
        if 'client' in locals() :
            client.close()


if __name__ == "__main__" :
    # 配置参数
    # 指定要导出的db_name
    DB_NAME = "zhiu_yjh"
    # 数据库地址
    HOST = "localhost:27017"
    # 导出的保存路径
    OUTPUT_DIR = "E:/mongosh_output/tem"
    # 导出db_name下多少个collections
    max_collections = 2
    success = export_and_delete_collections(DB_NAME,HOST,OUTPUT_DIR,max_collections)
    sys.exit(0 if success else 1)
```



#### linux实现

批量导出

```
#!/bin/bash
DB_URI="mongodb://localhost:27017/video_info_new_1"
OUTPUT_DIR="/mnt/mongosh_output/数据库名"
mkdir -p "$OUTPUT_DIR"

# 获取所有集合名
collections=$(mongosh "$DB_URI" --quiet --eval "db.getCollectionNames().join('\n')")

echo "开始导出集合..."
for collection in $collections; do
    echo "正在导出: $collection"
    mongoexport --uri="$DB_URI" \
               --collection="$collection" \
               --limit=10 \
               --out="$OUTPUT_DIR/${collection}.json"
done
echo "导出完成！"
```



批量导入

```
#!/bin/bash
DB_URI="mongodb://localhost:27017/all_disease_video_info"
INPUT_DIR="/mnt/sdb1/liuhu/bilibili_all_disease/bvids_comment/tem"

echo "开始导入集合..."
for json_file in "$INPUT_DIR"/*.json; do
    # 提取集合名（去掉.json后缀）
    collection=$(basename "$json_file" .json)
    echo "正在导入: $collection"
    
    mongoimport --uri="$DB_URI" \
               --collection="$collection" \
               --file="$json_file"
               # 移除了 --jsonArray 参数
done
echo "导入完成！"
```



jsonarray

```
#!/bin/bash
DB_URI="mongodb://localhost:27017/all_disease_video_info"
INPUT_DIR="/mnt/sdb1/liuhu/bilibili_all_disease/bvids_comment/tem"

echo "开始导入集合..."
for json_file in "$INPUT_DIR"/*.json; do
    # 提取集合名（去掉.json后缀）
    collection=$(basename "$json_file" .json)
    echo "正在导入: $collection"
    
    mongoimport --uri="$DB_URI" \
               --collection="$collection" \
               --file="$json_file" \
               --jsonArray  # 恢复这个参数
done
echo "导入完成！"
```









批量导出并删除：无需认证

```
#!/bin/bash
DB_NAME="video_info_new_1"
HOST="localhost:27017"
OUTPUT_DIR="/mnt/sdb1/liuhu/mongo_data/analysis_comment"
mkdir -p "$OUTPUT_DIR"

# 获取所有集合名
collections=$(mongosh --host "$HOST" "$DB_NAME" --quiet --eval "db.getCollectionNames().join('\n')")

# 限制只处理前5000个集合
counter=0
echo "开始导出并删除前5000个集合..."
for collection in $collections; do
    if [ $counter -ge 10000 ]; then
        break
    fi
    
    echo "正在导出: $collection"
    # 导出集合
    mongoexport --host "$HOST" \
               --db "$DB_NAME" \
               --collection="$collection" \
               --out="$OUTPUT_DIR/${collection}.json"
    
    # 检查导出是否成功
    if [ $? -eq 0 ]; then
        echo "导出成功，正在删除集合: $collection"
        # 删除集合
        mongosh --host "$HOST" \
               "$DB_NAME" \
               --quiet \
               --eval "db.${collection}.drop()"
        
        if [ $? -eq 0 ]; then
            echo "成功删除集合: $collection"
        else
            echo "警告: 删除集合 $collection 失败"
        fi
    else
        echo "错误: 导出集合 $collection 失败，跳过删除"
    fi
    
    counter=$((counter + 1))
    echo "----------------------------------------"
done
echo "操作完成！导出的文件保存在: $OUTPUT_DIR"
```



批量导出并删除：用户认证登录

```
#!/bin/bash
USERNAME="wq"
PASSWORD="uxxi+>#6:@#o*dZpPAUso*a@eV)"
DB_NAME="video_info"
AUTH_DB="admin"
HOST="localhost:27017"
OUTPUT_DIR="/home/wq/tem/bilibili/13"
mkdir -p "$OUTPUT_DIR"

# 获取所有集合名
collections=$(mongosh --username "$USERNAME" --password "$PASSWORD" --authenticationDatabase "$AUTH_DB" --host "$HOST" "$DB_NAME" --quiet --eval "db.getCollectionNames().join('\n')")

# 限制只处理前10个集合
counter=0
echo "开始导出并删除前10个集合..."
for collection in $collections; do
    if [ $counter -ge 2000 ]; then
        break
    fi
    
    echo "正在导出: $collection"
    # 导出集合
    mongoexport --username "$USERNAME" \
               --password "$PASSWORD" \
               --authenticationDatabase "$AUTH_DB" \
               --host "$HOST" \
               --db "$DB_NAME" \
               --collection="$collection" \
               --out="$OUTPUT_DIR/${collection}.json"
    
    # 检查导出是否成功
    if [ $? -eq 0 ]; then
        echo "导出成功，正在删除集合: $collection"
        # 删除集合
        mongosh --username "$USERNAME" \
               --password "$PASSWORD" \
               --authenticationDatabase "$AUTH_DB" \
               --host "$HOST" \
               "$DB_NAME" \
               --quiet \
               --eval "db.${collection}.drop()"
        
        if [ $? -eq 0 ]; then
            echo "成功删除集合: $collection"
        else
            echo "警告: 删除集合 $collection 失败"
        fi
    else
        echo "错误: 导出集合 $collection 失败，跳过删除"
    fi
    
    counter=$((counter + 1))
    echo "----------------------------------------"
done
echo "操作完成！导出的文件保存在: $OUTPUT_DIR"
```





只导出不删除

```
#!/bin/bash
DB_NAME="video_info_new_1"
HOST="localhost:27017"
OUTPUT_DIR="/mnt/sdb1/liuhu/bilibili_all_disease/12"
mkdir -p "$OUTPUT_DIR"

# 获取所有集合名
collections=$(mongosh --host "$HOST" "$DB_NAME" --quiet --eval "db.getCollectionNames().join('\n')")

# 限制只处理前2000个集合
counter=0
total=0
echo "开始导出只导出不删除）..."
echo "========================================"

for collection in $collections; do
    if [ $counter -ge 10 ]; then
        break
    fi
    
    # 每500个集合输出一次进度
    if [ $((counter % 500)) -eq 0 ] && [ $counter -ne 0 ]; then
        echo "进度：已导出 $counter 个集合"
        echo "----------------------------------------"
    fi
    
    # 静默导出，不显示每个集合的进度
    mongoexport --host "$HOST" \
               --db "$DB_NAME" \
               --collection="$collection" \
               --out="$OUTPUT_DIR/${collection}.json" \
               --quiet > /dev/null 2>&1
    
    # 检查导出是否成功
    if [ $? -eq 0 ]; then
        total=$((total + 1))
    else
        echo "警告: 导出集合 $collection 失败"
    fi
    
    counter=$((counter + 1))
done

echo "========================================"
echo "操作完成！"
echo "成功导出 $total/$counter 个集合"
echo "导出的文件保存在: $OUTPUT_DIR"
```









单个导入 大文件也没问题

```
mongoimport --uri "mongodb://localhost:27017" --db zhihu_new --collection answer_answer_article_7topic --file /mnt/sdb1/liuhu/zhihu/answer_answer_article_7topic.json
```



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
