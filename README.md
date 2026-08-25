# RK1828 Qwen3 Embedding Reranker Service

本服务包含两个常驻 C++ JSONL 进程：Qwen3 Embedding 与 Qwen3 Reranker。
它们供 LightRAG 的向量化和重排阶段调用；另外两张 RK1828 保留给两卡
Qwen3.5-9B。

## 设备分配

| Device ID | 进程 |
| --- | --- |
| 0001:11:00.0 | Qwen3.5-9B stage 0 |
| 0003:31:00.0 | Qwen3.5-9B stage 1 |
| 0004:41:00.0 | Embedding 与 Reranker |

Embedding 和 Reranker 当前都使用第三张卡，配置中的 core_mask 为 0xff。
二者可同时常驻，但请求应由上层网关串行发送，避免争用同一张卡的 NPU 核。

## 构建

~~~
cmake -S . -B build -DRKNN3_MODEL_ZOO_ROOT=/path/to/rknn3-model-zoo
cmake --build build -j
~~~

也可使用 scripts/build-rk1828.sh；它需要 RKNN3_MODEL_ZOO_ROOT、
RK1828_C_COMPILER 与 RK1828_CXX_COMPILER 三个环境变量。

## 打包与部署

~~~
./scripts/build-rk1828.sh
./scripts/package-rk1828.sh
./scripts/deploy-adb.sh <adb-serial>
./scripts/verify-adb.sh <adb-serial>
~~~

打包产物位于 dist/RK1828-qwen3-embedding-reranker-service。部署脚本不会重启
model_gateway；只有确认网关配置已切换到 --config 命令后，才应重启网关。

## 配置与常驻行为

两个二进制均读取 config/config.json。配置分别包含 embedding 与 reranker
的模型、权重、Tokenizer、Embedding 表、device_id 和 core_mask。

device_id 会直接传给 rknn3_init_extend；不再读取 RKNN3_DEVICE_ID 环境变量。

启动后固定执行：

~~~
读取 config.json
→ 初始化 RKNN context、模型、Tokenizer、Embedding 表、session 一次
→ 输出 ready JSON
→ 持续读取 JSONL 请求
→ stdin 关闭时释放资源并退出
~~~

## 板端启动

模型目录须包含 .rknn、.weight、.tokenizer.gguf 和 .embed.bin 文件。复制
config.json 到板端部署目录后：

~~~
export LD_LIBRARY_PATH=/userdata/RK1828-qwen3-embedding-reranker-service/lib

./bin/rk1828_embedding_daemon --config ./config.json
./bin/rk1828_reranker_daemon --config ./config.json
~~~

Embedding 请求（单行 JSON）：

~~~json
{"id":"embed-1","input":["检索增强生成","向量检索"]}
~~~

Reranker 请求：

~~~json
{"id":"rank-1","query":"什么是向量检索？","documents":["向量检索通过相似度在嵌入空间查找文本。","今天北京天气晴朗。"]}
~~~

Reranker 返回与输入 documents 顺序一致的原始 relevance logits，仅用于排序，
不能将其视为已校准概率。
