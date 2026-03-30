from pymilvus import MilvusClient

try:
    client = MilvusClient(uri="http://localhost:19530")
    print("连接成功")
    print(client.list_collections())
except Exception as e:
    print("未带认证连接失败：", e)

try:
    client = MilvusClient(uri="http://localhost:19530", token="root:Milvus")
    print("带认证连接成功")
    print(client.list_collections())
except Exception as e:
    print("带认证连接失败：", e)