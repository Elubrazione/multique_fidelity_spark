# 测试一下encoder的功能是否正常
from sql_encoder.script import encode
import torch
import numpy as np

sqls = ["q1"]
sql_embeddings = []

for sql_id in sqls:
    with open(f"./data/test-sqls/tpch/{sql_id}.sql")as sql_file:
        sql = sql_file.read()
        sql_embeddings.append(encode(sql))

print(sql_embeddings)
sql_embeddings =  torch.tensor(sql_embeddings)
print(sql_embeddings)
sql_embedding = torch.max(sql_embeddings, dim=0)[0].tolist()
norm = np.linalg.norm(sql_embedding)
sql_embedding = list(map(lambda x: x / norm, sql_embedding))
print(sql_embedding)