import requests
import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# def create_embedding(text_list):
#     # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
#     r = requests.post("http://localhost:11434/api/embed", json={
#         "model": "bge-m3",
#         "input": text_list
#     })

#     embedding = r.json()["embeddings"] 
#     return embedding

def create_embedding(text_list):
    r = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "bge-m3",
            "input": text_list
        }
    )

    print("Status Code:", r.status_code)

    response = r.json()

    print(response)

    if "embeddings" not in response:
        raise Exception(response)

    return response["embeddings"]


jsons = os.listdir("jsons")  # List all the jsons 
my_dicts = []
chunk_id = 0

for json_file in jsons:
    with open(f"jsons/{json_file}") as f:
        content = json.load(f)
    print(f"Creating Embeddings for {json_file}")
    # embeddings = create_embedding([c['text'] for c in content['chunks']])
    # embeddings = create_embedding([c['text'] for c in content['chunks'][:10]])
    

    # print("Total chunks:", len(texts))
    # print("First chunk:", texts[0])
    # print("Second chunk:", texts[1])
    # embeddings = create_embedding([c['text'] for c in content['chunks']])

    texts = [c["text"] for c in content["chunks"]]

    print("Total chunks:", len(texts))
    print("First chunk:", texts[0])
    print("Second chunk:", texts[1])

    batch_size = 100
    embeddings = []

    for i in range(0, len(texts), batch_size):

        batch = texts[i:i + batch_size]

        print(f"Creating embeddings for chunks {i} to {i + len(batch) - 1}")

        batch_embeddings = create_embedding(batch)

        embeddings.extend(batch_embeddings)
        
    for i, chunk in enumerate(content['chunks']):
        chunk['chunk_id'] = chunk_id
        chunk['embedding'] = embeddings[i]
        chunk_id += 1
        my_dicts.append(chunk) 
        # if(i==3): # read 5 chunks from now 
        #     break
    # break
# print(my_dicts)

df = pd.DataFrame.from_records(my_dicts)
#save this data frame
joblib.dump(df,'embeddings.joblib')
