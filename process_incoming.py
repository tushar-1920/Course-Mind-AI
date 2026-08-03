# now using the joblib we will save the embeddings so that everytimee the embeddings donot load
#save this data frame
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
import requests
from openai import OpenAI
from config import api_key

client = OpenAI(api_key=api_key)

# df = joblib.load('embeddings.joblib')
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


def inference(prompt):
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2",
            "prompt": prompt,
            "stream" : False
        }
    )
    response = r.json()
    print(response)
    return response

def inference_openai(prompt):
    print("Thinking.....")
    response = client.responses.create(
    model="gpt-5",
    input=prompt
    )

    return response.output_text



df = joblib.load('embeddings.joblib')


incoming_query = input("Ask a question: ")
question_embedding = create_embedding([incoming_query])[0]

print(question_embedding)

similarity = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()
# print(similarity)
top_results = 5
max_indx = similarity.argsort()[::-1][0:top_results]
# print(max_indx)
new_df = df.loc[max_indx]
# print(new_df[["title","number","text"]])


prompt = f''''I am teaching wed development in WEB DEVELOPMENT COURSE. Here are the video subtitle chunks containing video title, video number, start time in seconds, end time in seconds, the text at that time :

{new_df[["title","number","start","end","text"]].to_json(orient="records")}
--------------------------
{incoming_query}
User asked the question realted to the video chunks, you have to answer in a human way (dont mention the above format, its just for you) where and how much content is taught where (in which video and at what timestamp) and guide the user to go to that particular video. If user asked unrelated question, tell hum that you can only ask the question related to he course
'''

# for index, item in new_df.iterrows():
#     print(index, item['title'], item['number'], item['text'], item['start'], item['end'])


with open("prompt.txt", "w") as f:
    f.write(prompt)
response = inference(prompt)["response"]
print(response)

with open("response.txt","w", encoding="utf-8") as f:
    f.write(response)