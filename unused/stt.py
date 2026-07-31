# speech to text conversion in this file
# this is  only for the 1 sample.mp3 only rest all mp3 to text are in yh vreate chunke.py

import whisper
import json

model = whisper.load_model("large-v2")

result = model.transcribe(
    audio="audios/sample.mp3",
    language="hi",
    task="translate",
    word_timestamps=False
)

# print(result["segments"])
chunks = []
for segment in result["segments"]:
    chunks.append({"start": segment["start"], "end": segment["end"], "text":segment["text"]})

print(chunks)

with open("output.json", "w") as f:
    json.dump(chunks,f)


# print(result["text"])
# with open("output.json","w") as f:
#     json.dump(f,result)


# print(result["text"])

# import os

# for file in os.listdir("audios"):
#     print(repr(file))