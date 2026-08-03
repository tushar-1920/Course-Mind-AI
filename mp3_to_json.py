import whisper
import os
import json

model = whisper.load_model("large-v2")

audios = os.listdir("audios")

audio_files = [
    "2.mp4_Your First HTML Website.mp3",
    "3.mp4_Basic Structure of an HTML Website.mp3"
    
]

for audio in audio_files:
    # print(audio)
    if("_" in audio):
        number = audio.split("_")[0][:-3]
        title = audio.split("_")[1][:-4]
        print(number,title)
        # result = model.transcribe(audio = f"audios/{audio}",
        # result = model.transcribe(audio = f"audios/sample.mp3",
        # audio=f"audios/{audio}",
        result = model.transcribe(
            audio = f"audios/{audio}",

        
            language = "hi",
            task = "translate",
            word_timestamps=False)
        chunks = []
        for segment in result["segments"]:
            chunks.append({"number": number,"title": title,"start": segment["start"], "end": segment["end"], "text":segment["text"]})

        chunks_with_metadata = {"chunks": chunks,"text" : result["text"]}

        with open(f"jsons/{audio}.json", "w") as f:
            json.dump(chunks_with_metadata,f)