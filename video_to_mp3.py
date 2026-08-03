# using the ffpeg to convert the videos to the mp3
# we will give the whisper only the mp3 filenot the mp4 because it will take more load
# covert videos to mp3
import os
import subprocess
files = os.listdir("videos")
print(files)
for file in files:
    # print(files)
    tutorial_number = file.split(" [")[0].split(" #")[1]
    # print(tutorial_number)
    file_name = file.split(" ｜")[0]
    # print(file_name)
    print(tutorial_number, file_name)
    subprocess.run(["ffmpeg", "-i", f"videos/{file}", f"audios/{tutorial_number}_{file_name}.mp3"])

