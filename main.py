# import os
# import cv2

# def FrameCapture(video_path, output_path):
#     vidObj = cv2.VideoCapture(video_path)
#     count = 0
#     success = 1
#     while success:
#         success, image = vidObj.read()
#         if success:
#             cv2.imwrite(output_path + "/%d.jpg" % count, image)
#             count += 1

# path = upload_path
# output_path = os.path.join(basepath, "frames")

# videos = [video for video in os.listdir(path) if video.endswith(".avi") or video.endswith(".mp4")]
# videos.sort()
# print(videos)

# print("extracting frames from input video")
# for idx, video in enumerate(videos):
#     frame_path = os.path.join(output_path, video.split(".")[0])
#     os.makedirs(frame_path)
#     FrameCapture(os.path.join(path, video), frame_path)
from utils.util import (folder2vid)

folder2vid(image_folder="./sample_videos/clips/v04", output_dir="./", filename="result.avi")
