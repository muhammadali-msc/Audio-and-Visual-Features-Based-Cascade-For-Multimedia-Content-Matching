import os
import time
import tkinter as tk
from tkinter import filedialog
from multiprocessing import Manager, Process
import moviepy.editor as mp
from video.video_processing import multimediaVisualContentMatching
from matching.multimedia_matching import casecadeMultimediaContentMatching
from audio.audio_processing import multimediaAudioContentMatching
from display.display_process_video import displayProcessVideo


def select_video(title="Select a video file"):
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title=title, filetypes=[("Video files", "*.mp4;*.avi;*.mov;*.mkv")])
    return file_path

def init_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

# --- Main Execution ---
if __name__ == "__main__":
    start_time = time.process_time()

    print("Select Reference Video")
    ref_videoPath = select_video("Select the Reference video")
    print("Select Query Video")
    query_videoPath = select_video("Select the Query video")

    ref_video_file = mp.VideoFileClip(ref_videoPath)
    query_video_file = mp.VideoFileClip(query_videoPath)

    clip_frame_width, clip_frame_height = ref_video_file.size
    video_ssim_threshold = 0.6
    threshold = 0.65
    tolerance = 0.8

    channel_media_source = "PTV SPORTS"
    reference_video = os.path.basename(ref_videoPath)
    query_video = os.path.basename(query_videoPath)

    clip_path = init_dir(f"./data/ClipFrameData/{channel_media_source}/{reference_video}/Detection/{query_video}")
    video_path = init_dir(f"./data/VideoFrameData/{channel_media_source}/{reference_video}/Detection/{reference_video}_{query_video}")

    print("Select Detection Model:")
    print("1. Visual Detection Model")
    print("2. Audio Detection Model")
    print("3. Cascade Multimedia Detection Model")
    
    model_choice = input("Enter 1, 2, or 3: ")

    with Manager() as manager:
        detect_frame_list_pos1 = manager.list()

        if model_choice == '1':
            video_ssim_threshold = 0.8
            videoDetectCount, detect_frame_list_pos, detect_timeline, video_partial_detection_timeline, clip_frame_list = \
                multimediaVisualContentMatching(ref_videoPath, query_videoPath, clip_path, video_path,
                                                video_ssim_threshold, ref_video_file.fps, 0,
                                                ref_video_file.duration, 1, -1, queryVideoHashCode=[],
                                                cascade_detection=False)
            detect_frame_list_pos1.extend(detect_frame_list_pos)
            title = "Video Processed Result Using Visual Detection"

        elif model_choice == '2':
            total_detection_count, detect_timeline, clip_frame_list = \
                multimediaAudioContentMatching(ref_videoPath, query_videoPath, clip_path,
                                                             detect_frame_list_pos1, 128, 64, threshold, tolerance)

            title = "Video Processed Result Using Audio Detection"

        elif model_choice == '3':
            total_detection_count, detect_timeline, clip_frame_list = \
                casecadeMultimediaContentMatching(ref_videoPath, query_videoPath, clip_path, video_path,
                                                  video_ssim_threshold, detect_frame_list_pos1, 128, 64,
                                                  threshold, tolerance, cascade_detection=True)

            title = "Video Processed Result Using Cascade Detection"

        else:
            print("Invalid choice. Exiting.")
            exit()

        execuation_time = (time.process_time() - start_time)

        p = Process(target=displayProcessVideo,
                    args=[(ref_videoPath, clip_frame_width, clip_frame_height, title,
                           detect_frame_list_pos1, -1, -1, "Video Clip detected")])
        p.start()
        p.join()
