import cv2
   
def rescale_frame(frame, frame_width, frame_height, percent=75):
    width = int(frame_width * percent/ 100)
    height = int(frame_height * percent/ 100)
    return cv2.resize(frame, (width, height), interpolation =cv2.INTER_AREA)

def init_video(video_uri):
    cap_video = cv2.VideoCapture(video_uri)

    if not cap_video.isOpened():
        print("video unable to be read")
    return cap_video

def displayProcessVideo(args):
    query_videoPath, frame_width, frame_height, display_title, detect_frame_list_pos, start_frame, end_frame, display_text = args

    cap_video = init_video(query_videoPath)
    frame_count = int(cap_video.get(cv2.CAP_PROP_FRAME_COUNT))
    success = True

    if start_frame < 0:
        start_frame = 0
    if end_frame < 0:
        end_frame = frame_count

    cap_video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    detect_frame_index = 0

    while True:
        success, frame = cap_video.read()
        if not success:
            break

        current_frame_index = int(cap_video.get(cv2.CAP_PROP_POS_FRAMES)) - 1

        # Check for detection
        if detect_frame_index < len(detect_frame_list_pos):
            start_detecting, end_detecting = detect_frame_list_pos[detect_frame_index]

            if start_detecting <= current_frame_index <= end_detecting:
                font = cv2.FONT_HERSHEY_SIMPLEX
                org = (50, 50)
                fontScale = 1
                color = (0, 0, 255)
                thickness = 2
                text = f"{detect_frame_index + 1} {display_text}"
                frame = cv2.putText(frame, text, org, font, fontScale, color, thickness, cv2.LINE_AA)

            elif current_frame_index > end_detecting:
                detect_frame_index += 1  # Only move to the next detection AFTER current detection ends

        cv2.imshow(display_title, frame)
        start_frame += 1

        if cv2.waitKey(25) & 0xFF == ord('q') or start_frame >= end_frame:
            break

    cap_video.release()
    cv2.destroyAllWindows()
    return True
