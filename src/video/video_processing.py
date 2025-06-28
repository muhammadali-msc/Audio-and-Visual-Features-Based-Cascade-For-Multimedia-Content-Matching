import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import datetime
from image_similarity_measures.quality_metrics import ssim, fsim, issm, psnr, rmse, sam, uiq
from scipy.spatial.distance import hamming
import plotly.express as px

#### Capturing the Video using the opencv based on input parametter of video_uri
def init_video(video_uri): 
    
    cap_video = cv2.VideoCapture(video_uri)

    if not cap_video.isOpened():
        print("video unable to be read")
    return cap_video

#### Getting the capture video details : fps, frame_count, duration 
def getVideo_detail(cap_video):
    '''getVideo_detail provide the video frame rate, frame count, duration details
    Remember: if frame count is 60 then actual readed number of frame will be between 0 to 59'''
    # Get the frames per second
    fps = cap_video.get(cv2.CAP_PROP_FPS)
    # Get the width and height of the video
    width = int(cap_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Get the total numer of frames in the video.
    frame_count = int(cap_video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate the duration of the video in seconds
    duration = frame_count / fps
    
    return fps, frame_count, duration, width, height

#### read_video function allows to read the capture video object and return the frame list as output.
def read_video(cap_video, frame_width, frame_height, display_title, frame_path, start_frame = -1, end_frame = -1, display_image = True, frame_hashcode = False, clip_frame_width = -1, clip_frame_height = -1, last_frame = list(), hist = 0.15):
    '''read_video funcation take 6 argument as following:
    cap_video = to get the open cv2 captureVideo object
    display_title =  to show the title on window screen while displaying images
    frame_path =  folder path to save the frames
    start_frame = starting frame position form where you want to start reading the frame. 
                  Befault value is -1 that means you want to read at the beginning frame 0.
    end_frame =  last frame index you want to read. 
                 By default it is -1 that means you want to read whole video until end of frame (= frame_count)
    display_image = True or False, to display the image. By default it is ture.
    return = it will reture the readed frame list.
    Other:
    if start_frame = -1, end_frame = -1 then it will read the whole video.
    if start_frame = end_frame then it will read a single frame.
    '''

    #print("Hist = ", hist)
    frame_count = cap_video.get(cv2.CAP_PROP_FRAME_COUNT)
    #print("////////////////////////////////////", len(last_frame))
    frame_list = list(last_frame)
    #print("////////////////////////////////////", len(frame_list))
    hash_code = 0
    # if start isn't specified lets assume 0
    if start_frame < 0:  
        start_frame = 0
    # if end isn't specified assume the end of the video
    if end_frame < 0:
        end_frame = frame_count
    
    cap_video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    
    while True:
       
        success,frame = cap_video.read()
        
        if success:
            frame_resize = rescale_frame(frame, frame_width, frame_height, 60)
            if display_image:
                #cv2.namedWindow("output", cv2.WINDOW_NORMAL) 
              
                #frame_resize = rescale_frame(frame, 60)
                
                cv2.imshow(display_title,frame_resize)
                # Resize the Window
                #cv2.resizeWindow("ouput", 1280, 720)
                
            noise_remove = cv2.GaussianBlur(frame, (37, 37), 0)
            #noise_remove = cv2.medianBlur(noise_remove, 15)
            
            #cv2.imwrite("./"+frame_path+"/frame" + str(start_frame)+ '.jpg',frame)
            ### frame are in BRG mode it should be convert to RGB
            #img = cv2.cvtColor(noise_remove, cv2.COLOR_BGR2RGB)
            
            readed_frame = noise_remove.astype(int)
    
            if len(frame_list) > 0:
                #print(frame_list[-1])
                #print("-------------------------------------------")
                prev_gray = cv2.cvtColor(frame_list[-1].astype(np.uint8), cv2.COLOR_BGR2GRAY)
                curr_gray = cv2.cvtColor(readed_frame.astype(np.uint8), cv2.COLOR_BGR2GRAY)
                # Calculate histograms for the frames
                prev_hist = cv2.calcHist([prev_gray], [0], None, [256], [0, 256])
                curr_hist = cv2.calcHist([curr_gray], [0], None, [256], [0, 256])
                # Calculate histogram difference using Bhattacharyya coefficient
                hist_diff = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_BHATTACHARYYA)
                #print("Hist Diff", hist_diff >= hist , hist_diff, hist)
                # If the histogram difference exceeds a certain threshold, consider it a unique keyframe
                if hist_diff >= hist:
                    #print("Added FRAME")
                    cv2.imwrite("./"+frame_path+"/frame" + str(start_frame)+ '.jpg',frame)
                    if frame_hashcode:
                        hash_code = getHashCode(readed_frame, hash_code)
                    #print("Index Start Frame", start_frame)
                    frame_list.append(readed_frame)
                #else:
                    #print("NOT ADDED")
            
            #print(len(frame_list))
            if len(frame_list) == 0:
                #print("//////////////////////////////////////// ADDED")
                cv2.imwrite("./"+frame_path+"/frame" + str(start_frame)+ '.jpg',frame)
                hash_code = getHashCode(readed_frame, hash_code)
                #print("Index Start Frame", start_frame)
                #print("LAST FRAME frame_list", len(last_frame))
                frame_list.append(readed_frame)
                #print("LAST FRAME frame_list", len(last_frame))
            start_frame += 1
            if cv2.waitKey(25) & 0xFF == ord('q') or start_frame >= end_frame:
                #print("video is closed!")
                cv2.destroyAllWindows()
                break
        else:
            #print("end of video")
            cv2.destroyAllWindows()
            # exit()
    #print("Total Frames ", len(frame_list))
    #print("LAST FRAME", len(last_frame))
    return frame_list, hash_code

def rescale_frame(frame, frame_width, frame_height, percent=75):
    
    width = int(frame_width * percent/ 100)
    height = int(frame_height * percent/ 100)
    
    return cv2.resize(frame, (width, height), interpolation =cv2.INTER_AREA)

def extractClip(cap_video,frame_width, frame_height, display_title,frame_path, start_frame_index, end_frame_index):
    
    video_clip_frame_list = list()
    if cap_video.isOpened():
        video_clip_frame_list, hash_code = read_video(cap_video, frame_width, frame_height,display_title, frame_path,start_frame_index,end_frame_index,True, True)
    
    return video_clip_frame_list

def addNextFrame(cap_video,frame_path, current_frame_list, prev_frame_end_index):
    
    readed_frame = read_video(cap_video, frame_path, prev_frame_end_index,prev_frame_end_index)
    current_frame_list.pop(0)
    current_frame_list.extend(readed_frame)
    
    return current_frame_list

#### Getting the hashcode by adding the two frames in given frame list and return the mod result list based on prime number.
def getHashCode(next_frame, prev_frames_hashcode):
    
    hash_code = (prev_frames_hashcode + next_frame) %251
    
    return hash_code

def getUpdatedHashCode(init_frame, next_frame, hash_code):
    
    hash_code = ((hash_code + 251 - init_frame )+ next_frame) % 251
    
    return hash_code

#### displayImage function will display an image on window using cv2
def displayImage(frame):
    '''This funcation will take a frame list and use open cv to display it as image'''
    cv2.imshow("image", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#### displayImageWithText function will display an image along with text on window using cv2
def displayImageWithText(loadImage, text):
    
    #image = cv2.imread(loadImage)
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 50)
    fontScale = 1
    color = (0, 0, 255)
    thickness = 2
    # Using cv2.putText() method
    image = cv2.putText(loadImage, text, org, font, 
               fontScale, color, thickness, cv2.LINE_AA)
    # Displaying the image
    cv2.imshow("image", image)
    #cv2.waitKey(0)
    
    cv2.destroyAllWindows()

#### Getting the duration of video by providing the frames_count and fps as argument while calling it.
def getVideoDuration(frames_count, fps):
    
    seconds = round(frames_count / fps)
    video_duration = datetime.timedelta(seconds=seconds)
    
    return video_duration

#### Getting the video percentage
def getVideoPercentage(current_frame_count, fps):
    
    percentage = round((current_frame_count/fps)*100, 2)
    
    return percentage

#### Matching the hashcode result of both video and clip.
def matchHashCode(videoHashCode, clipHashCode):
    
    return np.array_equal(videoHashCode, clipHashCode)

def matchHashCode_hamming(videoHashCode, clipHashCode):
    
    video_hashcode = np.reshape(videoHashCode, (np.product(videoHashCode.shape),))
    clip_hashcode = np.reshape(clipHashCode, (np.product(clipHashCode.shape),))
    
    hamming_dist = hamming(clip_hashcode, video_hashcode)
    
    # Calculate the Euclidean distance
    distance = np.linalg.norm(clipHashCode - videoHashCode)

    # Calculate the maximum possible distance
    max_distance = np.sqrt(np.sum(np.square(255))) * np.sqrt(videoHashCode.size)

    # Normalize the distance between 0 and 1
    euclidean_dist = distance / max_distance

    # Flatten the binary images into 1D arrays
    flatten_image1 = clipHashCode.flatten()
    flatten_image2 = videoHashCode.flatten()

    # Calculate the Hamming distance
    hamming_distance = hamming(flatten_image1, flatten_image2)

    ssim_dist = ssim(clipHashCode, videoHashCode)
    
    #print("Eculidean Distance {}, hamming dist {}, SSIM dist {}".format(euclidean_dist, hamming_distance, ssim_dist))
    return euclidean_dist, hamming_dist, ssim_dist

def extractVideo(cap_video,frame_path, clip_frame_count):
    
    video_framelist = read_video(cap_video,frame_path, 0, end_frame=clip_frame_count)
    video_hashCode = getHashCode(video_framelist)
    
    return video_hashCode

#### Displaying the Experiment Result
def displayGraph(videos_df, x_col, y_col, hover_name_col):
    
    fig = px.line(videos_df, x=x_col, y=y_col,
                     hover_name=hover_name_col, hover_data=videos_df.columns)
    fig.update_traces(mode="markers+lines")
    fig.update_layout(
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        )
    )
    fig.show()

#### Creating CSV file to save the matched video result
def createMultiMediaContentMatchingCSV(file_name):
    
    df = pd.DataFrame(columns=["video_index","media","reference_video_name", "query_video_name","reference_video_duration", "query_video_duration", "ref_video_fps", "query_video_fps", "total_reference_frame", "total_query_frame", "total_query_keyframe_hashcode","Threshold", "Tolerance", "Video SSIM Threshold", "DetectionModel", "execuation_time_sec", "video_detection_count", "detected_times"])
    df.to_csv(file_name, index=False)

#### Reading the saved CSV file
def readCSVfile(cascade_detection_model_csv):
    
    # reading the csv file
    df = pd.read_csv(cascade_detection_model_csv)
    
    return df

#### Extracting the reference video frame and using hamming distance to matched the query video hash code with reference video hash code
def extractVideoFrame(cap_video, reference_video, query_video, video_path, video_ssim_threshold, frame_width, frame_height, video_frame_count, clip_frame_list_count, clip_frame_count, clipHashCode, query_duration, video_fps, frame_start_at=0, cascade_detection = False):
    
    
    videoDetectCount = 0
    video_frame_list = list()
    detect_frame_list_pos = list()
    detect_timeline = []
    videoHashCode = 0
    video_partial_detection_timeline = []
    i = 0
    frame_index = frame_start_at
    previous_ssim = 0
    skip_frame_index = 0
    select_frame_index = []
    precentage_index = 0
    last_frames = list()
    query_frame_index = 0
    #print("STARTING", frame_index, video_frame_count, query_durations )
    while frame_index <  video_frame_count:
        i += 1
        query_frame_index += 1
        #print("Check Frame Index", i, clip_frame_count)
        video_precentage = getVideoPercentage(frame_index,video_frame_count-1)
        if video_precentage >= precentage_index:
            print("{} % of video is processed! Current frame index {}".format(video_precentage, frame_index))
            precentage_index += 1
        #print("Last frame", len(last_frames))
        #if len(video_frame_list) == 0:
            #print("Last Frame")
            #last_frames = list()
        #else:
            #last_frames = [video_frame_list[-1]]
           

        readed_frame, hash_code = read_video(cap_video, frame_width, frame_height,"VIDEO PROCESS",video_path,start_frame= frame_index, end_frame=frame_index, display_image = False, frame_hashcode= False, last_frame = last_frames, hist = 0.15)
        #print("Read Frames ", len(readed_frame), len(readed_frame) > 1)
        #print("Last frame", len(last_frames))
        
     
        if len(readed_frame) > 1:
            video_frame_list.append(readed_frame[1])
            last_frames = [video_frame_list[-1]]
            select_frame_index.append(frame_index)
            #print("VIDEO FRAME ADDED", len(video_frame_list))
            if (len(video_frame_list) <=  clip_frame_list_count):
                #print("ADD GET", len(readed_frame), readed_frame[1].shape, videoHashCode.shape)
                videoHashCode = getHashCode(readed_frame[1], videoHashCode)
            else:
                #print("ADD Update")
                pervious_init_frame = video_frame_list.pop(0)
                select_frame_index.pop(0)
                videoHashCode = getUpdatedHashCode(pervious_init_frame, readed_frame[1], videoHashCode)
                #print(videoHashCode.shape)
            #print("Check /////////////////////////////////", len(video_frame_list) ,  clip_frame_list_count, i ,  clip_frame_count, i >=  clip_frame_count)
            
        
        
        #print(len(video_frame_list))
        #print(cascade_detection , frame_index ==  video_frame_count)
        if (cascade_detection and frame_index ==  video_frame_count and videoDetectCount == 0) or (query_frame_index >=  (clip_frame_count-5) and len(video_frame_list) >= clip_frame_list_count):
                #print("matchHashCode_hamming_dist", len(video_frame_list), clip_frame_list_count)
                eculidean_dist, matchHashCode_hamming_dist, ssim_dist = matchHashCode_hamming(videoHashCode, clipHashCode)
                #print("VIUSAL Hashcode Matching", ssim_dist, eculidean_dist)
                
                if ssim_dist >= video_ssim_threshold:
                    #print("VISUAL HASHCODE INSIDE")
                    #Image.fromarray((video_frame_list[0]* 1).astype(np.uint8)).convert('RGB').show()
                    #Image.fromarray((video_frame_list[-1]* 1).astype(np.uint8)).convert('RGB').show()
                    if select_frame_index[0] < (frame_index - (clip_frame_count-1)):
                        start_frames_count = frame_index - (clip_frame_count-1)
                    else:
                        start_frames_count = select_frame_index[0]
                    
                    ending_frame_count = select_frame_index[-1]
                    #print(int(query_duration*video_fps), clip_frame_list_count)
                    #print("Duration ", getVideoDuration((frame_index - (clip_frame_list_count-1)),video_fps))
                    #print("Duration ", getVideoDuration(int(frame_index - (int(query_duration*video_fps) - clip_frame_list_count)),video_fps))
                    
                    time_start_at = getVideoDuration(start_frames_count,video_fps) 
                    time_end_at = getVideoDuration(ending_frame_count,video_fps)
                    
                    #print("SSIM", ssim_dist, previous_ssim, ssim_dist > previous_ssim)
                    
                    
                    detect_timeline.append(str(time_start_at) + " to " + str(time_end_at))

                    detect_frame_list_pos.append([start_frames_count,ending_frame_count])
                    
                    video_partial_detection_timeline.append({"Detection Model" : "Video", 
                                                          "Timeline" : str(time_start_at) + " to " + str(time_end_at),
                                                         "Start Frame Position" : start_frames_count,
                                                         "Ending Frame Position": ending_frame_count,
                                                         "Matched Result" : ssim_dist,
                                                         "Video SSIM Threshold": video_ssim_threshold})
                    #print("{} % of video is processed! Current frame index {}".format(getVideoPercentage(frame_index,video_frame_count-1), frame_index))
                    print("query video \"{}\" visual hashcode is matched with \"{}\" by matched distance \"{}\" at timestamp \"{}\" to \"{}\"".format(query_video, reference_video, ssim_dist, time_start_at, time_end_at))
                
                    videoDetectCount = len(video_partial_detection_timeline)
                    
                    
                    videoHashCode = 0
                    readed_frame = None
                    video_frame_list.clear()
                    #print("video_frame_list clear", len(video_frame_list))
                    select_frame_index.clear()
                    query_frame_index = 0
                    #print("Current Frame Index SSIM Inside", frame_index)
                    
                    #print("After Frame Index", frame_index, (query_duration*video_fps), clip_frame_list_count )
        #print("Last frame", len(last_frames))
        if len(last_frames) == 0 and readed_frame is not None:
            #print("ENTRING FIRST", len(readed_frame))
            videoHashCode = hash_code
            video_frame_list.append(readed_frame[0])
            last_frames = [video_frame_list[-1]]
            select_frame_index.append(frame_index)
            #frame_index += 1
        frame_index += 1
    return videoDetectCount, detect_frame_list_pos, detect_timeline, video_partial_detection_timeline, videoHashCode

def multimediaVisualContentMatching(ref_videoPath, query_videoPath, clip_path, video_path, video_ssim_threshold, video_fps, video_start_at, video_end_at, margin, clip_frame_list, queryVideoHashCode, cascade_detection):
    
    ref_video_cap_cv = init_video(ref_videoPath)
    query_video_cap_cv = init_video(query_videoPath)
    
    reference_video = ref_videoPath.split("/")[-1]
    query_video = query_videoPath.split("/")[-1] 
    
    video_fps, video_frame_count, ref_duration, video_frame_width, video_frame_height = getVideo_detail(ref_video_cap_cv)

    #print(ref_duration)
    # Getting the frame info
    clip_fpss, clip_frame_counts, query_duration, clip_frame_width, clip_frame_height   = getVideo_detail(query_video_cap_cv)
    #print(queryVideoHashCode)
    if len(queryVideoHashCode) == 0:
        # Getting the clip frame list
        #print("QUERY VIDEO")
        clip_frame_list, queryVideoHashCode = read_video(query_video_cap_cv, clip_frame_width, clip_frame_height, "CLIP PROCESS", clip_path,start_frame= -1, end_frame=-1, display_image = True, frame_hashcode= True, last_frame = list(), hist= 0.15)
        #print("Total Query Video Frame", clip_frame_list)
    
    if video_start_at is None or int(video_start_at*video_fps) - margin < 0:
        start_at = 0
    else:
        start_at =  int(video_start_at*video_fps) - margin
    
   
    if video_end_at is None or int(video_end_at*video_fps) + margin > ref_duration*video_fps:
       
        end_at = int(ref_duration*video_fps)
    else:
        end_at = int(video_end_at*video_fps) + margin
    
    frame_count = end_at - start_at
    print("Total query frames {} \nStarting frame in reference video {} \nEnding frame in reference video {} \nProcessing total frames {} of video duration {} to {}".format(len(clip_frame_list), start_at,end_at,frame_count, video_start_at, video_end_at))
    print("//////////////////////////////////////////////////////")
    
    #print(queryVideoHashCode)
    videoDetectCount , detect_frame_list_pos, detect_timeline, video_partial_detection_timeline, videoHashCode = extractVideoFrame(ref_video_cap_cv, reference_video, query_video, video_path, video_ssim_threshold, clip_frame_width, clip_frame_height, end_at, len(clip_frame_list), clip_frame_counts, queryVideoHashCode, query_duration, video_fps, start_at, cascade_detection)

    return videoDetectCount, detect_frame_list_pos, detect_timeline, video_partial_detection_timeline , clip_frame_list
