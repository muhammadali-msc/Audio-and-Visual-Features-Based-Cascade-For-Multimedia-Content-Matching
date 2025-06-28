import datetime
import moviepy.editor as mp
import librosa
from image_similarity_measures.quality_metrics import ssim, fsim, issm, psnr, rmse, sam, uiq
from scipy.spatial.distance import hamming
from video.video_processing import multimediaVisualContentMatching, init_video, getVideo_detail, read_video
from audio.audio_processing import extract_audio_features

def cascade_audio_visual_matching(ref_videoPath, query_videoPath, clip_path, video_path, video_ssim_threshold, detect_frame_list_pos1, query_audio_features, ref_audio_features, audio_sr, hop_len, n_ffts, index_from, ref_video_fps, threshold, tolerance, clip_frame_list, queryVideoHashCode, cascade_detection):
    #print("cascade_audio_visual_matching")
    partial_detected_timeline = []
    partial_detected_features = []
    index = 0
    video_margin_frame = 40
    #print(len(ref_audio_features), len(query_audio_features), hop_len, n_ffts, index_from)
    while index <= (len(ref_audio_features)-len(query_audio_features)):
        #print("Current index ", index)
        current_feature_list = ref_audio_features[index : len(query_audio_features)+index]
        matched_result = hamming(query_audio_features, current_feature_list)
        #print(matched_result)
          
        #if matched_result > 0.70 and matched_result < 0.85:
            #start_at = datetime.timedelta(seconds=librosa.frames_to_time(frames=index_from*919, sr=query_sr, hop_length=hop_len, n_fft= n_ffts))
            #partial_matched.append([matched_result, index])
        if matched_result <= tolerance:
            
            start_at = datetime.timedelta(seconds=librosa.frames_to_time(frames=index_from+index, sr=audio_sr, hop_length=hop_len, n_fft= n_ffts))
            end_at = datetime.timedelta(seconds=librosa.frames_to_time(frames= (len(query_audio_features)+index_from+index), sr=audio_sr,  hop_length=hop_len, n_fft= n_ffts))
            #print("Start At",index_from+index,"End At", (len(query_audio_features)+index_from+index) )
            if matched_result <= threshold:
    
                detect_frame_list_pos1.append([start_at.total_seconds()*ref_video_fps, end_at.total_seconds()*ref_video_fps])
                partial_detected_timeline.append({"Detection Model" : "Audio", 
                                                  "Timeline" : str(start_at) + " to " + str(end_at),
                                                 "Start Frame Position" : start_at.total_seconds()*ref_video_fps,
                                                 "Ending Frame Position": end_at.total_seconds()*ref_video_fps,
                                                 "Matched Result" : matched_result,
                                                 "Threshold" : threshold })
                partial_detected_features.append([int(start_at.total_seconds()), end_at.total_seconds()])
                print("query video \"{}\" audio features are matched with \"{}\" by matched distance \"{}\" at timestamp \"{}\" to \"{}\"".format(query_videoPath.split("/")[-1], ref_videoPath.split("/")[-1], matched_result, start_at, end_at))
            elif cascade_detection and matched_result > threshold and matched_result <= tolerance:
                print("//////////////////////////////////////////////////////")
                print("Audio features hamming distance is {} which is greater then threshold {} but less than tolerance {} therefore matching the visual feature...{} to {}".format(matched_result, threshold, tolerance, start_at, end_at))
            
                videoDetectCount, detect_frame_list_pos, video_detect_timeline, video_partial_detection_timeline, clip_frame_list = multimediaVisualContentMatching(ref_videoPath, query_videoPath, clip_path, video_path, video_ssim_threshold, ref_video_fps, int(start_at.total_seconds()), end_at.total_seconds(), video_margin_frame, clip_frame_list, queryVideoHashCode, cascade_detection)
                partial_detected_timeline.extend(video_partial_detection_timeline)
                detect_frame_list_pos1.append([start_at.total_seconds()*ref_video_fps, end_at.total_seconds()*ref_video_fps])
                partial_detected_features.append([int(start_at.total_seconds()), end_at.total_seconds()])
                #print("outside video match")
            #print("Update index ", index, len(query_audio_features)+index)
            index += len(query_audio_features)
    
        else:
            index += 1
    #print("cascade_audio_visual_matching")
    return partial_detected_timeline, partial_detected_features

def casecadeMultimediaContentMatching(ref_videoPath, query_videoPath, clip_path, video_path, video_ssim_threshold, detect_frame_list_pos1, window_size, hop_length, thresholds, tolerances, cascade_detection):
    
    i = 0
    total_features = 0
    start_display = True
    n_fft = window_size
    total_detection_count = 0
    detect_timeline = {}
    
    ref_video_file = mp.VideoFileClip(ref_videoPath)
    query_video_file = mp.VideoFileClip(query_videoPath)
    
    print("{} duration is {} sec".format(query_videoPath.split('\\')[-1], query_video_file.duration))
    print("{} duration is {} sec".format(ref_videoPath.split('\\')[-1], ref_video_file.duration))
   
    query_durations = query_video_file.duration

    query_sr = query_video_file.audio.fps
    audio_sr = 44100
    extract_ref_duration = query_durations*2
    
    print("Extracting query video {} viusal hashcode... ".format(query_videoPath.split('\\')[-1]))
    
    query_video_cap_cv = init_video(query_videoPath)
    clip_fps, clip_frame_count, duration, clip_frame_width, clip_frame_height   = getVideo_detail(query_video_cap_cv)
    clip_frame_list, queryVideoHashCode = read_video(query_video_cap_cv, clip_frame_width, clip_frame_height, "CLIP PROCESS", clip_path, start_frame= -1, end_frame=-1, display_image = True, frame_hashcode= True, last_frame = list(), hist= 0.15)
    #print("CHECK QUERY FRAME COUNT" , len(clip_frame_list))
    print("Extracting query video {} audio features...".format(query_videoPath.split('\\')[-1]))
    queryvideo_audio_features = extract_audio_features(query_video_file.audio.to_soundarray(fps = query_sr).T, n_fft, hop_length, window_size, 'hann', query_sr)
    print("Processing the reference video {} using cascade multimedia content matching technique...".format(ref_videoPath.split('\\')[-1]))
    
    while i < ref_video_file.duration-query_video_file.duration:
       
        if i + extract_ref_duration < ref_video_file.duration:
            t_ends = i+ extract_ref_duration
        else:
            t_ends = ref_video_file.duration
        print("Extracting audio features from {} to {} sec...".format(i, t_ends))
        video_subclip = ref_video_file.subclip(t_start=i, t_end=t_ends)
        refvideo_audio_features = extract_audio_features(video_subclip.audio.to_soundarray(fps = query_sr).T, n_fft, hop_length, window_size, 'hann', query_sr)
        
        partial_detected_timeline, partial_detected_features = cascade_audio_visual_matching(ref_videoPath, query_videoPath, clip_path, video_path, video_ssim_threshold, detect_frame_list_pos1, queryvideo_audio_features[0], refvideo_audio_features[0], audio_sr, hop_len= hop_length, n_ffts= n_fft, index_from = total_features, ref_video_fps = ref_video_file.fps, threshold= thresholds, tolerance= tolerances, clip_frame_list = clip_frame_list, queryVideoHashCode = queryVideoHashCode, cascade_detection = cascade_detection)
        
        
        
        for detect_item in range(0, len(partial_detected_timeline)):
            #print("ENTER")
            total_detection_count += 1
            detect_timeline[total_detection_count] = partial_detected_timeline[detect_item]
      
        #if len(dectected_count) == 0 and len(partial_matched) > 0:
            #print("INSIDE..........")
            #partial_matched.sort(key= lambda x : x[0])
            #print(partial_matched)
            #i = i+datetime.timedelta(seconds=librosa.frames_to_time(frames=current_features+partial_matched[0][1], sr=query_sr, hop_length=hop_length, n_fft= n_fft)).total_seconds()
            #print(current_features,partial_matched[0][1])
        #else:
        #if dispaly_video_process.is_alive() == False:
            #dispaly_video_process.start()
        #if len(partial_detected_timeline) > 0:
            #i = i + extract_ref_duration
        if t_ends <= ref_video_file.duration:
            
            if len(partial_detected_features) > 0 and  i + query_durations < partial_detected_features[-1][1]:
                i = t_ends
                total_features += len(refvideo_audio_features[0])
            else:
                i = i + query_durations
                total_features += len(refvideo_audio_features[0])/2

    return total_detection_count, detect_timeline, clip_frame_list
