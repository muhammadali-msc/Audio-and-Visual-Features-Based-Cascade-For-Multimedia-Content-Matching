import datetime
import moviepy.editor as mp
import librosa
from image_similarity_measures.quality_metrics import ssim, fsim, issm, psnr, rmse, sam, uiq
from scipy.spatial.distance import hamming
from  video.video_processing import init_video, getVideo_detail, read_video

### Audio Processing 
def extract_audioToSave(video_path, save_audio_path):
    
    video_file = mp.VideoFileClip(video_path)
    video_file.audio.write_audiofile(save_audio_path, fps = 44100)
    
    return video_file

def getAudio_Details(video_file_mp):
    
    audio_duration = video_file_mp.audio.duration
    audio_fps = video_file_mp.audio.fps
    
    return audio_duration, audio_fps

def load_audio_file(audio_path , audio_offset = 0.0, audio_duration = None, audio_sr = None):
    
    audio_y, audio_sr = librosa.load(audio_path,  offset= audio_offset,  duration= audio_duration, sr= audio_sr)
    
    return audio_y, audio_sr

def extract_audio_features(audio_y, n_ffts, hop_len, window_len, windows, audio_sr):
    
    spectral_roundoff_features = librosa.feature.spectral_rolloff(y= audio_y, n_fft= n_ffts, hop_length= hop_len, win_length= window_len, window= windows, sr = audio_sr)
    
    return spectral_roundoff_features[0]

def getfeatures_timeline(audio_features, n_ffts, hop_len, audio_sr):
    
    timestamps = librosa.frames_to_time(frames=range(len(audio_features)), sr=audio_sr, hop_length=hop_len, n_fft= n_ffts)
    
    return timestamps

def cascade_audio_matching(ref_videoPath, query_videoPath, detect_frame_list_pos1, query_audio_features, ref_audio_features, audio_sr, hop_len, n_ffts, index_from, ref_video_fps, threshold, tolerance, clip_frame_list, queryVideoHashCode):

    partial_detected_timeline = []
    partial_detected_features = []
    index = 0
    video_margin_frame = 40
    
    while index <= (len(ref_audio_features)-len(query_audio_features)):
        
        current_feature_list = ref_audio_features[index : len(query_audio_features)+index]
        matched_result = hamming(query_audio_features, current_feature_list)
        
        if matched_result <= tolerance:
            
            start_at = datetime.timedelta(seconds=librosa.frames_to_time(frames=index_from+index, sr=audio_sr, hop_length=hop_len, n_fft= n_ffts))
            end_at = datetime.timedelta(seconds=librosa.frames_to_time(frames= (len(query_audio_features)+index_from+index), sr=audio_sr,  hop_length=hop_len, n_fft= n_ffts))
            
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
            index += len(query_audio_features)
    
        else:
            index += 1

    return partial_detected_timeline, partial_detected_features

def multimediaAudioContentMatching(ref_videoPath, query_videoPath, clip_path, detect_frame_list_pos1, window_size, hop_length, thresholds, tolerances):
    
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
        
        partial_detected_timeline, partial_detected_features = cascade_audio_matching(ref_videoPath, query_videoPath, detect_frame_list_pos1, queryvideo_audio_features[0], refvideo_audio_features[0], audio_sr, hop_len= hop_length, n_ffts= n_fft, index_from = total_features, ref_video_fps = ref_video_file.fps, threshold= thresholds, tolerance= tolerances, clip_frame_list = clip_frame_list, queryVideoHashCode = queryVideoHashCode)
        
        
        
        for detect_item in range(0, len(partial_detected_timeline)):
            total_detection_count += 1
            detect_timeline[total_detection_count] = partial_detected_timeline[detect_item]
      
        if t_ends <= ref_video_file.duration:
            if len(partial_detected_features) > 0 and  i + query_durations < partial_detected_features[-1][1]:
                i = t_ends
                total_features += len(refvideo_audio_features[0])
            else:
                i = i + query_durations
                total_features += len(refvideo_audio_features[0])/2

    return total_detection_count, detect_timeline, clip_frame_list
