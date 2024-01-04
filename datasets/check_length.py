import subprocess
import os
import glob
import json
from multiprocessing import Pool

def get_media_duration(file_path):
    # Run ffprobe to get the duration of the media file
    result = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode == 0:
        return float(result.stdout.strip())
    else:
        return None


# Function to process each video
def process_video(video, val):
    uid = os.path.basename(video).split('.')[0]
    # val = data_dict.get(f'{uid}.mp4')  # Use get to handle missing keys gracefully
    if val is None:
        print(f'Data not found for')
        print(uid)
        return

    start = val['duration']['start_sec']
    end = val['duration']['end_sec']
    raw_duration = end - start

    try:
        video_duration = get_media_duration(video)
    except:
        print(f'Something went wrong with {uid}')
        print(uid)
        return

    if video_duration is not None:
        if abs(video_duration - raw_duration) <= 0.03:
            pass
        else:
            print(f'Audio and video durations are different for {uid}: {video_duration - raw_duration}')
            print(uid)
    else:
        print(f'Failed to get duration information for')
        print(uid)

if __name__ == "__main__":
    with open('C:\\Users\\jisoo.kim\\Meshtalk\\EMOTE\\datasets\\celebvtext_info.json') as f:
        data_dict = json.load(f)

    # Replace these with the paths to your audio and video files
    videos = glob.glob('C:\\Users\\jisoo.kim\\Desktop\\Meshtalk\\audio_crop\\*.wav')

    # Number of processes to run in parallel
    num_processes = 24  # Adjust as needed

    # Use a Pool to parallelize the processing of videos
    with Pool(num_processes) as pool:
        # pool.map(process_video, videos)
        pool.starmap(process_video, [(video, data_dict[f'{os.path.basename(video).split(".")[0]}.mp4']) for video in videos])
