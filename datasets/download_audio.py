"""
Downloader
"""

import os
import json
import cv2
import time
import shutil
from multiprocessing import Pool, cpu_count
from functools import partial



def download(audio_path, ytb_id, file_name):
    """
    ytb_id: youtube_id
    save_folder: save video folder
    proxy: proxy url, defalut None
    """
    if os.path.exists(audio_path):
        print(audio_path)
        url = f'https://www.youtube.com/watch?v={ytb_id}'
        down_audio = f'yt-dlp -P {audio_path} -o "{file_name}.%(ext)s" -x --audio-format wav {url}'
        print(down_audio)
        status_audio = os.system(down_audio)
        if status_audio != 0:
            print(f"audio not found: {ytb_id}")


def process_ffmpeg(raw_aud_path, aud_save_folder, save_vid_name, time):
    """
    raw_vid_path:
    save_folder:
    save_vid_name:
    bbox: format: top, bottom, left, right. the values are normalized to 0~1
    time: begin_sec, end_sec
    """
    def secs_to_timestr(secs):
        hrs = secs // (60 * 60)
        min = (secs - hrs * 3600) // 60
        sec = secs % 60
        end = (secs - int(secs)) * 100
        return "{:02d}:{:02d}:{:02d}.{:02d}".format(int(hrs), int(min), int(sec), int(end))

    out_audio_path = os.path.join(f'{aud_save_folder}', f'{save_vid_name}.wav')
    start_sec, end_sec = time

    cmd_audio = f'ffmpeg -y -i {raw_aud_path} -ss {start_sec} -to {end_sec} -c copy {out_audio_path}'
    os.system(cmd_audio)

    # shutil.rmtree(f'{raw_aud_path}')
    return out_audio_path


def load_data(file_path):
    with open(file_path) as f:
        data_dict = json.load(f)
    
    for key in data_dict.keys() :

    # for key, val in data_dict.items():
        save_name = key
        val = data_dict[save_name]
        ytb_id = val['ytb_id']
        file_name = key.split('.')[0]
        time = val['duration']['start_sec'], val['duration']['end_sec']
        bbox = [val['bbox']['top'], val['bbox']['bottom'], val['bbox']['left'], val['bbox']['right']]
        yield ytb_id, save_name, time, bbox, file_name


def process_video(args):
    vid_id, save_vid_name, duration, raw_aud_root, processed_aud_root = args
    raw_aud_path = os.path.join(raw_aud_root, f'{save_vid_name}.wav')

    # Downloading is io bounded and processing is cpu bounded.
    # It is better to download all videos firstly and then process them via multiple cpu cores.
    try :
        download(raw_aud_root, vid_id, save_vid_name)
        process_ffmpeg(raw_aud_path, processed_aud_root, save_vid_name, duration)
        os.remove(raw_aud_path)
    except :
        print(f'{raw_aud_path} not processed')
        

if __name__ == '__main__':
    json_path = 'celebvtext_info.json'  # json file path
    raw_aud_root = 'C:\\Users\\jisoo.kim\\Desktop\\Meshtalk\\audio'
    processed_aud_root = 'C:\\Users\\jisoo.kim\\Desktop\\Meshtalk\\audio_crop'

    os.makedirs(raw_aud_root, exist_ok=True)
    os.makedirs(processed_aud_root, exist_ok=True)

    # Load data
    with open(json_path) as f:
        data_dict = json.load(f)

    # Create a list of arguments for each video
    video_args = []
    for key, val in data_dict.items():
        save_name = key
        ytb_id = val['ytb_id']
        file_name = key.split('.')[0]
        duration = val['duration']['start_sec'], val['duration']['end_sec']
        video_args.append((ytb_id, save_name, duration, raw_aud_root, processed_aud_root))

    # Determine the number of CPUs to use
    num_cpus = min(cpu_count(), len(video_args))
    
    # Use multiprocessing.Pool to parallelize the process
    with Pool(num_cpus) as pool:
        total_start = time.time()
        pool.map(process_video, video_args)
        total_end = time.time()

    print(f'Takes {total_end - total_start}')



# if __name__ == '__main__':
#     json_path = 'celebvtext_info.json'  # json file path
#     # raw_aud_root = os.path.join('downloaded_celebvtext', 'audio')
#     # processed_aud_root = os.path.join('downloaded_celebvtext', 'audio_crop')
#     raw_aud_root = 'C:\\Users\\jisoo.kim\\Desktop\\Meshtalk\\audio'
#     processed_aud_root = 'C:\\Users\\jisoo.kim\\Desktop\\Meshtalk\\audio_crop'
#     proxy = None  # proxy url example, set to None if not use

#     os.makedirs(raw_aud_root, exist_ok=True)
#     os.makedirs(processed_aud_root, exist_ok=True)
#     total_start = time.time()
#     ffmpeg_time = 0
#     for vid_id, save_vid_name, duration, bbox, file_name in load_data(json_path):
#         # raw_aud_path = os.path.join(raw_aud_root, file_name + ".wav")
#         raw_aud_path = f'{raw_aud_root}\\{file_name}.wav'
#         # Downloading is io bounded and processing is cpu bounded.
#         # It is better to download all videos firstly and then process them via multiple cpu cores.
#         start = time.time()
#         download(raw_aud_root, vid_id, file_name)
#         process_ffmpeg( raw_aud_path, processed_aud_root, file_name, duration)
#         end = time.time()
#         print(f'{file_name} takes {end-start}')
#     total_end = time.time()
#     print(f'Takes {total_end - total_start}')
