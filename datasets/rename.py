import glob
import os
from multiprocessing import Pool

def rename_file(video):
    uid = os.path.basename(video).split('.')[0]
    new_path = f'C:\\Users\\jisoo.kim\\Desktop\\Meshtalk\\audio_crop\\{uid}.wav'
    os.rename(video, new_path)
    return new_path

if __name__ == "__main__":
    videos = glob.glob('C:\\Users\\jisoo.kim\\Desktop\\Meshtalk\\audio_crop\\*.wav')

    # Use multiprocessing.Pool for parallel processing
    with Pool() as pool:
        new_paths = pool.map(rename_file, videos)

    print("Files renamed successfully.")
    print("New file paths:", new_paths)
