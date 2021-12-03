import os 
import time 
import requests
import threading
import pandas as pd


def download_if_image(url, count):
    '''Downloads image if the url contains an image'''
    global OUT_PATH
    global N_DOWNLOADED

    with open(f"{OUT_PATH}/image_{count}.jpg", 'wb') as write_image_file:
        with requests.get(url) as req:
            for chunk in req.iter_content():
                write_image_file.write(chunk)
    N_DOWNLOADED += 1

OUT_PATH = "../../../Datasets/Ni_Bondha"
os.system(f"mkdir -p {OUT_PATH}")

# Load links from savefile
posts_file_path = f"./Ni_Bondha__from_1572892200_to_1638675666.txt"
all_post_links = pd.read_csv(posts_file_path)['POST_URLS'].to_numpy()
all_post_links = [link for link in all_post_links if link.endswith('.jpg')]
print(f"Eligible urls: {len(all_post_links)}")
all_post_links.sort()

N_THREADS = 20
thread_feed = [all_post_links[i*N_THREADS:(i+1)*N_THREADS] for i in range(0, int(len(all_post_links)/N_THREADS))]

# Start downloading
start_clock = time.perf_counter()
N_DOWNLOADED, count, n_exceptions = 0, 0, 0
for idx, feed in enumerate(thread_feed):
    try:
        threads = []
        for url in feed:
            count += 1
            thread = threading.Thread(target=download_if_image, args=[url, count])
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()
    except Exception as E:
        n_exceptions += N_THREADS
    # display metrics
    elapsed = time.perf_counter() - start_clock
    eta = elapsed*(len(thread_feed)-(idx+1))/(idx+1)
    print(f"  ({idx+1}/{len(thread_feed)}) | downloaded: {N_DOWNLOADED} | Errors: {n_exceptions}", end=' ')
    print(f"| Elapsed: {elapsed/60:.2f} Min. | ETA: {eta/60:.2f} Min.", end='\r')

print(f"\n\nDownloaded {N_DOWNLOADED} files.")
