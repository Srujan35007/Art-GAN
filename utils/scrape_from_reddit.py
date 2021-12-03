import os 
import time 
import requests
from datetime import datetime
import pandas as pd


SUBREDDIT = "Ni_Bondha"
START_UTC = int(datetime(year=2019, month=11, day=7).strftime("%s")) - 48*60*60
END_UTC = int(time.time()) + 48*60*60
INTERVAL = 1500 #Minutes
temp_file = f"temp_{END_UTC}.json"
out_filepath = f"{SUBREDDIT}__from_{START_UTC}_to_{END_UTC}.txt"

print(f"Scraping IDs from {SUBREDDIT}")
print(f"Start UTC: {START_UTC}")
print(f"End UTC: {END_UTC}\n")
SCRAPE_INTERVALS = [(int(t),int(t+INTERVAL*60)) for t in range(START_UTC, END_UTC, int(INTERVAL*60))]
POST_IDS = []
count = 0
exceptions = 0
start_clock = time.perf_counter()

# Create outfile and make column headers
with open(out_filepath, 'w') as write_file:
    write_file.write(f"CREATED_UTCS,POST_IDS,POST_URLS\n")

# Start scraping
for idx, (start_utc, end_utc) in enumerate(SCRAPE_INTERVALS):
    try:
        log = ''
        url = "https://api.pushshift.io/reddit/" + \
            f"search/submission/?subreddit={SUBREDDIT}&sort=desc&sort_type=created_utc" + \
            f"&after={start_utc}&before={end_utc}&size=1000"
        # download data from api
        with open(temp_file, 'wb') as write_file:
            with requests.get(url, stream=True) as r:
                for chunk in r.iter_content(chunk_size=512):
                    write_file.write(chunk)
        # read downloaded data
        d = pd.read_json(f"{temp_file}")
        for post in d['data']:
            post_id = post['id']
            post_created_utc = post['created_utc']
            post_url = post['url']
            if post_id not in POST_IDS:
                log = log + f"{post_created_utc},{post_id},{post_url}\n"
                POST_IDS.append(post_id)
                count += 1
        # Write to storage file
        with open(out_filepath, 'a') as write_file:
            write_file.write(log)
        # Display metrics
        interval_end_clock = time.perf_counter()
        elapsed = interval_end_clock-start_clock
        eta = ((len(SCRAPE_INTERVALS)-(idx+1))/(idx+1))*elapsed
        print(f"  ({idx+1}/{len(SCRAPE_INTERVALS)}) | Posts: {count} |", end=' ')
        print(f"Elapsed: {(elapsed/60):.2f} Min. | Exceptions: {exceptions} |", end=' ')
        print(f"ETA: {(eta/60):.2f} Min.", end='\r')

    except Exception as E:
        exceptions += 1

os.system(f"rm {temp_file}")
print(f"\n\nScraped {count} Posts")
