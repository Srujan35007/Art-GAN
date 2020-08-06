import time 
b = time.time()
import praw 
import urllib.request as req
from tqdm import tqdm
a = time.time()
print(f'Imports complete in {a-b} seconds')

reddit = praw.Reddit(
    client_id = 'CLIENT_ID',
    client_secret = 'CLIENT_SECRET',
    username = 'USERNAME',
    password = 'PASSWORD',
    user_agent = 'WebScraper1'
)

def is_image(string):
    string = string.lower()
    if string.endswith('jpg') or string.endswith('png') or string.endswith('jpeg'):
        return True
    else:
        return False
    
def print_iter(iterable):
    for i in range(len(iterable)):
        print(f'{i+1} => {iterable[i]}')

subs = [LIST OF SUBREDDITS]        

image_urls = []
for sub in subs:
    count = 0
    all_content = [
        reddit.subreddit(sub).top('all', limit = 1000),
        reddit.subreddit(sub).top('month', limit = 1000),
        reddit.subreddit(sub).top('week', limit = 1000),
        reddit.subreddit(sub).hot(limit = 1000),
        reddit.subreddit(sub).new(limit = 1000),
        reddit.subreddit(sub).controversial('all', limit = 1000),
        reddit.subreddit(sub).rising(limit = 1000)
                  ]
    for top_content in all_content:
        for content in top_content:
            if is_image(content.url):
                image_urls.append(content.url)
                count += 1
    print(f'{sub} => {count}')
print(f'Total images = {len(image_urls)}')
print(len(list(set(image_urls))), 'Original images')

image_urls = list(set(image_urls))
with open('./art.txt', 'w', newline='\n') as foo:
    for elem in image_urls:
        foo.writelines(str(elem)+'\n')
print('Done')
