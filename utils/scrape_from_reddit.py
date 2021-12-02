import time 
import cv2
import praw 
from secrets import reddit_scraper


reddit = praw.Reddit(
client_id = reddit_scraper['client_id'],
client_secret = reddit_scraper['client_secret'],
username = reddit_scraper['username'],
password = reddit_scraper['password'],
user_agent = reddit_scraper['user_agent']
)
print(f"User authorized")

out_data_dir = '../../../Datasets/AbstractArt'
