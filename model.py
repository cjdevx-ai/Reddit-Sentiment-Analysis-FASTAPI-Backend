from transformers import pipeline
from reddit_client import fetch_comments
import praw

reddit = praw.Reddit(
    client_id="w2BbGi9Kdkirs8aQYkTo",
    client_secret="kM5oH2oUTSD_vRb35YyMousQmruYw",
    user_agent="Sentiment App by u/Ahley-6928"
)

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    truncation=True
    )

def run_pipeline(url: str):
    # get comments from reddit_client
    comments = fetch_comments(url)
    print(f"Fetched {len(comments)} comments!")
    
    sentiment_pipeline(comments)

if __name__ == "__main__":
    test_url = "https://www.reddit.com/r/GigilAko/comments/1o22kh4/gigil_ako_caterer_no_show_tapos_ayaw_magrefund"
    run_pipeline(test_url)