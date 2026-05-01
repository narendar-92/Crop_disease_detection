import urllib.request
import os
import sys

def download_image(url, filename):
    urllib.request.urlretrieve(url, filename)
    print(f"Downloaded {filename}")

if __name__ == "__main__":
    download_image("https://upload.wikimedia.org/wikipedia/commons/thumb/d/d4/George-W-Bush.jpeg/400px-George-W-Bush.jpeg", "person.jpg")
