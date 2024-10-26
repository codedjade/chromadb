import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")
UNSPLASH_URL = "https://api.unsplash.com/search/photos"

# Function to download images from Unsplash
def download_images_from_unsplash(query, num_images=5, folder="unsplash_images"):
    headers = {
        "Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"
    }
    params = {
        "query": query,
        "per_page": num_images
    }
    
    response = requests.get(UNSPLASH_URL, headers=headers, params=params)
    
    if response.status_code != 200:
        raise Exception(f"Failed to fetch images from Unsplash. Status code: {response.status_code}")
    
    # Create the folder if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    image_data = response.json()
    
    image_paths = []
    for idx, image in enumerate(image_data['results']):
        image_url = image['urls']['regular']
        image_response = requests.get(image_url)

        if image_response.status_code == 200:
            image_path = os.path.join(folder, f"img_{idx + 1}.jpg")
            with open(image_path, 'wb') as f:
                f.write(image_response.content)
            image_paths.append(image_path)
            print(f"Downloaded: {image_path}")
        else:
            print(f"Failed to download image {idx + 1}")
    
    return image_paths
