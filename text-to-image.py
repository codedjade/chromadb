# TODO#1: Import necessary libraries
import os
import requests
import torch
import chromadb
from PIL import Image
import gradio as gr
import time
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
from googleapiclient.discovery import build

# Folder to store downloaded images
local_image_folder = "downloaded_images"
if not os.path.exists(local_image_folder):
    os.makedirs(local_image_folder)

# YouTube API Key
YOUTUBE_API_KEY = "AIzaSyCJCavrTsQmdlilAxmuo0NlCcUCwDb0RbU"
youtube_service = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

# Setup ChromaDB
client = chromadb.Client()
collection = client.create_collection("image_collection")

# Load CLIP model and processor for generating image and text embeddings
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load and preprocess images
# Ensure your dataset images are accessible from these paths
image_paths = [
    "img/A_famous_landmark_in_Paris_1.jpg",
    "img/A_famous_landmark_in_Paris_2.jpg",
    "img/A_famous_landmark_in_Paris_3.jpg",
    "img/A_famous_landmark_in_Paris_4.jpg",
    "img/A_famous_landmark_in_Paris_5.jpg",
    "img/A_famous_landmark_in_Paris_1 copy.jpg",
    "img/A_famous_landmark_in_Paris_2 copy.jpg",
    "img/A_famous_landmark_in_Paris_3 copy.jpg",
    "img/A_famous_landmark_in_Paris_4 copy.jpg",
    "img/A_famous_landmark_in_Paris_5 copy.jpg",
    "img/A_hot_pizza_fresh_from_the_oven_1.jpg",
    "img/A_hot_pizza_fresh_from_the_oven_1 copy.jpg",
    "img/A_hot_pizza_fresh_from_the_oven_2.jpg",
    "img/A_hot_pizza_fresh_from_the_oven_3.jpg",
    "img/A_hot_pizza_fresh_from_the_oven_4.jpg",
    "img/A_hot_pizza_fresh_from_the_oven_5.jpg",
    "img/A_Painter_1.jpg",
    "img/A_Structure_in_Europe_1.jpg",
    "img/A_Structure_in_Europe_1 copy.jpg",
    "img/A_Structure_in_Europe_2.jpg",
    "img/A_Structure_in_Europe_3.jpg",
    "img/A_Structure_in_Europe_4.jpg",
    "img/An_Artist_1.jpg",
    "img/Food_1 copy.jpg",
    "img/Food_2 copy.jpg",
    "img/Food_3 copy.jpg",
    "img/Food_4 copy.jpg",
    "img/Food_5 copy.jpg",
    "img/Food_1.jpg",
    "img/Food_2.jpg",
    "img/Food_3.jpg",
    "img/Food_4.jpg",
    "img/Food_5.jpg",
    "img/Animals_1 copy.jpg", 
    "img/Animals_2.jpg",
    "img/Animals_3.jpg",
    "img/Animals_4.jpg",
    "img/Animals_5.jpg",
    "img/hungry_people_1.jpg",
    "img/img_1.jpg",
    "img/img_2.jpg",
    "img/img_3.jpg",
    "img/img_4.jpg",
    "img/img_5.jpg",
    "img/img_6.jpg",
    "img/img_7.jpg",
    "img/img_8.jpg",
    "img/img_9.jpg",
    "img/img_10.jpg",
    "img/polar_bears_1 copy.jpg",
    "img/polar_bears_2 copy.jpg",
    "img/polar_bears_3 copy.jpg",
    "img/polar_bears_1.jpg",
    "img/polar_bears_2.jpg",
    "img/polar_bears_3.jpg",
    "img/polar_bears_4.jpg",
    "img/polar_bears_5.jpg",
]

# Preprocess images and generate embeddings
images = [Image.open(image_path) for image_path in image_paths]
inputs = processor(images=images, return_tensors="pt", padding=True)

with torch.no_grad():
    image_embeddings = model.get_image_features(**inputs).numpy()
image_embeddings = [embedding.tolist() for embedding in image_embeddings]

# Add image embeddings to the collection
collection.add(
    embeddings=image_embeddings,
    metadatas=[{"image": image_path} for image_path in image_paths],
    ids=[str(i) for i in range(len(image_paths))]
)

# Function for calculating cosine similarity for accuracy score
def calculate_accuracy(image_embedding, query_embedding):
    return cosine_similarity([image_embedding], [query_embedding])[0][0]

# Vector-based image search
def search_images(query, num_results=3):
    inputs = processor(text=query, return_tensors="pt", padding=True)
    with torch.no_grad():
        query_embedding = model.get_text_features(**inputs).numpy().tolist()
    
    results = collection.query(query_embeddings=query_embedding, n_results=num_results)
    result_images = [Image.open(res['image']) for res in results['metadatas'][0]]
    scores = [calculate_accuracy(image_embeddings[int(res_id)], query_embedding[0]) for res_id in results['ids'][0]]
    
    return result_images, scores

# YouTube video search
def search_youtube_videos(query):
    response = youtube_service.search().list(part="snippet", q=query, type="video", maxResults=5).execute()
    videos = response.get('items', [])
    video_iframes = []
    
    for video in videos:
        video_id = video['id']['videoId']
        video_title = video['snippet']['title']
        video_iframe = f'<iframe width="560" height="315" src="https://www.youtube.com/embed/{video_id}" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>'
        video_iframes.append(video_iframe)
    
    return video_iframes

# Define Gradio function for handling the multimedia search
def multimedia_search(query, num_images, search_type):
    if search_type == "Images":
        retrieved_images, scores = search_images(query, num_images)
        captions = [f"Accuracy score: {score:.4f}" for score in scores]
        return retrieved_images, captions
    elif search_type == "Videos":
        videos = search_youtube_videos(query)
        return None, "\n".join(videos)  # Return embedded video iframes

# Suggested queries
queries = [
    "polar bears",
    "A famous landmark in Paris",
    "A hot pizza fresh from the oven",
    "Food",
    "A Place",
    "A Structure in Europe",
    "Animals"
]

# Function to populate the query input box with the suggested query
def populate_query(suggested_query):
    return suggested_query

# Gradio Interface Layout
with gr.Blocks() as gr_interface:
    gr.Markdown("# Multimedia Search (Images & Videos)")
    with gr.Row():
        # Left Panel
        with gr.Column():
            gr.Markdown("### Input Panel")
            
            # Input box for custom query
            custom_query = gr.Textbox(placeholder="Enter your custom query here", label="What are you looking for?")

            # Slider for number of images to retrieve
            num_images_slider = gr.Slider(minimum=1, maximum=5, value=1, step=1, label="Number of Images")

            # Dropdown for selecting search type
            search_type = gr.Dropdown(choices=["Images", "Videos"], label="Search Type", value="Images")

            # Buttons for cancel and submit actions
            with gr.Row():
                submit_button = gr.Button("Submit Query")
                cancel_button = gr.Button("Cancel")

            # Suggested search phrases as buttons styled like tags
            gr.Markdown("#### Suggested Search Phrases")
            with gr.Row(elem_id="button-container"):
                for query in queries:
                    gr.Button(query).click(fn=lambda q=query: populate_query(q), outputs=custom_query)

        # Right Panel
        with gr.Column():
            gr.Markdown("### Retrieved Content")
            image_output = gr.Gallery(label="Result Images", show_label=True, elem_id="gallery", scale=2)
            result_output = gr.HTML()  # Changed from Textbox to HTML to display iframes for videos

        # Button click handler for custom query submission
        submit_button.click(fn=multimedia_search, inputs=[custom_query, num_images_slider, search_type], outputs=[image_output, result_output])

        # Live search for images
        custom_query.change(fn=lambda query: search_images(query, num_images_slider.value) if search_type == "Images" else ([], []), 
                            inputs=[custom_query], 
                            outputs=[image_output, result_output])

        # Cancel button to clear the inputs
        cancel_button.click(fn=lambda: (None, ""), outputs=[image_output, result_output])

# TODO#4: Launch the Gradio interface
gr_interface.launch(share=True)
