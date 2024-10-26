# TODO#1: Import necessary libraries
import torch
import chromadb
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import gradio as gr
import time
from sklearn.metrics.pairwise import cosine_similarity
from googleapiclient.discovery import build  # For YouTube API

# TODO#2: Setup ChromaDB
client = chromadb.Client()
collection = client.create_collection("image_collection")

# TODO#3: Load CLIP model and processor for generating image and text embeddings
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# YouTube API setup
API_KEY = "YOUR_YOUTUBE_API_KEY"  # Replace with your YouTube API key
youtube_service = build("youtube", "v3", developerKey=API_KEY)

# TODO#4: Load and preprocess images
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
    "img/A_Painter_1.jpg",
    "img/A_Place_1.jpg",
    "img/A_Structure_in_Europe_1.jpg",
    "img/An_Artist_1.jpg",
    "img/Animals_1.jpg",
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
    "img/Animals_1.jpg",
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
    "img/polar_bears_4 copy.jpg",
    "img/polar_bears_5 copy.jpg",
    "img/polar_bears_1.jpg",
    "img/polar_bears_2.jpg",
    "img/polar_bears_3.jpg",
    "img/polar_bears_4.jpg",
    "img/polar_bears_5.jpg",
]
# Preprocess images and generate embeddings
images = [Image.open(image_path) for image_path in image_paths]
inputs = processor(images=images, return_tensors="pt", padding=True)

# Measure image ingestion time
start_ingestion_time = time.time()

with torch.no_grad():
    image_embeddings = model.get_image_features(**inputs).numpy()

# Convert numpy arrays to lists and add to collection
image_embeddings = [embedding.tolist() for embedding in image_embeddings]

# Add images and metadata to the collection
collection.add(
    embeddings=image_embeddings,
    metadatas=[{"image": image_path} for image_path in image_paths],
    ids=[str(i) for i in range(len(image_paths))]
)

# Measure ingestion time and log it
end_ingestion_time = time.time()
ingestion_time = end_ingestion_time - start_ingestion_time
print(f"Image Data ingestion time: {ingestion_time:.4f} seconds")

# Define a function to calculate "accuracy" score based on cosine similarity
def calculate_accuracy(image_embedding, query_embedding):
    similarity = cosine_similarity([image_embedding], [query_embedding])[0][0]
    return similarity

# Define the vector-based image search function
def search_image(query):
    if not query.strip():
        return None, "Oops! You forgot to type something on the query input!", "", None, ""

    print(f"\nQuery: {query}")
    
    # Start query processing time measurement
    start_time = time.time()
    
    # Generate an embedding for the query text
    inputs = processor(text=query, return_tensors="pt", padding=True)
    with torch.no_grad():
        query_embedding = model.get_text_features(**inputs).numpy()
    
    # Convert the query embedding to a list
    query_embedding = query_embedding.tolist()

    # Perform vector search in the collection
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=1
    )

    # Retrieve matched image path and embedding
    result_image_path = results['metadatas'][0][0]['image']
    matched_image_index = int(results['ids'][0][0])
    matched_image_embedding = image_embeddings[matched_image_index]
    
    # Calculate accuracy score
    accuracy_score = calculate_accuracy(matched_image_embedding, query_embedding[0])

    # Measure query processing time
    end_time = time.time()
    query_time = end_time - start_time

    # Display result with accuracy, query time, and file name
    result_image = Image.open(result_image_path)
    file_name = result_image_path.split('/')[-1]
    
    # YouTube search results
    youtube_video_id, youtube_title = search_youtube(query)
    
    return result_image, f"Accuracy score: {accuracy_score:.4f}\nQuery time: {query_time:.4f} seconds", file_name, youtube_video_id, youtube_title

# YouTube video search function
def search_youtube(query):
    request = youtube_service.search().list(
        q=query,
        part="snippet",
        type="video",
        maxResults=1
    )
    response = request.execute()
    if response['items']:
        video_id = response['items'][0]['id']['videoId']
        video_title = response['items'][0]['snippet']['title']
        return video_id, video_title
    return None, "No video found"

# Suggested queries
queries = [
    "A group of polar bears",
    "A famous landmark in Paris",
    "A hot pizza fresh from the oven",
    "Food",
    "A Place",
    "A Structure in Europe",
    "Animals"
]

# Gradio Interface Layout
with gr.Blocks() as gr_interface:
    gr.Markdown("# Text-to-Image and Video Search using ChromaDB and YouTube")
    with gr.Row():
        # Left Panel
        with gr.Column():
            gr.Markdown(f"**Image Ingestion Time**: {ingestion_time:.4f} seconds")
            gr.Markdown("### Input Panel")
            
            # Input box for custom query
            custom_query = gr.Textbox(placeholder="Enter your custom query here", label="What are you looking for?")

            # Buttons for cancel and submit actions
            with gr.Row():
                submit_button = gr.Button("Submit Query")
                cancel_button = gr.Button("Cancel")

            # Suggested search phrases
            gr.Markdown("#### Suggested Search Phrases")
            with gr.Row(elem_id="button-container"):
                for query in queries:
                    gr.Button(query).click(fn=lambda q=query: q, outputs=custom_query)

        # Right Panel
        with gr.Column():
            gr.Markdown("### Retrieved Image")
            image_output = gr.Image(type="pil", label="Result Image")
            accuracy_output = gr.Textbox(label="Performance")

            gr.Markdown("### Retrieved YouTube Video")
            youtube_video = gr.Video(label="Video Result")
            youtube_title_output = gr.Textbox(label="Video Title")

        submit_button.click(fn=search_image, inputs=custom_query, outputs=[image_output, accuracy_output, youtube_video, youtube_title_output])
        cancel_button.click(fn=lambda: (None, "", None, ""), outputs=[image_output, accuracy_output, youtube_video, youtube_title_output])

# Launch the Gradio interface
gr_interface.launch()
