# TODO#1: Import necessary libraries
import torch
import chromadb
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import gradio as gr
import time
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from itertools import chain

# Load spaCy's English model for query expansion
nlp = spacy.load("en_core_web_sm")

# Synonym dictionary for enhanced query parsing
synonym_dict = {
    "dog": ["puppy", "canine", "pet"],
    "cat": ["feline", "kitten", "pet"],
    # Add more mappings as needed
}

# Function for expanding query with synonyms
def expand_query_with_synonyms(query):
    doc = nlp(query.lower())
    expanded_terms = set(chain(*[synonym_dict.get(token.text, [token.text]) for token in doc]))
    return " ".join(expanded_terms)

# TODO#2: Setup ChromaDB
client = chromadb.Client()

# Create a new collection for storing image embeddings
collection = client.create_collection("image_collection")

# TODO#3: Load CLIP model and processor for generating image and text embeddings
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# TODO#4: Load and preprocess images
image_paths = [
    "img/A_famous_landmark_in_Paris_1.jpg",
    "img/A_famous_landmark_in_Paris_2.jpg",
    "img/A_famous_landmark_in_Paris_3.jpg",
    "img/A_famous_landmark_in_Paris_4.jpg",
    "img/A_famous_landmark_in_Paris_5.jpg",
    "img/A_hot_pizza_fresh_from_the_oven_1.jpg",
    "img/A_Painter_1.jpg",
    "img/A_Place_1.jpg",
    "img/A_Structure_in_Europe_1.jpg",
    "img/An_Artist_1.jpg",
    "img/Animals_1.jpg",
    "img/Food_1.jpg",
    "img/Food_2.jpg",
    "img/Food_3.jpg",
    "img/Food_4.jpg",
    "img/Food_5.jpg",
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

# Convert numpy arrays to lists
image_embeddings = [embedding.tolist() for embedding in image_embeddings]

# Measure total ingestion time
end_ingestion_time = time.time()
ingestion_time = end_ingestion_time - start_ingestion_time

# TODO#5: Add image embeddings to the collection with metadata and display ingestion time
collection.add(
    embeddings=image_embeddings,
    metadatas=[{"image": image_path} for image_path in image_paths],
    ids=[str(i) for i in range(len(image_paths))]
)

# Log the ingestion performance
print(f"Image Data ingestion time: {ingestion_time:.4f} seconds")

# TODO#6: Create a function to calculate "accuracy" score based on cosine similarity
def calculate_accuracy(image_embedding, query_embedding):
    # Cosine similarity between query and image embeddings
    similarity = cosine_similarity([image_embedding],[query_embedding])[0][0]
    return similarity

# Define Gradio function
def search_image(query):
    # Simple validation: if the query is empty, show an error message
    if not query.strip():
        return None, "Oops! You forgot to type something on the query input!", ""

    print(f"\nOriginal Query: {query}")
    
    # Expand query with synonyms
    expanded_query = expand_query_with_synonyms(query)
    print(f"Expanded Query: {expanded_query}")
    
    # Start measuring the query processing time
    start_time = time.time()
    
    # TODO#7: Generate an embedding for the expanded query text
    inputs = processor(text=expanded_query, return_tensors="pt", padding=True)
    with torch.no_grad():
        query_embedding = model.get_text_features(**inputs).numpy()
    
    # TODO#8: Convert the query embedding from numpy array to a list
    query_embedding = query_embedding.tolist()
    
    # TODO#9: Perform a vector search in the collection
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=1
    )
    
    # TODO#10: Retrieve the matched image
    result_image_path = results['metadatas'][0][0]['image']
    matched_image_index = int(results['ids'][0][0])
    matched_image_embedding = image_embeddings[matched_image_index]
    
    # Calculate accuracy score based on cosine similarity
    accuracy_score = calculate_accuracy(matched_image_embedding, query_embedding[0])
    
    # End time for query processing
    end_time = time.time()
    query_time = end_time - start_time
    
    # TODO#11: Display result with accuracy, query time, and file name
    result_image = Image.open(result_image_path)
    file_name = result_image_path.split('/')[-1]
    
    return result_image, f"Accuracy score: {accuracy_score:.4f}\nQuery time: {query_time:.4f} seconds", file_name

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

# Function to populate the query input box with the suggested query
def populate_query(suggested_query):
    return suggested_query

# Gradio Interface Layout
with gr.Blocks() as gr_interface:
    gr.Markdown("# Text-to-Image Vector Search using ChromaDB")
    with gr.Row():
        # Left Panel
        with gr.Column():
            # TODO#12: Display the ingestion time of image embeddings
            gr.Markdown(f"**Image Ingestion Time**: {ingestion_time:.4f} seconds")
            gr.Markdown("### Input Panel")
            
            # Input box for custom query
            custom_query = gr.Textbox(placeholder="Enter your custom query here", label="What are you looking for?")

            # Buttons for cancel and submit actions
            with gr.Row():
                submit_button = gr.Button("Submit Query")
                cancel_button = gr.Button("Cancel")

            # Suggested search phrases as buttons styled like tags
            gr.Markdown("#### Suggested Search Phrases")
            with gr.Row(elem_id="button-container"):
                for query in queries:
                    gr.Button(query).click(fn=lambda q=query: q, outputs=custom_query)

        # Right Panel
        with gr.Column():
            gr.Markdown("### Retrieved Image")
            # TODO#13: Output for image result
            image_output = gr.Image(type="pil", label="Result Image")
            # Output for accuracy score and query time
            accuracy_output = gr.Textbox(label="Performance")

        # Button click handler for custom query submission
        submit_button.click(fn=search_image, inputs=custom_query, outputs=[image_output, accuracy_output])

        # Cancel button to clear the inputs
        cancel_button.click(fn=lambda: (None, ""), outputs=[image_output, accuracy_output])

# TODO#14: Launch the Gradio interface
gr_interface.launch()
