import requests
import gradio as gr

# API key and base URL
YOUTUBE_API_KEY = "AIzaSyCJCavrTsQmdlilAxmuo0NlCcUCwDb0RbU"
YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"

# Function to search for YouTube videos
def search_youtube_videos(query, num_results=5):
    # YouTube API search parameters
    params = {
        'part': 'snippet',
        'q': query,
        'key': YOUTUBE_API_KEY,
        'type': 'video',
        'maxResults': num_results
    }

    # Make request to YouTube API
    response = requests.get(YOUTUBE_SEARCH_URL, params=params)

    # Check if the response is successful
    if response.status_code == 200:
        video_results = response.json().get('items', [])
        video_links = []

        # Parse results
        for video in video_results:
            video_id = video['id']['videoId']
            video_title = video['snippet']['title']
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            video_links.append((video_title, video_url))

        return video_links
    else:
        return f"Error: {response.status_code}"

# Gradio Interface for Video Search
def display_videos(query, num_videos):
    videos = search_youtube_videos(query, num_videos)
    if isinstance(videos, str):  # Handle errors
        return [], videos
    else:
        # Display video titles and links
        return gr.Markdown('\n'.join([f"[{title}]({url})" for title, url in videos])), ""

# Gradio layout
with gr.Blocks() as gr_interface:
    gr.Markdown("# YouTube Video Search Engine")

    with gr.Row():
        # Input section
        with gr.Column():
            query = gr.Textbox(placeholder="Enter video search query", label="Search for YouTube Videos")
            num_videos = gr.Slider(minimum=1, maximum=10, value=5, label="Number of Videos")

            # Submit button
            submit_button = gr.Button("Search")

        # Output section
        with gr.Column():
            video_output = gr.Markdown()
            result_output = gr.Textbox(label="Result")

        # Button click handler
        submit_button.click(fn=display_videos, inputs=[query, num_videos], outputs=[video_output, result_output])

# Launch interface
gr_interface.launch()
