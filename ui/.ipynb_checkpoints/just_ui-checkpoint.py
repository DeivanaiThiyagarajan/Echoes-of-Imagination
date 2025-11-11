import gradio as gr
from PIL import Image
import nltk
import base64
from io import BytesIO
from transformers import pipeline
from diffusers import StableDiffusionPipeline
import torch
import os

os.environ["USE_TF"] = "0"

# --- Ensure sentence tokenizer is available ---
nltk.download('punkt', quiet=True)

# --- Determine device (GPU if available, else CPU) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Load summarizer for shorter prompts (<77 tokens) ---
print("Loading summarizer...")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if torch.cuda.is_available() else -1)

# --- Load Stable Diffusion pipeline for text-to-image generation ---
print("Loading Stable Diffusion model...")
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",  # Lightweight and reliable
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    use_safetensors=True
)
pipe = pipe.to(device)

# Enable memory-efficient inference if using GPU
if device.type == "cuda":
    pipe.enable_attention_slicing()  # Reduces memory usage

print("✓ Models loaded successfully!")

# --- Utilities ---
def split_into_storylets(text, sentences_per_chunk=5):
    """Split text into logical chunks for image generation."""
    sentences = nltk.sent_tokenize(text)
    return [' '.join(sentences[i:i+sentences_per_chunk]) for i in range(0, len(sentences), sentences_per_chunk)]

# --- Summarize to make it fit model token limits ---
def summarize_text(text):
    """Summarize text to fit within CLIP token limit (77 tokens)."""
    try:
        # If text is too short, just return it as is
        if len(text.split()) < 20:
            return text
        
        # Summarize to 30-75 tokens (roughly 6-15 words)
        summary = summarizer(text, max_length=75, min_length=30, do_sample=False)[0]['summary_text']
        return summary
    except Exception as e:
        print(f"Summarization error: {e}")
        # Fallback: just return first 75 characters
        return text[:75] if len(text) > 75 else text

def generate_for_storylet(storylet):
    """Generate an image for a single story segment using Stable Diffusion."""
    try:
        # Summarize the storylet to fit token limit
        prompt = summarize_text(storylet)
        print(f"Generating image for prompt: '{prompt}'")
        
        # Generate image using the pretrained model
        with torch.no_grad():
            image = pipe(
                prompt=prompt,
                num_inference_steps=30,  # Balance between speed and quality
                guidance_scale=7.5,
                height=512,
                width=512
            ).images[0]
        
        return prompt, image
    except Exception as e:
        print(f"Error generating image: {e}")
        # Return error image
        error_image = Image.new("RGB", (512, 512), color=(100, 100, 100))
        return "Error generating image", error_image

def generate_story_images(story_text):
    """Generate images for all story segments."""
    if not story_text.strip():
        return []
    
    storylets = split_into_storylets(story_text)
    output_blocks = []
    
    for idx, storylet in enumerate(storylets, 1):
        print(f"Processing segment {idx}/{len(storylets)}...")
        prompt, image = generate_for_storylet(storylet)
        
        block = {
            "text": storylet.strip(),
            "image": image,
            "prompt": prompt
        }
        output_blocks.append(block)

    return output_blocks

# --- Display results as HTML + Images ---
def display_results(blocks):
    """Display story segments with generated images."""
    if not blocks:
        return ""
    
    html = "<div style='display: grid; gap: 20px;'>"
    for idx, b in enumerate(blocks, 1):
        html += f"""
        <div style='border: 2px solid #ddd; padding: 15px; border-radius: 8px;'>
            <h3>📖 Segment {idx}</h3>
            <p><strong>Story:</strong> {b['text']}</p>
            <p><strong>Prompt:</strong> <em>{b['prompt']}</em></p>
            <img src="{b['image_url']}" style='max-width: 100%; height: auto; border-radius: 5px;'><br><br>
        </div>
        """
    html += "</div>"
    return html

# --- Gradio app ---
with gr.Blocks(title="Story to Image Generator", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎨 Story to Image Generator\nTransform your story into a visual sequence!")
    
    state = gr.State([])

    with gr.Row():
        # Left column: input + buttons
        with gr.Column(scale=1):
            story_input = gr.Textbox(
                lines=15, 
                placeholder="Paste your story here...", 
                label="📝 Enter your story"
            )
            generate_btn = gr.Button("✨ Generate Story Sequence", size="lg", variant="primary")
            clear_btn = gr.Button("🧹 Clear All", size="lg")

        # Right column: output gallery
        with gr.Column(scale=2):
            output_gallery = gr.Gallery(
                label="📸 Generated Images",
                show_label=True,
                columns=1,
                rows=2,
                object_fit="scale-down",
                height="auto"
            )
            output_info = gr.Textbox(label="� Story Segments & Prompts", lines=5, interactive=False)
            
            with gr.Row():
                regenerate_btn = gr.Button("🔄 Regenerate All", size="md")
                like_btn = gr.Button("👍 Like", size="md")
                dislike_btn = gr.Button("👎 Dislike", size="md")

    # --- Callbacks ---
    def on_generate(story_text):
        """Generate images for all story segments."""
        blocks = generate_story_images(story_text)
        
        # Prepare gallery data
        images = [b["image"] for b in blocks]
        
        # Prepare info text
        info_text = ""
        for idx, b in enumerate(blocks, 1):
            info_text += f"Segment {idx}:\n"
            info_text += f"Story: {b['text'][:80]}...\n"
            info_text += f"Prompt: {b['prompt']}\n\n"
        
        return images, info_text, blocks

    def on_regenerate(blocks):
        """Regenerate images for stored story segments."""
        if not blocks:
            return [], "No story data to regenerate", blocks
        
        # Reconstruct the full story
        full_story = " ".join([b["text"] for b in blocks])
        
        # Generate new images
        new_blocks = generate_story_images(full_story)
        
        # Prepare gallery data
        images = [b["image"] for b in new_blocks]
        
        # Prepare info text
        info_text = ""
        for idx, b in enumerate(new_blocks, 1):
            info_text += f"Segment {idx}:\n"
            info_text += f"Story: {b['text'][:80]}...\n"
            info_text += f"Prompt: {b['prompt']}\n\n"
        
        return images, info_text, new_blocks

    def on_clear():
        """Clear all outputs."""
        return [], "", []

    def feedback_like():
        """Handle like feedback."""
        return "👍 Feedback recorded: You liked this result!"

    def feedback_dislike():
        """Handle dislike feedback."""
        return "👎 Feedback recorded: You disliked this result!"

    # --- Connect buttons ---
    generate_btn.click(
        fn=on_generate, 
        inputs=story_input, 
        outputs=[output_gallery, output_info, state],
        show_progress=True
    )
    
    regenerate_btn.click(
        fn=on_regenerate, 
        inputs=state, 
        outputs=[output_gallery, output_info, state],
        show_progress=True
    )
    
    clear_btn.click(
        fn=on_clear, 
        outputs=[output_gallery, output_info, state]
    )
    
    like_btn.click(
        fn=feedback_like,
        outputs=output_info
    )
    
    dislike_btn.click(
        fn=feedback_dislike,
        outputs=output_info
    )

print("🚀 Launching Gradio interface...")
demo.launch(share=True)
