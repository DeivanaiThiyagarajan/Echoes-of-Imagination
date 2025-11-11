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

# --- Determine device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if device.type == "cpu":
    print("⚠️  WARNING: Running on CPU will be VERY SLOW (30-60 min per image)")
    print("   Recommendation: Use GPU for practical use\n")
else:
    print("✓ GPU detected! Using GPU acceleration.\n")

# --- Load summarizer for shorter prompts (<77 tokens) ---
print("Loading summarizer...")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if torch.cuda.is_available() else -1)

# --- Load Stable Diffusion v1.5 pipeline ---
print("Loading Stable Diffusion v1.5 model...")
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    use_safetensors=True
)
pipe = pipe.to(device)

# Enable memory-efficient inference if using GPU
if device.type == "cuda":
    pipe.enable_attention_slicing()

print("✓ Models loaded successfully!")

# --- Utilities ---
def split_into_storylets(text, sentences_per_chunk=3):
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
    """Generate an image for a single story segment using Stable Diffusion v1.5."""
    try:
        # Summarize the storylet to fit token limit
        prompt = summarize_text(storylet)
        print(f"Generating image for prompt: '{prompt}'")
        
        # Generate image using Stable Diffusion v1.5
        with torch.no_grad():
            image = pipe(
                prompt=prompt,
                num_inference_steps=30,
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
    """Display story segments with generated images in a single combined view."""
    if not blocks:
        return ""
    
    html = "<div style='display: flex; flex-direction: column; gap: 30px;'>"
    
    for idx, b in enumerate(blocks, 1):
        # Convert PIL image to base64
        buffered = BytesIO()
        b["image"].save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        html += f"""
        <div style='border: 3px solid #2196F3; padding: 20px; border-radius: 10px; background-color: #000000;'>
            <h3 style='color: #FFFFFF; margin-top: 0;'>📖 Segment {idx}</h3>
            
            <div style='margin-bottom: 15px;'>
                <p><strong style='color: #FFFFFF;'>📝 Story Text:</strong></p>
                <p style='color: #FFFFFF; line-height: 1.6;'>{b['text']}</p>
            </div>
            
            <div style='margin-bottom: 15px;'>
                <p><strong style='color: #FFFFFF;'>💬 Prompt Used:</strong></p>
                <p style='color: #CCCCCC; font-style: italic;'>"{b['prompt']}"</p>
            </div>
            
            <div>
                <p><strong style='color: #FFFFFF;'>🎨 Generated Image:</strong></p>
                <img src="data:image/png;base64,{img_str}" style='max-width: 100%; height: auto; border-radius: 5px; border: 2px solid #2196F3;'><br>
            </div>
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

        # Right column: output
        with gr.Column(scale=2):
            output_html = gr.HTML(
                label="� Story Segments with Images",
                show_label=True
            )
            
            with gr.Row():
                regenerate_btn = gr.Button("🔄 Regenerate All", size="md")
                like_btn = gr.Button("👍 Like", size="md")
                dislike_btn = gr.Button("👎 Dislike", size="md")

    # --- Callbacks ---
    def on_generate(story_text):
        """Generate images for all story segments."""
        blocks = generate_story_images(story_text)
        
        # Prepare HTML with images embedded
        html = display_results(blocks)
        
        return html, blocks

    def on_regenerate(blocks):
        """Regenerate images for stored story segments."""
        if not blocks:
            return "No story data to regenerate", blocks
        
        # Reconstruct the full story
        full_story = " ".join([b["text"] for b in blocks])
        
        # Generate new images
        new_blocks = generate_story_images(full_story)
        
        # Prepare HTML with images embedded
        html = display_results(new_blocks)
        
        return html, new_blocks

    def on_clear():
        """Clear all outputs."""
        return "", []

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
        outputs=[output_html, state],
        show_progress=True
    )
    
    regenerate_btn.click(
        fn=on_regenerate, 
        inputs=state, 
        outputs=[output_html, state],
        show_progress=True
    )
    
    clear_btn.click(
        fn=on_clear, 
        outputs=[output_html, state]
    )
    
    like_btn.click(
        fn=feedback_like,
        outputs=output_html
    )
    
    dislike_btn.click(
        fn=feedback_dislike,
        outputs=output_html
    )

print("🚀 Launching Gradio interface...")
demo.launch(share=True)
