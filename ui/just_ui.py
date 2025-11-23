import gradio as gr
from PIL import Image
import nltk
import base64
from io import BytesIO
from transformers import pipeline
from diffusers import DiffusionPipeline, StableDiffusionImg2ImgPipeline
import torch
import numpy as np
import os

os.environ["USE_TF"] = "0"

# --- Ensure sentence tokenizer is available ---
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
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

# --- Load two models for sequential generation ---
print("Loading Model 1: Text-to-Image Generation...")
pipe_text2img = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32,
    use_safetensors=True
)
pipe_text2img.to(device)

print("Loading Model 2: Image-to-Image Refinement...")
pipe_img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32,
    use_safetensors=True
)
pipe_img2img.to(device)

# Enable memory-efficient inference if using GPU
if device.type == "cuda":
    pipe_text2img.enable_attention_slicing()
    pipe_img2img.enable_attention_slicing()

print("✓ Both models loaded successfully!")

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
        if len(text.split()) < 50:
            return text
        
        # Summarize to 30-75 tokens (roughly 6-15 words)
        summary = summarizer(text, max_length=75, min_length=30, do_sample=False)[0]['summary_text']
        return summary
    except Exception as e:
        print(f"Summarization error: {e}")
        # Fallback: just return first 75 characters
        return text[:75] if len(text) > 75 else text

def generate_for_storylet(storylet, previous_image=None):
    """
    Two-stage image generation:
    1. Model 1 (Text-to-Image): Generate initial image from storylet text
    2. Model 2 (Image-to-Image): Refine using previous image + current text (if available)
    
    Args:
        storylet: Story text for current segment
        previous_image: Previous generated image (for refinement), or None for first image
    
    Returns:
        tuple: (summarized_text, final_image)
    """
    try:
        # Summarize the storylet to fit token limit
        summarized = summarize_text(storylet)
        print(f"  → Stage 1: Generating from text: '{summarized[:50]}...'", flush=True)
        
        # Stage 1: Generate initial image from text using Model 1
        with torch.no_grad():
            image_stage1 = pipe_text2img(
                prompt=summarized,
                num_inference_steps=50,
                guidance_scale=7.5,
                height=512,
                width=512
            ).images[0]
        
        print(f"  ✓ Stage 1 complete", flush=True)
        
        # Stage 2: Refine with image-to-image if we have a previous image
        if previous_image is not None:
            print(f"  → Stage 2: Refining with previous image + text", flush=True)
            try:
                # Ensure previous image is PIL Image
                if isinstance(previous_image, np.ndarray):
                    previous_image = Image.fromarray((previous_image * 255).astype(np.uint8))
                
                # Refine: use previous image as visual context + current text using Model 2
                with torch.no_grad():
                    image_final = pipe_img2img(
                        prompt=summarized,
                        image=previous_image,
                        strength=0.6,  # 0.6 = 60% modification (balance between context and new content)
                        num_inference_steps=50,
                        guidance_scale=7.5
                    ).images[0]
                
                print(f"  ✓ Stage 2 complete", flush=True)
                return summarized, image_final
            except Exception as e:
                print(f"  ⚠️ Refinement failed: {e}, using Stage 1 output", flush=True)
                return summarized, image_stage1
        else:
            # First image: no previous context
            print(f"  → First image (no previous context)", flush=True)
            return summarized, image_stage1
    except Exception as e:
        print(f"Error generating image: {e}", flush=True)
        # Return error image
        error_image = Image.new("RGB", (512, 512), color=(100, 100, 100))
        return "Error generating image", error_image

def generate_story_images(story_text):
    """Generate images for all story segments with two-model sequential pipeline."""
    if not story_text.strip():
        return []
    
    storylets = split_into_storylets(story_text)
    output_blocks = []
    previous_image = None
    
    print(f"\n📖 Total segments to generate: {len(storylets)}", flush=True)
    
    for idx, storylet in enumerate(storylets, 1):
        print(f"\n[Segment {idx}/{len(storylets)}] Processing...", flush=True)
        summarized, image = generate_for_storylet(storylet, previous_image)
        
        block = {
            "text": storylet.strip(),
            "image": image,
            "prompt": summarized
        }
        output_blocks.append(block)
        
        # Pass current image as context for next storylet
        previous_image = image
        print(f"✓ Completed segment {idx}/{len(storylets)}", flush=True)

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
with gr.Blocks(title="Story to Image Generator") as demo:
    gr.Markdown("""
    # 🎨 Story to Image Generator
    Transform your story into a visual sequence!
    
    **Dual-Model Sequential Pipeline:**
    - **Model 1 (Text-to-Image):** Generates initial image from story segment
    - **Model 2 (Image-to-Image):** Refines output using previous image + current text
    
    This two-model approach maintains **visual continuity** across story segments while ensuring each scene accurately reflects the narrative.
    """)
    
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
                label="📖 Story Segments with Images",
                show_label=True
            )
            
            with gr.Row():
                regenerate_btn = gr.Button("🔄 Regenerate All", size="md")
                like_btn = gr.Button("👍 Like", size="md")
                dislike_btn = gr.Button("👎 Dislike", size="md")
            
            # Add review textbox
            review_input = gr.Textbox(
                lines=4,
                placeholder="Share your feedback about the generated images (optional)...",
                label="📝 Write a Review",
                show_label=True
            )
            submit_review_btn = gr.Button("📤 Submit Review", size="md", variant="secondary")

    # --- Callbacks ---
    def on_generate(story_text):
        """Generate images for all story segments."""
        print("\n" + "="*60, flush=True)
        print("🔄 STARTING GENERATION", flush=True)
        print("="*60, flush=True)
        
        blocks = generate_story_images(story_text)
        
        print("="*60, flush=True)
        print("✅ GENERATION COMPLETE", flush=True)
        print("="*60 + "\n", flush=True)
        
        # Prepare HTML with images embedded
        html = display_results(blocks)
        
        return html, blocks

    def on_regenerate(blocks):
        """Regenerate images for stored story segments with two-model pipeline."""
        if not blocks:
            return "No story data to regenerate", blocks
        
        # Regenerate with two-model pipeline for existing storylets
        regenerated = []
        previous_image = None
        
        for idx, b in enumerate(blocks, 1):
            print(f"\n[Regenerating {idx}/{len(blocks)}]")
            summarized, image = generate_for_storylet(b["text"], previous_image)
            regenerated.append({
                "text": b["text"], 
                "image": image, 
                "prompt": summarized
            })
            previous_image = image
        
        # Prepare HTML with images embedded
        html = display_results(regenerated)
        
        return html, regenerated

    def on_clear():
        """Clear all outputs."""
        return "", []

    def feedback_like():
        """Handle like feedback."""
        return "👍 Feedback recorded: You liked this result!"

    def feedback_dislike():
        """Handle dislike feedback."""
        return "👎 Feedback recorded: You disliked this result!"
    
    def submit_review(review_text):
        """Handle review submission."""
        if not review_text.strip():
            return "⚠️ Please write a review before submitting!"
        
        # Save review to file
        import json
        from datetime import datetime
        
        review_data = {
            "timestamp": datetime.now().isoformat(),
            "review": review_text.strip()
        }
        
        # Append to reviews file
        reviews_file = "reviews.jsonl"
        try:
            with open(reviews_file, "a") as f:
                f.write(json.dumps(review_data) + "\n")
            return f"✅ Review submitted! Thank you for your feedback.\n\n📝 Your review:\n{review_text}"
        except Exception as e:
            return f"❌ Error saving review: {e}"

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
    
    submit_review_btn.click(
        fn=submit_review,
        inputs=review_input,
        outputs=output_html
    )

print("🚀 Launching Gradio interface...")
demo.launch(share=True)
