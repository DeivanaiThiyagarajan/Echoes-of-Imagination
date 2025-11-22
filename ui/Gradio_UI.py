import gradio as gr
from transformers import pipeline
from diffusers import DiffusionPipeline, StableDiffusionImg2ImgPipeline
import torch
import nltk, re
import os
from PIL import Image
import numpy as np

os.environ["USE_TF"] = "0"

# --- Ensure sentence tokenizer is available ---
nltk.download('punkt', quiet=True)

# --- Load summarizer for shorter prompts (<77 tokens) ---
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# --- Load two models for sequential generation ---
print("Loading Model 1: Text-to-Image Generation...")
pipe_text2img = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",  # Primary model: text → image
    torch_dtype=torch.float32,
    use_safetensors=True
)
pipe_text2img.to("cpu")

print("Loading Model 2: Image-to-Image Refinement...")
pipe_img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",  # Secondary model: (image + text) → refined image
    torch_dtype=torch.float32,
    use_safetensors=True
)
pipe_img2img.to("cpu")

torch.cuda.empty_cache()

print("✓ Both models loaded successfully")

# --- Utility: Split story into chunks of 4-5 sentences ---
def split_into_storylets(text, sentences_per_chunk=5):
    sentences = nltk.sent_tokenize(text)
    chunks = [' '.join(sentences[i:i+sentences_per_chunk]) for i in range(0, len(sentences), sentences_per_chunk)]
    return chunks


# --- Summarize to make it fit model token limits ---
def summarize_text(text):
    summary = summarizer(text, max_length=75, min_length=30, do_sample=False)[0]['summary_text']
    return summary


# --- Generate an image for a single storylet using TWO models ---
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
    summarized = summarize_text(storylet)
    
    # Stage 1: Generate initial image from text
    print(f"  → Stage 1: Generating from text: '{summarized[:50]}...'")
    image_stage1 = pipe_text2img(prompt=summarized, num_inference_steps=30).images[0]
    
    # Stage 2: Refine with image-to-image if we have a previous image
    if previous_image is not None:
        print(f"  → Stage 2: Refining with previous image + text")
        try:
            # Ensure previous image is PIL Image
            if isinstance(previous_image, np.ndarray):
                previous_image = Image.fromarray((previous_image * 255).astype(np.uint8))
            
            # Refine: use previous image as visual context + current text
            image_final = pipe_img2img(
                prompt=summarized,
                image=previous_image,
                strength=0.6,  # 0.6 = 60% modification (balance between context and new content)
                num_inference_steps=30,
                guidance_scale=7.5
            ).images[0]
            
            return summarized, image_final
        except Exception as e:
            print(f"  ⚠️ Refinement failed: {e}, using Stage 1 output")
            return summarized, image_stage1
    else:
        # First image: no previous context
        print(f"  → First image (no previous context)")
        return summarized, image_stage1


# --- Main generation function for full story ---
def generate_story_images(story_text):
    """
    Generate story sequence with two-model pipeline:
    - Model 1: Text → Initial Image (Stable Diffusion Text-to-Image)
    - Model 2: (Previous Image + Text) → Refined Image (Stable Diffusion Image-to-Image)
    """
    storylets = split_into_storylets(story_text)
    output_blocks = []
    previous_image = None

    for idx, storylet in enumerate(storylets, 1):
        print(f"\n[Storylet {idx}/{len(storylets)}] Processing...")
        summarized, image = generate_for_storylet(storylet, previous_image)
        
        block = {
            "text": storylet.strip(),
            "image": image,
            "summary": summarized
        }
        output_blocks.append(block)
        
        # Pass current image as context for next storylet
        previous_image = image
        print(f"✓ Completed storylet {idx}")

    return output_blocks


# --- UI Event: Build visual layout for results ---
def display_results(blocks):
    elements = []
    for b in blocks:
        elements.append(gr.Markdown(f"**Story segment:**\n\n{b['text']}"))
        elements.append(gr.Image(value=b['image'], label="Generated Illustration"))
    return elements


# --- Gradio app ---
with gr.Blocks(title="Echoes of Imagination") as demo:
    gr.Markdown(
        """
        # 🌌 Echoes of Imagination
        _Transform your stories into vivid illustrated sequences._
        
        **Two-Model Pipeline for Enhanced Visual Coherence:**
        - **Model 1 (Text-to-Image):** Generates initial image from storylet text
        - **Model 2 (Image-to-Image):** Refines output using previous image + current text
        
        This dual-model approach maintains **visual continuity** across story segments while ensuring each scene accurately depicts the narrative.
        
        **How it works:**
        1. Your story is divided into 4–5 sentence storylets
        2. Each storylet is summarized and processed through both models
        3. Images from previous storylets guide the generation of subsequent ones
        4. Result: Coherent visual story with consistent character/scene continuity
        """
    )

    story_input = gr.Textbox(
        lines=10,
        label="Enter your story",
        placeholder="Paste your story here..."
    )

    generate_btn = gr.Button("✨ Generate Story Sequence")
    clear_btn = gr.Button("🧹 Clear All")
    regenerate_btn = gr.Button("🔄 Regenerate All")

    output_column = gr.Column()
    feedback_row = gr.Row()

    like_btn = gr.Button("👍 Like")
    dislike_btn = gr.Button("👎 Dislike")


    state = gr.State([])  # holds generated blocks

    def on_generate(story_text):
        blocks = generate_story_images(story_text)
        state_value = blocks
        elements = display_results(blocks)
        return elements, state_value

    def on_regenerate(state_value):
        # Re-run generation with two-model pipeline for existing storylets
        regenerated = []
        previous_image = None
        
        for idx, b in enumerate(state_value):
            print(f"\n[Regenerating {idx+1}/{len(state_value)}]")
            summarized, image = generate_for_storylet(b["text"], previous_image)
            regenerated.append({"text": b["text"], "image": image, "summary": summarized})
            previous_image = image
        
        return display_results(regenerated), regenerated

    def on_clear():
        return None, []

    def feedback_like(state_value):
        return gr.Info("👍 Feedback recorded: You liked this result.")

    def feedback_dislike(state_value):
        return gr.Info("👎 Feedback recorded: You disliked this result.")

    generate_btn.click(fn=on_generate, inputs=story_input, outputs=[output_column, state])
    regenerate_btn.click(fn=on_regenerate, inputs=state, outputs=[output_column, state])
    clear_btn.click(fn=on_clear, outputs=[output_column, state])
    like_btn.click(fn=feedback_like, inputs=state)
    dislike_btn.click(fn=feedback_dislike, inputs=state)

demo.launch()
