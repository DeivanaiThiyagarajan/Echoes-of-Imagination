import gradio as gr
from PIL import Image
import nltk
import base64
from io import BytesIO
from transformers import pipeline
from diffusers import DiffusionPipeline
import torch
import nltk, re

os.environ["USE_TF"] = "0"

# --- Ensure sentence tokenizer is available ---
nltk.download('punkt', quiet=True)

# --- Load summarizer for shorter prompts (<77 tokens) ---
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# --- Load diffusion pipeline (placeholder model until custom one is ready) ---
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1-base",  # smaller, CPU-safe
    torch_dtype=torch.float32,
    use_safetensors=True
)
pipe.to("cpu")

torch.cuda.empty_cache()

# --- Utilities ---
def split_into_storylets(text, sentences_per_chunk=5):
    sentences = nltk.sent_tokenize(text)
    return [' '.join(sentences[i:i+sentences_per_chunk]) for i in range(0, len(sentences), sentences_per_chunk)]

# --- Summarize to make it fit model token limits ---
def summarize_text(text):
    summary = summarizer(text, max_length=75, min_length=30, do_sample=False)[0]['summary_text']
    return summary

def generate_for_storylet(storylet):
    # black dummy image
    #image = Image.new("RGB", (64, 64), color=(0, 0, 0))
    # encode image as base64 for HTML
    #buffered = BytesIO()
    #image.save(buffered, format="PNG")
    #img_str = base64.b64encode(buffered.getvalue()).decode()

    summarized = summarize_text(storylet)
    image = pipe(prompt=summarized).images[0]
    return summarized, image

def generate_story_images(story_text):
    storylets = split_into_storylets(story_text)
    output_blocks = []
    for idx, storylet in enumerate(storylets, 1):
        summarized, image = generate_for_storylet(storylet)
        block = {
            "text": storylet.strip(),
            "image": image,
            "summary": summarized
        }
        output_blocks.append(block)

    return output_blocks

# --- Display results as HTML ---
def display_results(blocks):
    html = ""
    for b in blocks:
        html += f"<h4>Story segment:</h4><p>{b['text']}</p>"
        html += f'<img src="data:image/png;base64,{b["img_str"]}" width="512" height="512"><br><br>'
    return html

# --- Gradio app ---
with gr.Blocks() as demo:
    state = gr.State([])

    with gr.Row():
        # Left column: input + buttons
        with gr.Column(scale=1):
            story_input = gr.Textbox(lines=15, placeholder="Paste your story here...", label="Enter your story")
            generate_btn = gr.Button("✨ Generate Story Sequence")
            clear_btn = gr.Button("🧹 Clear All")

        # Right column: output + regenerate + feedback
        with gr.Column(scale=2):
            output_html = gr.HTML()
            regenerate_btn = gr.Button("🔄 Regenerate All")
            with gr.Row():
                like_btn = gr.Button("👍 Like")
                dislike_btn = gr.Button("👎 Dislike")

    # --- Callbacks ---
    def on_generate(story_text):
        blocks = generate_story_images(story_text)
        state_value = blocks
        return display_results(blocks), state_value

    def on_regenerate(blocks):
        regenerated = generate_story_images(" ".join([b["text"] for b in blocks]))
        return display_results(regenerated), regenerated

    def on_clear():
        return "", []

    def feedback_like(blocks):
        return gr.Info("👍 Feedback recorded: You liked this result.")

    def feedback_dislike(blocks):
        return gr.Info("👎 Feedback recorded: You disliked this result.")

    # --- Connect buttons ---
    generate_btn.click(fn=on_generate, inputs=story_input, outputs=[output_html, state])
    regenerate_btn.click(fn=on_regenerate, inputs=state, outputs=[output_html, state])
    clear_btn.click(fn=on_clear, outputs=[output_html, state])
    like_btn.click(fn=feedback_like, inputs=state)
    dislike_btn.click(fn=feedback_dislike, inputs=state)

demo.launch()
