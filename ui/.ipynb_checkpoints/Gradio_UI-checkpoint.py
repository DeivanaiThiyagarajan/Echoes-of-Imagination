import gradio as gr
from transformers import pipeline
from diffusers import DiffusionPipeline
import torch
import nltk, re
import os

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

# --- Utility: Split story into chunks of 4-5 sentences ---
def split_into_storylets(text, sentences_per_chunk=5):
    sentences = nltk.sent_tokenize(text)
    chunks = [' '.join(sentences[i:i+sentences_per_chunk]) for i in range(0, len(sentences), sentences_per_chunk)]
    return chunks


# --- Summarize to make it fit model token limits ---
def summarize_text(text):
    summary = summarizer(text, max_length=75, min_length=30, do_sample=False)[0]['summary_text']
    return summary


# --- Generate an image for a single storylet ---
def generate_for_storylet(storylet):
    summarized = summarize_text(storylet)
    image = pipe(prompt=summarized).images[0]
    return summarized, image


# --- Main generation function for full story ---
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
        
        Paste your story below. The system will divide it into 4–5 sentence storylets,
        summarize each, and generate an image that best represents the scene.
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
        # Re-run generation for existing storylets
        regenerated = []
        for b in state_value:
            summarized, image = generate_for_storylet(b["text"])
            regenerated.append({"text": b["text"], "image": image, "summary": summarized})
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
