import gradio as gr
from huggingface_hub import InferenceClient
import os 

# --- SETUP ---
# Ensure you have added HF_TOKEN to your Space's "Secrets" in Settings
HF_TOKEN = os.environ.get("HF_TOKEN") 

client = InferenceClient("meta-llama/Meta-Llama-3-8B-Instruct", token=HF_TOKEN)

def polymer_expert(product_name):
    # This structured system message forces the AI to be professional and organized
    system_message = (
        "You are a Polymer Science Expert. Provide a technical 'Sustainability Report' for the product mentioned. "
        "Use this EXACT format:\n\n"
        "## 🔬 Material Identification\n"
        "- **Likely Resin ID:** [1-7]\n"
        "- **Chemical Name:** [e.g. Polyethylene Terephthalate]\n\n"
        "## ♻️ Recycling Guidance\n"
        "- [Provide specific instructions for this material]\n\n"
        "## 🧪 Technical Properties\n"
        "- [Mention heat resistance, durability, or toxicity]\n\n"
        "## 💡 Sustainable Alternatives\n"
        "- [Suggest an eco-friendly switch]"
    )
    
    try:
        response = client.chat_completion(
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Analyze this specific product: {product_name}"}
            ],
            max_tokens=1500, # Increased tokens to prevent cutting off
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"### ⚠️ Error\n{str(e)}\n\n*Check if your HF_TOKEN is correctly set in the Space Secrets.*"

# --- PRO HACKATHON UI (Using Gradio Blocks) ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="green", secondary_hue="emerald")) as demo:
    gr.Markdown("# ♻️ PolymerCycle AI")
    gr.Markdown("### *Professional Decision-Support for Circular Economy & Plastic Waste*")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Textbox(
                label="Input Product", 
                placeholder="e.g., Clear soda bottle, Styrofoam tray, HDPE milk jug...",
                lines=2
            )
            analyze_btn = gr.Button("Generate Technical Report", variant="primary")
            
            gr.Examples(
                examples=["Water Bottle", "Yogurt Cup", "Styrofoam Container", "Grocery Bag"],
                inputs=input_text
            )
            
        with gr.Column(scale=2):
            output_md = gr.Markdown("### 📄 Analysis will appear here...")

    analyze_btn.click(
        fn=polymer_expert, 
        inputs=input_text, 
        outputs=output_md
    )
    
    gr.HTML("<br><hr><center>Built for the Hackathon | Powered by Llama-3 & Gradio</center>")

# Launch the app
if __name__ == "__main__":
    demo.launch()
