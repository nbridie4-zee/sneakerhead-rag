import gradio as gr
from dotenv import load_dotenv
from implementation.answer import answer_question

load_dotenv(override=True)


def format_context(docs):
    """Format retrieved context for display.."""
    result = ""
    for doc in docs:
        source = doc.metadata.get("filename", "Unknown")
        doc_type = doc.metadata.get("doc_type", "Sneaker")
        result += f"👟 Source: {source} ({doc_type})\n\n{doc.page_content}\n\n{'-'*50}\n\n"
    return result


def chat(message, history):
    """Handle user input and return assistant response."""
    history = history or []

    answer, docs = answer_question(message, history)

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})

    return "", history, format_context(docs)


def main():
    with gr.Blocks(title="Sneakerhead AI Assistant") as ui:
        gr.Markdown("""
        # 👟 Sneakerhead AI Assistant

        Your personal guide to sneaker culture, brands, and history! 
        Ask me anything about:
        - 👟 Sneaker brands (Nike, Adidas, Puma, etc.)
        - 🏀 Iconic models (Air Jordan, Yeezy, Air Max)
        - 📅 Sneaker history and releases
        - 💡 Sneaker culture and collecting

        *Disclaimer: I provide information based on available data. For the latest releases and prices, check official sources.*
        """)

        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    label="💬 Conversation",
                    height=500,
                    avatar_images=("👟", "🤖")
                )
                message = gr.Textbox(
                    placeholder="Ask about sneakers, brands, models, or history...",
                    show_label=False,
                    lines=2
                )
            with gr.Column(scale=1):
                context_box = gr.Textbox(
                    label="📚 Retrieved Knowledge Base",
                    lines=25,
                    interactive=False
                )

        message.submit(chat, inputs=[message, chatbot], outputs=[message, chatbot, context_box])

        gr.Markdown("""
        ---
        **How this works:** I search through a comprehensive sneaker knowledge base
        to provide you with accurate information about sneaker brands, models, and culture.
        """)


if __name__ == "__main__":
    main()
    ui.launch(theme=gr.themes.Soft(), css="""
        .gradio-container {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .chatbot-container {
            border-radius: 12px;
            border: 1px solid #e0e0e0;
        }
    """)