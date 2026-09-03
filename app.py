from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
import gradio as gr
import warnings

warnings.filterwarnings('ignore')


embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma(
    persist_directory="./chroma_db_medium",
    embedding_function=embedding_model,
    collection_name="rag_collection_medium"
)

llm = OllamaLLM(
    model="llama3.2",
    base_url="http://host.docker.internal:11434"
)

def answer_rag(query):
    if not query.strip():
        return "Te rog introdu o intrebare valida."

    docs_with_scores = vector_store.similarity_search_with_score(query, k=3)
    relevant_docs = [doc for doc, score in docs_with_scores if score < 0.8]

    if not relevant_docs:
        return "Nu am gasit informatii relevante in documente."

    context = "\n".join(
        [f"• {d.page_content}" for d, score in docs_with_scores])

    prompt = f"""Answer the question based strictly on the context provided below.

CONTEXT:
{context}

QUESTION:
{query}

DETAILED ANSWER:"""

    response = llm.invoke(prompt)

    return response


custom_css = """
#main-container { max-width: 850px; margin: 0 auto; padding: 25px; }
footer { display: none !important; }
.header-title { text-align: center; color: #4F46E5; font-size: 28px; font-weight: bold; margin-bottom: 20px; }
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo"), css=custom_css) as demo:
    with gr.Column(elem_id="main-container"):
        gr.HTML('<div class="header-title">LucaGBT-beta (Ollama Engine)</div>')

        with gr.Row():
            with gr.Column(scale=1):
                query_input = gr.Textbox(
                    lines=4, placeholder="Introdu intrebarea", label="Intrebare")
                submit_btn = gr.Button("Trimite", variant="primary")

            with gr.Column(scale=1):
                response_output = gr.Textbox(
                    label="Raspuns", lines=5, interactive=False)

        submit_btn.click(fn=answer_rag, inputs=query_input,
                         outputs=[response_output])
        query_input.submit(fn=answer_rag, inputs=query_input,
                           outputs=[response_output])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
