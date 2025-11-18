
#!pip install gradio
#!pip install transformers>=4.41.2 accelerate>=0.31.0

import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=False,
)

# Create a pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    max_new_tokens=50,
    do_sample=False,
)

# --- Configurações Iniciais ---
TITULO = "💬 Meu chat robô"
DESCRICAO = "Este é um template de interface de chatbot. Substitua a função 'responder_chatbot' pela integração com seu modelo de linguagem (LLM)."

# --- Função Principal de Resposta do Chatbot ---

def responder_chatbot(mensagem, historico):


    # === LÓGICA DO CHATBOT/MODELO ===
    #
    # Use o modelo LLM carregado anteriormente (certifique-se de que 'generator' esteja acessível)
    try:
        output = generator(mensagem)
        resposta = output[0]['generated_text']
    except Exception as e:
        print(f"Erro ao chamar o modelo LLM: {e}")
        resposta = "Desculpe, houve um erro ao gerar a resposta. Por favor, tente novamente."

    return resposta

# --- Definição da Interface Gradio ---

# O gr.ChatInterface é o componente mais recomendado para criar chatbots
interface = gr.ChatInterface(
    fn=responder_chatbot,  # A função Python que o chatbot irá chamar
    title=TITULO,
    description=DESCRICAO,
    # Personalização dos botões
    submit_btn="Enviar Mensagem",
    # undo_btn="Desfazer Última Ação", # Removido pois não é um argumento válido
    # clear_btn="Limpar Histórico", # Removido pois não é um argumento válido
    # Exemplos para o usuário começar rapidamente
    examples=[
        ["O que é Gradio?"],
        ["Qual a sua função principal?"],
        ["Olá, bom dia!"]
    ]
)

# --- Lançamento da Aplicação ---

if __name__ == "__main__":
    # O .launch() inicia o servidor web local
    # Remova o 'if __name__ == "__main__":' se for usar no Hugging Face Spaces
    print("\nIniciando interface Gradio...")
    interface.launch()