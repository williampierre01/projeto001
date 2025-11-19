
#!pip install gradio
#!pip install transformers>=4.41.2 accelerate>=0.31.0
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import time # Adicionar esta importação

time.sleep(60) # Adicionar um atraso de 5 segundos
# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="auto",
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
generation_args = { 
    "max_new_tokens": 500, 
    "return_full_text": False, 
    "temperature": 0.0, 
    "do_sample": False, 
}

# --- Configurações Iniciais ---
TITULO = "💬 Meu chat robô"
DESCRICAO = "Este é um template de interface de chatbot. Substitua a função 'responder_chatbot' pela integração com seu modelo de linguagem (LLM)."

# --- Função Principal de Resposta do Chatbot ---
def responder_chatbot(mensagem, historico):

    # O histórico do chat é ignorado neste código, mas vamos usá-lo para criar o prompt.
    
    # 1. Crie o prompt formatado
    # O modelo Phi-3-mini-4k-instruct usa um formato de conversação específico.
    messages = []
    
    # Adicione as mensagens do histórico, se houver
    for user_msg, model_msg in historico:
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if model_msg:
            messages.append({"role": "assistant", "content": model_msg})
            
    # Adicione a mensagem atual do usuário
    messages.append({"role": "user", "content": mensagem})
    
    # Use o tokenizer para aplicar o formato correto
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # === LÓGICA DO CHATBOT/MODELO CORRIGIDA ===
    try:
        # AQUI É A CORREÇÃO: use **generation_args para passar os parâmetros como kwargs
        # generator(prompt) aceita 1 argumento posicional (a prompt), 
        # e o resto como argumentos nomeados (kwargs) desempacotados
        output = generator(prompt, **generation_args)
        
        # O resultado é uma lista, onde o primeiro item é o dicionário de saída
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

print("\nIniciando interface Gradio...")
interface.launch(ssr_mode=False)









