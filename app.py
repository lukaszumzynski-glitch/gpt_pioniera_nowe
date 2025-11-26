import json
from pathlib import Path
import streamlit as st
from openai import OpenAI, AuthenticationError
from dotenv import load_dotenv
import base64
import os
import bcrypt
from datetime import datetime
import tomllib
import sys

# Używamy tomllib (Python 3.11+) lub zainstaluj pip install toml
try:
    import tomllib as toml
except ImportError:
    import toml

# --- Konfiguracja i Ceny ---
model_pricings = {
    "gpt-4o": {
        "input_tokens": 5.00 / 1_000_000,  # per token
        "output_tokens": 15.00 / 1_000_000,  # per token
    },
    "gpt-4o-mini": {
        "input_tokens": 0.150 / 1_000_000,  # per token
        "output_tokens": 0.600 / 1_000_000,  # per token
    }
}
MODEL = "gpt-4o-mini"
USD_TO_PLN = 3.97
PRICING = model_pricings[MODEL]

# --- Funkcje Pomocnicze ---
def img_to_bytes(img_path):
    """Konwertuje obraz na ciąg bajtów zakodowany w Base64."""
    if Path(img_path).exists():
        img_bytes = Path(img_path).read_bytes()
        encoded = base64.b64encode(img_bytes).decode()
        return encoded
    return ""

def init_openai_client():
    if "openai_client" not in st.session_state:
        api_key = st.session_state.get("openai_api_key")
        if api_key:
            try:
                st.session_state.openai_client = OpenAI(api_key=api_key)
                # Testujemy klucz, żeby upewnić się, że działa
                # st.session_state.openai_client.models.list() 
            except AuthenticationError:
                st.error("Podany klucz API OpenAI jest nieprawidłowy. Sprawdź plik secret.toml.")
                st.session_state["logged_in"] = False
                st.session_state.pop("openai_api_key", None)
                st.rerun()
            except Exception as e:
                st.error(f"Wystąpił błąd podczas inicjalizacji klienta OpenAI: {e}")
        else:
            st.warning("Klucz OpenAI API nie został znaleziony w secret.toml.")
            return None
    return st.session_state.get("openai_client")

def chatbot_reply(user_prompt, memory, openai_client_instance):
    # ... (logika chatbot_reply pozostaje taka sama, używa przekazanego klienta) ...
    messages = [
        {
            "role": "system",
            "content": st.session_state["chatbot_personality"],
        },
    ]
    for message in memory:
        messages.append({"role": message["role"], "content": message["content"]})

    messages.append({"role": "user", "content": user_prompt})

    response = openai_client_instance.chat.completions.create(
        model=MODEL,
        messages=messages
    )
    usage = {}
    if response.usage:
        usage = {
            "completion_tokens": response.usage.completion_tokens,
            "prompt_tokens": response.usage.prompt_tokens,
            "total_tokens": response.usage.total_tokens,
        }

    return {
        "role": "assistant",
        "content": response.choices[0].message.content,
        "usage": usage,
    }

#
# CONVERSATION HISTORY AND DATABASE (bez zmian)
#
DEFAULT_PERSONALITY = """
Jesteś pomocnikiem, który odpowiada na wszystkie pytania użytkownika.
Odpowiadaj na pytania w sposób zwięzły i zrozumiały.
""".strip()

DB_PATH = Path("db")
DB_CONVERSATIONS_PATH = DB_PATH / "conversations"

def load_conversation_to_state(conversation):
    st.session_state["id"] = conversation["id"]
    st.session_state["name"] = conversation["name"]
    st.session_state["messages"] = conversation["messages"]
    st.session_state["chatbot_personality"] = conversation["chatbot_personality"]
    st.session_state["new_conversation_name_input"] = conversation["name"] # Ustawiamy wartość początkową inputu

def load_current_conversation():
    if not DB_PATH.exists():
        DB_PATH.mkdir()
        DB_CONVERSATIONS_PATH.mkdir()
        conversation_id = 1
        conversation = {
            "id": conversation_id,
            "name": "Konwersacja 1",
            "chatbot_personality": DEFAULT_PERSONALITY,
            "messages": [],
        }
        with open(DB_CONVERSATIONS_PATH / f"{conversation_id}.json", "w") as f:
            f.write(json.dumps(conversation))
        with open(DB_PATH / "current.json", "w") as f:
            f.write(json.dumps({"current_conversation_id": conversation_id,}))
    else:
        with open(DB_PATH / "current.json", "r") as f:
            data = json.loads(f.read())
            conversation_id = data["current_conversation_id"]
        with open(DB_CONVERSATIONS_PATH / f"{conversation_id}.json", "r") as f:
            conversation = json.loads(f.read())

    load_conversation_to_state(conversation)


def save_current_conversation_messages():
    conversation_id = st.session_state["id"]
    new_messages = st.session_state["messages"]
    with open(DB_CONVERSATIONS_PATH / f"{conversation_id}.json", "r") as f:
        conversation = json.loads(f.read())
    with open(DB_CONVERSATIONS_PATH / f"{conversation_id}.json", "w") as f:
        f.write(json.dumps({**conversation, "messages": new_messages,}))


def save_current_conversation_name():
    conversation_id = st.session_state["id"]
    # Pobieramy wartość z pola tekstowego w sidebarze
    new_conversation_name = st.session_state["new_conversation_name_input"]

    with open(DB_CONVERSATIONS_PATH / f"{conversation_id}.json", "r") as f:
        conversation = json.loads(f.read())

    with open(DB_CONVERSATIONS_PATH / f"{conversation_id}.json", "w") as f:
        f.write(json.dumps({**conversation, "name": new_conversation_name,}))
    
    # Aktualizujemy st.session_state, żeby nazwa wyświetlana w głównym widoku też się zmieniła od razu
    st.session_state["name"] = new_conversation_name


def save_current_conversation_personality():
    conversation_id = st.session_state["id"]
    new_chatbot_personality = st.session_state["new_chatbot_personality"]
    with open(DB_CONVERSATIONS_PATH / f"{conversation_id}.json", "r") as f:
        conversation = json.loads(f.read())
    with open(DB_CONVERSATIONS_PATH / f"{conversation_id}.json", "w") as f:
        f.write(json.dumps({**conversation, "chatbot_personality": new_chatbot_personality,}))


def create_new_conversation():
    conversation_ids = []
    for p in DB_CONVERSATIONS_PATH.glob("*.json"):
        conversation_ids.append(int(p.stem))
    conversation_id = max(conversation_ids) + 1
    personality = DEFAULT_PERSONALITY
    if "chatbot_personality" in st.session_state and st.session_state["chatbot_personality"]:
        personality = st.session_state["chatbot_personality"]

    conversation = {
        "id": conversation_id,
        "name": f"Konwersacja {conversation_id}",
        "chatbot_personality": personality,
        "messages": [],
    }
    with open(DB_CONVERSATIONS_PATH / f"{conversation_id}.json", "w") as f:
        f.write(json.dumps(conversation))
    with open(DB_PATH / "current.json", "w") as f:
        f.write(json.dumps({"current_conversation_id": conversation_id,}))

    load_conversation_to_state(conversation)

def calculate_costs(messages):
    total_input_tokens = 0
    total_output_tokens = 0
    for message in messages:
        if "usage" in message:
            total_input_tokens += message["usage"]["prompt_tokens"]
            total_output_tokens += message["usage"]["completion_tokens"]
    
    input_cost_usd = total_input_tokens * PRICING["input_tokens"]
    output_cost_usd = total_output_tokens * PRICING["output_tokens"]
    total_cost_usd = input_cost_usd + output_cost_usd
    total_cost_pln = total_cost_usd * USD_TO_PLN
    return total_cost_pln

# --- Logowanie ---
def login():
    st.session_state["logged_in"] = False
    users = {}
    config = {}
    
    # DODAJEMY NAGŁÓWEK TUTAJ
    img_path = "logo.png"
    encoded_img = img_to_bytes(img_path)
    if encoded_img:
        header_html = f"""
            <div style="display: flex; align-items: center; justify-content: flex-start;">
                <img src="data:image/png;base64,{encoded_img}" width="100" height="100" style="vertical-align: middle; margin-right: 20px;">
                <h1 style="vertical-align: middle; margin: 0;">PIONIER GPT</h1>
            </div>
        """
        st.markdown(header_html, unsafe_allow_html=True)
    # KONIEC DODANEGO NAGŁÓWKA

    try:
        with open("secret.toml", "rb") as f:
            config = toml.load(f)
            users = config.get("users", {})
            # Próbujemy wczytać klucz API z TOML i zapisać w sesji
            st.session_state["openai_api_key"] = config.get("openai", {}).get("api_key")

    except FileNotFoundError:
        st.error("Brak pliku secret.toml. Proszę go utworzyć i skonfigurować zgodnie z instrukcjami.")
        return

    st.subheader("Logowanie")
    username = st.text_input("Nazwa użytkownika")
    password = st.text_input("Hasło", type="password")

    if st.button("Zaloguj"):
        hashed_password = users.get(username)
        if hashed_password and bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8')):
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.rerun()
        else:
            st.error("Nieprawidłowa nazwa użytkownika lub hasło")

def logout():
    st.session_state["logged_in"] = False
    st.session_state.pop("username", None)
    st.session_state.pop("openai_client", None)
    st.session_state.pop("openai_api_key", None)
    st.rerun()

# --- Główna logika aplikacji Streamlit ---

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    login()
else:
    # --- Nagłówek ---
    img_path = "logo.png"
    encoded_img = img_to_bytes(img_path)
    if encoded_img:
        header_html = f"""
            <img src="data:image/png;base64,{encoded_img}" class="img-fluid" width="100" height="100" style="display: inline-block; vertical-align: middle;">
            <h1 style="display: inline-block; vertical-align: middle; margin-left: 20px;">PIONIER GPT</h1>
        """
        st.markdown(header_html, unsafe_allow_html=True)

    # --- Inicjalizacja Klienta OpenAI ---
    # Klucz jest już w session_state dzięki funkcji login()
    openai_client = init_openai_client()

    if openai_client:
        # --- Sidebar (Pasek boczny) ---
        with st.sidebar:
            # 1. Guik wyloguj na samej górze
            st.button("Wyloguj", on_click=logout, type="primary")

            # 2. Koszty (mniejszy rozmiar)
            st.markdown("---")
            st.markdown(f"<p style='font-size: small;'>Model: <b>{MODEL}</b></p>", unsafe_allow_html=True)
            if "messages" in st.session_state:
                total_cost = calculate_costs(st.session_state["messages"])
                st.markdown(f"<p style='font-size: small;'>Całkowity koszt rozmowy: <b>{total_cost:.4f} PLN</b></p>", unsafe_allow_html=True)
            else:
                st.markdown("<p style='font-size: small;'>Całkowity koszt rozmowy: 0.0000 PLN</p>", unsafe_allow_html=True)
            st.markdown("---")

            # 3. Guik nowej rozmowy
            st.button("Nowa rozmowa", on_click=create_new_conversation)

            # 4. Aktywny guzik załaduj rozmowę (poprzednie rozmowy)
            st.markdown("---")
            st.subheader("Załaduj rozmowę")
            
            conversation_files = sorted(DB_CONVERSATIONS_PATH.glob("*.json"), key=os.path.getmtime, reverse=True)
            for file_path in conversation_files:
                with open(file_path, "r") as f:
                    conv_data = json.loads(f.read())
                    if st.button(conv_data["name"]):
                        load_conversation_to_state(conv_data)
                        st.session_state["new_chatbot_personality"] = conv_data["chatbot_personality"]
                        st.rerun()

            # 5. Okno z osobowością chatbota
            st.markdown("---")
            st.subheader("Osobowość chatbota")
            st.text_area(
                "Prompt systemowy",
                key="new_chatbot_personality",
                value=st.session_state.get("chatbot_personality", DEFAULT_PERSONALITY),
                on_change=save_current_conversation_personality
            )

        # --- Główny Kontener Aplikacji ---
        if "messages" not in st.session_state:
            load_current_conversation()
        
        # Edycja nazwy konwersacji w głównym widoku (np. nad czatem)
        st.text_input(
            "Nazwa konwersacji",
            key="new_conversation_name_input",
            on_change=save_current_conversation_name,
            value=st.session_state.get("name", "Ładowanie nazwy...")
        )

        # Wyświetlanie historii czatu
        for message in st.session_state["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Obsługa inputu użytkownika
        if user_prompt := st.chat_input("Napisz coś do chatbota..."):
            st.chat_message("user").markdown(user_prompt)
            st.session_state["messages"].append({"role": "user", "content": user_prompt})

            with st.spinner("Myślę..."):
                # Przekazujemy klienta OpenAI do funkcji reply, bo jest on zainicjowany w sesji
                response = chatbot_reply(user_prompt, st.session_state["messages"], openai_client)
            
            st.session_state["messages"].append(response)
            save_current_conversation_messages()
            st.rerun()