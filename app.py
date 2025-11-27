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

# Używamy tomllib (Python 3.11+)
try:
    import tomllib as toml
except ImportError:
    import toml

# Ładowanie zmiennych środowiskowych (działa lokalnie i w Streamlit Cloud Secrets)
load_dotenv()

# --- Konfiguracja i Ceny ---
model_pricings = {
    "gpt-4o": {"input_tokens": 5.00 / 1_000_000, "output_tokens": 15.00 / 1_000_000},
    "gpt-4o-mini": {"input_tokens": 0.150 / 1_000_000, "output_tokens": 0.600 / 1_000_000}
}
MODEL = "gpt-4o-mini"
USD_TO_PLN = 3.97
PRICING = model_pricings[MODEL]
DEFAULT_PERSONALITY = """
Jesteś pomocnikiem, który odpowiada na wszystkie pytania użytkownika.
Odpowiadaj na pytania w sposób zwięzły i zrozumiały.
""".strip()

# --- Funkcje Pomocnicze ---
def img_to_bytes(img_path):
    if Path(img_path).exists():
        img_bytes = Path(img_path).read_bytes()
        encoded = base64.b64encode(img_bytes).decode()
        return encoded
    return ""

def init_openai_client():
    if "openai_client" not in st.session_state:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                st.session_state.openai_client = OpenAI(api_key=api_key)
            except AuthenticationError:
                st.error("Podany klucz API OpenAI jest nieprawidłowy.")
                st.session_state["logged_in"] = False
            except Exception as e:
                st.error(f"Wystąpił błąd podczas inicjalizacji klienta OpenAI: {e}")
        else:
            st.warning("Klucz OpenAI API nie został znaleziony w zmiennych środowiskowych.")
            return None
    return st.session_state.get("openai_client")

def chatbot_reply(user_prompt, memory, openai_client_instance):
    messages = [{"role": "system", "content": st.session_state["chatbot_personality"]}]
    for message in memory:
        messages.append({"role": message["role"], "content": message["content"]})
    messages.append({"role": "user", "content": user_prompt})

    response = openai_client_instance.chat.completions.create(model=MODEL, messages=messages)
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
# CONVERSATION HISTORY AND DATABASE (ZAPIS PLIKOWY)
#

DB_PATH = Path("db")
DB_CONVERSATIONS_PATH = DB_PATH / "conversations"

def load_conversation_to_state(conversation):
    st.session_state["id"] = conversation["id"]
    st.session_state["name"] = conversation["name"]
    st.session_state["messages"] = conversation["messages"]
    st.session_state["chatbot_personality"] = conversation["chatbot_personality"]
    st.session_state["new_conversation_name_input"] = conversation["name"]

def load_current_conversation():
    # Tworzymy foldery jeśli nie istnieją
    if not DB_PATH.exists():
        DB_PATH.mkdir(exist_ok=True)
        DB_CONVERSATIONS_PATH.mkdir(exist_ok=True)
        # Tworzymy pierwszą konwersację startową
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
    
    # Ładujemy aktualną konwersację
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
    new_conversation_name = st.session_state["new_conversation_name_input"]

    with open(DB_CONVERSATIONS_PATH / f"{conversation_id}.json", "r") as f:
        conversation = json.loads(f.read())

    with open(DB_CONVERSATIONS_PATH / f"{conversation_id}.json", "w") as f:
        f.write(json.dumps({**conversation, "name": new_conversation_name,}))
    
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

# --- Logowanie (Uproszczone) ---

def login_form():
    # Używamy os.getenv do pobierania danych z secrets.toml/env vars
    correct_user = os.getenv("APP_USER", "admin") # Domyślny user jeśli brak zmiennej
    correct_pass_hash = os.getenv("APP_PASS_HASH", "$2b$12$EXAMPLEHASH") # Zmień na swój hash

    st.title("Pionier GPT - Logowanie")
    username = st.text_input("Użytkownik")
    password = st.text_input("Hasło", type="password")

    if st.button("Zaloguj"):
        # Weryfikacja hasła bcrypt
        if username == correct_user and bcrypt.checkpw(password.encode('utf-8'), correct_pass_hash.encode('utf-8')):
            st.session_state["logged_in"] = True
            st.success("Zalogowano pomyślnie!")
            st.rerun() # Przeładowanie strony, aby pokazać UI
        else:
            st.error("Nieprawidłowy login lub hasło.")

# --- Główna logika aplikacji (UI) ---

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if st.session_state["logged_in"]:
    # --------------------------------------------------
    # CAŁY INTERFEJS UŻYTKOWNIKA STREAMLIT
    # --------------------------------------------------

    # Inicjalizacja klienta OpenAI po zalogowaniu
    openai_client = init_openai_client()

    if "messages" not in st.session_state:
        # POBIERAMY DANE Z PLIKÓW PRZY PIERWSZYM URUCHOMIENIU SESJI
        load_current_conversation()

    # SIDEBAR
    with st.sidebar:
        st.header(st.session_state.get("name", "Nowa konwersacja"))
        st.text_input("Zmień nazwę:", key="new_conversation_name_input", on_change=save_current_conversation_name)
        
        st.text_area("Osobowość Chatbota:", key="new_chatbot_personality", value=st.session_state.get("chatbot_personality", DEFAULT_PERSONALITY), on_change=save_current_conversation_personality, height=150)
        
        st.button("Nowa Konwersacja", on_click=create_new_conversation, use_container_width=True)
        st.divider()
        st.button("Wyloguj", on_click=lambda: st.session_state.pop("logged_in", None), use_container_width=True)

        if "messages" in st.session_state:
            total_cost_pln = calculate_costs(st.session_state["messages"])
            st.info(f"Koszt tej konwersacji: {total_cost_pln:.4f} PLN")

    # GŁÓWNY WIDOK CHATBOTA
    st.title(st.session_state.get("name", "Pionier GPT"))

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Napisz coś do chatbota..."):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if openai_client:
            with st.spinner("Myślę..."):
                reply = chatbot_reply(prompt, st.session_state["messages"], openai_client)
            
            st.session_state["messages"].append(reply)
            
            # ZAPISUJEMY WIADOMOŚCI DO PLIKU PO KAŻDEJ ODPOWIEDZI
            save_current_conversation_messages() 
            
            with st.chat_message("assistant"):
                st.markdown(reply["content"])

else:
    # --------------------------------------------------
    # WYŚWIETLANIE STRONY LOGOWANIA
    # --------------------------------------------------
    login_form()