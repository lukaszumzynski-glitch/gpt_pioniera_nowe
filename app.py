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
import psycopg2 # Importujemy bibliotekę do PostgreSQL

# Ładowanie zmiennych środowiskowych z pliku .env (działa lokalnie, w Streamlit Cloud używa Secrets)
load_dotenv()

# Używamy tomllib (Python 3.11+) lub zainstaluj pip install toml
try:
    import tomllib as toml
except ImportError:
    import toml

# --- Konfiguracja i Ceny ---
model_pricings = {
    "gpt-4o": {
        "input_tokens": 5.00 / 1_000_000,
        "output_tokens": 15.00 / 1_000_000,
    },
    "gpt-4o-mini": {
        "input_tokens": 0.150 / 1_000_000,
        "output_tokens": 0.600 / 1_000_1_000,
    }
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
    """Konwertuje obraz na ciąg bajtów zakodowany w Base64."""
    if Path(img_path).exists():
        img_bytes = Path(img_path).read_bytes()
        encoded = base64.b64encode(img_bytes).decode()
        return encoded
    return ""

def init_openai_client():
    if "openai_client" not in st.session_state:
        api_key = os.getenv("OPENAI_API_KEY") # Pobieramy z env vars
        if api_key:
            try:
                st.session_state.openai_client = OpenAI(api_key=api_key)
            except AuthenticationError:
                st.error("Podany klucz API OpenAI jest nieprawidłowy.")
                st.session_state["logged_in"] = False
                st.session_state.pop("openai_api_key", None)
            except Exception as e:
                st.error(f"Wystąpił błąd podczas inicjalizacji klienta OpenAI: {e}")
        else:
            st.warning("Klucz OpenAI API nie został znaleziony w zmiennych środowiskowych.")
            return None
    return st.session_state.get("openai_client")

def chatbot_reply(user_prompt, memory, openai_client_instance):
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
# CONVERSATION HISTORY AND DATABASE (Logika przeniesiona do DB)
#
# Zmienne DB_PATH i DB_CONVERSATIONS_PATH zostały usunięte.

def get_db_connection():
    """Tworzy połączenie z bazą danych PostgreSQL na DigitalOcean."""
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD")
        )
        return conn
    except psycopg2.OperationalError as e:
        st.error(f"Błąd połączenia z bazą danych: {e}")
        return None

def load_conversation_to_state(conversation_data):
    """Ładuje dane konwersacji z obiektu (np. z wyniku zapytania SQL) do session_state."""
    st.session_state["id"] = conversation_data["id"]
    st.session_state["name"] = conversation_data["name"]
    # Zakładamy, że messages są przechowywane w DB jako JSON string lub array/json type
    st.session_state["messages"] = json.loads(conversation_data["messages"]) if isinstance(conversation_data["messages"], str) else conversation_data["messages"]
    st.session_state["chatbot_personality"] = conversation_data["chatbot_personality"]
    st.session_state["new_conversation_name_input"] = conversation_data["name"]

# --- FUNKCJE BAZODANOWE (SZABLONY - WYMAGAJĄ IMPLEMENTACJI SQL) ---

def load_current_conversation():
    """Pobiera aktualną konwersację z bazy danych."""
    conn = get_db_connection()
    if conn:
        with conn.cursor() as cur:
            # Pamiętaj: musisz stworzyć tabelę `conversations` w swojej bazie danych!
            # Przykład zapytania (wymaga dostosowania):
            # cur.execute("SELECT * FROM conversations WHERE is_current = TRUE LIMIT 1")
            # conversation = cur.fetchone() 
            # if conversation:
            #     # Konwertuj wynik krotki (tuple) na słownik (dict) jeśli to konieczne 
            #     load_conversation_to_state(conversation_dict)
            # else:
            #     create_new_conversation() # Jeśli brak, tworzymy nową
            pass # Zaimplementuj logikę SQL tutaj
        conn.close()

def save_current_conversation_messages():
    """Zapisuje wiadomości do bazy danych."""
    conn = get_db_connection()
    if conn:
        with conn.cursor() as cur:
            messages_json = json.dumps(st.session_state["messages"])
            conv_id = st.session_state["id"]
            # cur.execute("UPDATE conversations SET messages = %s WHERE id = %s", (messages_json, conv_id))
            conn.commit()
        conn.close()

def save_current_conversation_name():
    """Zapisuje nazwę konwersacji."""
    conn = get_db_connection()
    if conn:
        with conn.cursor() as cur:
            new_name = st.session_state["new_conversation_name_input"]
            conv_id = st.session_state["id"]
            # cur.execute("UPDATE conversations SET name = %s WHERE id = %s", (new_name, conv_id))
            conn.commit()
        conn.close()
    st.session_state["name"] = st.session_state["new_conversation_name_input"]


def save_current_conversation_personality():
    """Zapisuje osobowość chatbota."""
    conn = get_db_connection()
    if conn:
        with conn.cursor() as cur:
            new_personality = st.session_state["new_chatbot_personality"]
            conv_id = st.session_state["id"]
            # cur.execute("UPDATE conversations SET chatbot_personality = %s WHERE id = %s", (new_personality, conv_id))
            conn.commit()
        conn.close()


def create_new_conversation():
    """Tworzy nową konwersację w bazie danych i ustawia ją jako aktywną."""
    conn = get_db_connection()
    if conn:
        with conn.cursor() as cur:
            personality = st.session_state.get("chatbot_personality", DEFAULT_PERSONALITY)
            # cur.execute("INSERT INTO conversations (name, chatbot_personality, messages, is_current) VALUES (%s, %s, %s, TRUE) RETURNING id", 
            #             (f"Konwersacja {datetime.now().strftime('%Y%m%d%H%M')}", personality, json.dumps([])))
            # new_id = cur.fetchone()[0]
            # cur.execute("UPDATE conversations SET is_current = FALSE WHERE id != %s", (new_id,)) # Ustaw pozostałe na nieaktywne
            conn.commit()
            # conversation_data = {"id": new_id, "name": f"Konwersacja...", "messages": [], "chatbot_personality": personality}
            # load_conversation_to_state(conversation_data)
        conn.close()

# --- Pozostałe funkcje (Logowanie, Renderowanie UI) pozostają niezmienione ---

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

def login():
    st.session_state["logged_in"] = False
    users = {}
    config = {}
    
    # ... (kod logowania, który używa bcrypt i secret.toml/env vars) ...
    # Zakładamy, że ten kod działa i ustawia st.session_state["logged_in"] = True
    pass # Usuń 'pass' jeśli masz tu działającą logikę logowania

# --- Główna logika aplikacji (UI) ---

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if st.session_state["logged_in"]:
    # Kod interfejsu Streamlit (sidebar, chat_input itp.)
    # Ten kod wywoła funkcje load/save DB
    pass # Usuń 'pass' i wklej resztę UI swojej aplikacji tutaj, która wywoła funkcje bazodanowe
else:
    login()