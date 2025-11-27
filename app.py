import json
from pathlib import Path
import streamlit as st
from openai import OpenAI, AuthenticationError
from dotenv import load_dotenv
import base64
import os
import bcrypt
from datetime import datetime

# Ładowanie zmiennych środowiskowych
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
            st.error("Klucz OpenAI API nie został znaleziony w zmiennych środowiskowych. Sprawdź konfigurację Secrets.")
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

    assistant_message_content = response.choices[0].message.content

    return {
        "role": "assistant",
        "content": assistant_message_content,
        "usage": usage,
    }

#
# CONVERSATION HISTORY AND DATABASE (ZAPIS PLIKOWY Z OBSŁUGĄ USERÓW)
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
    if not DB_PATH.exists():
        DB_PATH.mkdir(exist_ok=True)
    if not DB_CONVERSATIONS_PATH.exists():
        DB_CONVERSATIONS_PATH.mkdir(exist_ok=True)

    current_user = st.session_state.get("username")
    if not current_user:
        return 

    available_convs = list_conversations(current_user)

    if not available_convs:
        create_new_conversation()
    else:
        # Uproszczenie: zawsze ładuj najnowszą konwersację usera przy starcie sesji
        latest_conv = available_convs[0] 
        with open(DB_CONVERSATIONS_PATH / f"{latest_conv['id']}.json", "r") as f:
             conversation = json.loads(f.read())

        # Upewniamy się, że konwersacja należy do aktualnego użytkownika
        if conversation.get('username') == current_user:
            load_conversation_to_state(conversation)
        else:
            create_new_conversation()  # Utwórz nową konwersację, jeśli coś poszło nie tak

def save_current_conversation_messages():
    try:
        conversation_id = st.session_state["id"]
        new_messages = st.session_state["messages"]
        file_path = DB_CONVERSATIONS_PATH / f"{conversation_id}.json"
        if file_path.exists():
            with open(file_path, "r") as f:
                conversation = json.loads(f.read())
            # Zaktualizowane: zachowujemy username podczas zapisu
            conversation["messages"] = new_messages
            with open(file_path, "w") as f:
                f.write(json.dumps(conversation))
    except Exception as e:
        st.error(f"Błąd podczas zapisu wiadomości do pliku: {e}")

def save_current_conversation_name():
    try:
        conversation_id = st.session_state["id"]
        new_conversation_name = st.session_state["new_conversation_name_input"] 

        file_path = DB_CONVERSATIONS_PATH / f"{conversation_id}.json"
        if file_path.exists():
            with open(file_path, "r") as f:
                conversation = json.loads(f.read())
            with open(file_path, "w") as f:
                f.write(json.dumps({**conversation, "name": new_conversation_name,}))
            st.session_state["name"] = new_conversation_name
    except Exception as e:
        st.error(f"Błąd podczas zapisu nazwy do pliku: {e}")

def save_current_conversation_personality():
    try:
        conversation_id = st.session_state["id"]
        new_chatbot_personality = st.session_state["new_chatbot_personality"]
        file_path = DB_CONVERSATIONS_PATH / f"{conversation_id}.json"
        if file_path.exists():
            with open(file_path, "r") as f:
                conversation = json.loads(f.read())
            with open(file_path, "w") as f:
                f.write(json.dumps({**conversation, "chatbot_personality": new_chatbot_personality,}))
    except Exception as e:
        st.error(f"Błąd podczas zapisu osobowości do pliku: {e}")

def create_new_conversation():
    new_conversation_name = st.session_state.get("new_conversation_name_input") or f"Konwersacja {len(list(DB_CONVERSATIONS_PATH.glob('*.json'))) + 1}"

    conversation_ids = []
    for p in DB_CONVERSATIONS_PATH.glob("*.json"):
        conversation_ids.append(int(p.stem))
    next_id = max(conversation_ids) + 1 if conversation_ids else 1

    personality = DEFAULT_PERSONALITY
    if "chatbot_personality" in st.session_state and st.session_state["chatbot_personality"]:
        personality = st.session_state["chatbot_personality"]

    conversation = {
        "id": next_id,
        "name": new_conversation_name,
        "chatbot_personality": personality,
        "messages": [],
        "username": st.session_state.get("username")  # Zapewnienie, że konwersacja jest przypisana do użytkownika
    }
    with open(DB_CONVERSATIONS_PATH / f"{next_id}.json", "w") as f:
        f.write(json.dumps(conversation))

    st.session_state["new_conversation_name_input"] = new_conversation_name
    st.session_state['reload_app_state'] = True

def list_conversations(username=None):
    user_to_filter = username if username else st.session_state.get("username")
    conversations_list = []
    if DB_CONVERSATIONS_PATH.exists() and user_to_filter:
        for p in DB_CONVERSATIONS_PATH.glob("*.json"):
            try:
                with open(p, "r") as f:
                    data = json.loads(f.read())
                    if data.get("username") == user_to_filter: 
                        conversations_list.append({
                            "id": data.get("id"),
                            "name": data.get("name", f"Konwersacja {data.get('id')}")
                        })
            except Exception as e:
                print(f"Błąd podczas listowania konwersacji: {e}")
    return sorted(conversations_list, key=lambda x: x['id'], reverse=True)

def select_conversation_callback(conversation_id):
    save_current_conversation_messages() 
    st.session_state['pending_conversation_id'] = conversation_id
    st.session_state['reload_app_state'] = True 

    # Załaduj wybraną konwersację
    with open(DB_CONVERSATIONS_PATH / f"{conversation_id}.json", "r") as f:
        conversation = json.loads(f.read())
    if conversation.get('username') == st.session_state.get("username"):  # Sprawdzamy, czy konwersacja należy do aktualnego użytkownika
        load_conversation_to_state(conversation)
    else:
        st.error("Nie możesz otworzyć tej konwersacji.")

def delete_conversation_callback(conversation_id):
    file_path = DB_CONVERSATIONS_PATH / f"{conversation_id}.json"
    if file_path.exists():
        os.remove(file_path)
        st.success(f"Konwersacja usunięta.")
        st.session_state['reload_app_state'] = True
    else:
        st.error("Nie można znaleźć konwersacji do usunięcia.")

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
def login_form():
    img_path = "logo.png"
    encoded_img = img_to_bytes(img_path)
    if encoded_img:
        header_html = f"""
            <div style="display: flex; align-items: center; justify-content: flex-start;">
                <img src="data:image/png;base64,{encoded_img}" width="100" height="100" style="vertical-align: middle; margin-right: 20px;">
                <h1 style="display: inline; vertical-align: middle;">Pionier GPT</h1>
            </div>
            <p style="margin-top: 10px;">Logowanie</p>
        """
        st.markdown(header_html, unsafe_allow_html=True)
    else:
        st.title("Pionier GPT - Logowanie")

    username_input = st.text_input("Użytkownik")
    password_input = st.text_input("Hasło", type="password")

    if st.button("Zaloguj"):
        users_db = {
            os.getenv("user_kasia"): os.getenv("hash_kasia"),
            os.getenv("user_ewunia"): os.getenv("hash_ewunia"),
            os.getenv("user_zbyszek"): os.getenv("hash_zbyszek"),
            os.getenv("user_Pionier"): os.getenv("hash_Pionier"),
            os.getenv("user_mentor"): os.getenv("hash_mentor"),
        }
        users_db = {k: v for k, v in users_db.items() if k is not None and v is not None}

        if username_input in users_db:
            correct_hash = users_db[username_input]
            if correct_hash and bcrypt.checkpw(password_input.encode('utf-8'), correct_hash.encode('utf-8')):
                st.session_state["logged_in"] = True
                st.session_state["username"] = username_input
                st.success(f"Zalogowano pomyślnie jako {username_input}!")
                st.rerun()
            else:
                st.error("Nieprawidłowy login lub hasło.")
        else:
            st.error("Nieprawidłowy login lub hasło.")

# --- Główna logika aplikacji (UI) ---

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if 'reload_app_state' not in st.session_state:
    st.session_state['reload_app_state'] = False

if st.session_state["logged_in"]:
    if st.session_state.get('reload_app_state'):
        if st.session_state.get('pending_conversation_id'):
             with open(DB_PATH / "current.json", "w") as f:
                f.write(json.dumps({"current_conversation_id": st.session_state['pending_conversation_id'],}))
             st.session_state.pop('pending_conversation_id', None) 

        st.session_state['reload_app_state'] = False
        st.rerun() 

    openai_client = init_openai_client()

    if openai_client is None:
        st.stop()

    if "messages" not in st.session_state:
        load_current_conversation()

    # --- POPRAWIONY UKŁAD SIDEBARA ---
    with st.sidebar:
        st.button("Wyloguj", on_click=lambda: st.session_state.pop("logged_in", None) or st.session_state.pop("username", None), use_container_width=True)
        st.divider()

        img_path = "logo.png"
        encoded_img = img_to_bytes(img_path)
        if encoded_img:
             st.markdown(f'<div style="display: flex; align-items: center;"><img src="data:image/png;base64,{encoded_img}" width="50" style="margin-right: 10px;"><h3>Pionier GPT</h3></div>', unsafe_allow_html=True)
        st.markdown(f"Zalogowany jako: **{st.session_state.get('username', 'Użytkownik')}**")
        st.divider()

        if "messages" in st.session_state:
            total_cost_pln = calculate_costs(st.session_state["messages"])
            st.info(f"Koszt tej konwersacji: {total_cost_pln:.4f} PLN")

        st.button("Nowa Konwersacja", on_click=create_new_conversation, use_container_width=True)

        st.text_input("Zmień nazwę bieżącej:", key="new_conversation_name_input", on_change=save_current_conversation_name, value=st.session_state.get("new_conversation_name_input"))
        st.divider()

        st.subheader("Historia konwersacji")
        conversations = list_conversations(st.session_state.get("username"))
        for conv in conversations:
            is_active = conv['id'] == st.session_state.get('id')

            col1, col2 = st.columns([0.8, 0.2])
            with col1:
                st.button(conv['name'] if conv['name'] else f"Konwersacja {conv['id']}", key=f"load_conv_{conv['id']}", use_container_width=True, disabled=is_active, on_click=select_conversation_callback, args=(conv['id'],))
            with col2:
                st.button("🗑️", key=f"delete_conv_{conv['id']}", help="Usuń konwersację", on_click=delete_conversation_callback, args=(conv['id'],))
        st.divider()

        st.subheader("Ustawienia Chatbota")
        st.text_area("Osobowość Chatbota:", key="new_chatbot_personality", value=st.session_state.get("chatbot_personality", DEFAULT_PERSONALITY), on_change=save_current_conversation_personality, height=150)

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
            save_current_conversation_messages() 

            with st.chat_message("assistant"):
                st.markdown(reply["content"])

else:
    login_form()