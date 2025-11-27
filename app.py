import json
from pathlib import Path
import streamlit as st
from openai import OpenAI, AuthenticationError
from dotenv import load_dotenv
import base64
import os
import bcrypt
import tomllib
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
    
    assistant_message_content = response.choices.message.content

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
    # *** POPRAWKA 1: Ustawienie inputu na załadowaną nazwę, aby odzwierciedlał stan bieżący ***
    st.session_state["new_conversation_name_input"] = conversation["name"] 

def load_current_conversation():
    if not DB_PATH.exists():
        DB_PATH.mkdir(exist_ok=True)
    if not DB_CONVERSATIONS_PATH.exists():
        DB_CONVERSATIONS_PATH.mkdir(exist_ok=True)

    current_user = st.session_state.get("username")
    if not current_user: return 

    available_convs = list_conversations(current_user)
    
    if not available_convs:
        create_new_conversation()
    else:
        # Ten kod (load_current_conversation) jest wywoływany przy starcie sesji,
        # więc powinien załadować aktualną konwersację z pliku 'current.json' (globalnego)
        # i sprawdzić czy należy do current_user. Jeśli nie, załadować najnowszą usera.
        
        # Uproszczenie: zawsze ładuj najnowszą konwersację usera przy starcie sesji
        latest_conv = available_convs # Poprawione indeksowanie listy
        with open(DB_CONVERSATIONS_PATH / f"{latest_conv['id']}.json", "r") as f:
             conversation = json.loads(f.read())
        load_conversation_to_state(conversation)

def save_current_conversation_messages():
    try:
        conversation_id = st.session_state["id"]
        new_messages = st.session_state["messages"]
        file_path = DB_CONVERSATIONS_PATH / f"{conversation_id}.json"
        if file_path.exists():
            with open(file_path, "r") as f:
                conversation = json.loads(f.read())
            with open(file_path, "w") as f:
                f.write(json.dumps({**conversation, "messages": new_messages,}))
    except Exception as e:
        # st.error(f"Błąd podczas zapisu wiadomości do pliku: {e}")
        print(f"Błąd podczas zapisu wiadomości do pliku: {e}")

def save_current_conversation_name():
    # *** POPRAWKA 1: Poprawiona logika zapisu nazwy z inputu st.text_input po naciśnięciu ENTER ***
    try:
        conversation_id = st.session_state["id"]
        # Odczytujemy wartość bezpośrednio z klucza session_state przypisanego do inputu
        new_conversation_name = st.session_state.get("new_conversation_name_input", "").strip()

        if not new_conversation_name:
             new_conversation_name = f"Konwersacja {conversation_id}"
        
        file_path = DB_CONVERSATIONS_PATH / f"{conversation_id}.json"
        if file_path.exists():
            with open(file_path, "r") as f:
                conversation = json.loads(f.read())
            with open(file_path, "w") as f:
                f.write(json.dumps({**conversation, "name": new_conversation_name,}))
            st.session_state["name"] = new_conversation_name
            # Wymuszamy odświeżenie listy konwersacji w sidebarze
            st.session_state['reload_app_state'] = True 
    except Exception as e:
        print(f"Błąd podczas zapisu nazwy do pliku: {e}")

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
        print(f"Błąd podczas zapisu osobowości do pliku: {e}")

def create_new_conversation():
    conversation_ids = []
    for p in DB_CONVERSATIONS_PATH.glob("*.json"):
        try:
            conversation_ids.append(int(p.stem))
        except ValueError:
            continue
    next_id = max(conversation_ids) + 1 if conversation_ids else 1
    
    personality = DEFAULT_PERSONALITY
    if "chatbot_personality" in st.session_state and st.session_state["chatbot_personality"]:
        personality = st.session_state["chatbot_personality"]

    # Nazwa początkowa
    initial_name = f"Konwersacja {next_id}"

    conversation = {
        "id": next_id,
        "name": initial_name,
        "chatbot_personality": personality,
        "messages": [],
        "username": st.session_state.get("username")
    }
    with open(DB_CONVERSATIONS_PATH / f"{next_id}.json", "w") as f:
        f.write(json.dumps(conversation))
    
    # *** POPRAWKA 1: Ładowanie nowej konwersacji do state od razu po jej utworzeniu ***
    load_conversation_to_state(conversation)

    st.session_state['reload_app_state'] = True


def list_conversations(username=None):
    """Zwraca listę słowników z ID i nazwami konwersacji, filtruje po username i sortuje malejąco po ID."""
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
                # print(f"Błąd podczas ładowania pliku konwersacji {p}: {e}")
                continue
    
    # Sortowanie listy konwersacji od najnowszej (najwyższe ID) do najstarszej
    conversations_list.sort(key=lambda x: x['id'], reverse=True)
    return conversations_list

# *** POPRAWKA 2: Nowa funkcja do aktywnego przełączania konwersacji (callback dla st.button) ***
def switch_conversation(conversation_id):
    """Ładuje wybraną konwersację z pliku do st.session_state."""
    file_path = DB_CONVERSATIONS_PATH / f"{conversation_id}.json"
    if file_path.exists():
        with open(file_path, "r") as f:
            conversation_data = json.loads(f.read())
        
        if conversation_data.get("username") == st.session_state.get("username"):
            load_conversation_to_state(conversation_data)
            st.session_state['reload_app_state'] = True # Wymuś odświeżenie UI

# --- Authentication System (Twoj oryginalny kod) ---

DB_USERS_PATH = DB_PATH / "users.toml"

def render_login_page():
    st.set_page_config(page_title="Chatbot Logowanie", page_icon=":speech_balloon:")
    
    # Użyj CSS, aby wyśrodkować formularz logowania
    st.markdown("""
    <style>
    .stApp {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh;
    }
    .stForm {
        width: 100%;
        max-width: 400px;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        background-color: #f9f9f9;
    }
    </style>
    """, unsafe_allow_html=True)

    with st.form("login_form"):
        st.markdown("<h1 style='text-align: center;'>Logowanie</h1>", unsafe_allow_html=True)
        username = st.text_input("Nazwa użytkownika")
        password = st.text_input("Hasło", type="password")
        col1, col2 = st.columns(2)
        with col1:
            submitted_login = st.form_submit_button("Zaloguj")
        with col2:
            submitted_register = st.form_submit_button("Zarejestruj")

    if submitted_login:
        on_login(username, password)
    if submitted_register:
        on_register(username, password)

def on_register(username, password):
    if not username or not password:
        st.error("Nazwa użytkownika i hasło nie mogą być puste.")
        return

    users_data = {}
    if DB_USERS_PATH.exists():
        with open(DB_USERS_PATH, mode="rb") as f:
            users_data = tomllib.load(f)

    if username in users_data:
        st.error("Użytkownik o tej nazwie już istnieje.")
        return

    # Hashowanie hasła
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    users_data[username] = {"password": hashed_password}

    with open(DB_USERS_PATH, mode="w") as f:
        import tomli_w
        tomli_w.dump(users_data, f)
    
    st.success("Rejestracja zakończona sukcesem. Możesz się teraz zalogować.")

def on_login(username, password):
    if not username or not password:
        st.error("Nazwa użytkownika i hasło nie mogą być puste.")
        return

    if DB_USERS_PATH.exists():
        with open(DB_USERS_PATH, mode="rb") as f:
            users_data = tomllib.load(f)
        
        if username in users_data:
            hashed_password = users_data[username]["password"].encode('utf-8')
            if bcrypt.checkpw(password.encode('utf-8'), hashed_password):
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                st.session_state["reload_app_state"] = True
                return

    st.error("Nieprawidłowa nazwa użytkownika lub hasło.")

def logout():
    st.session_state["logged_in"] = False
    st.session_state["username"] = None
    st.session_state["reload_app_state"] = True
    st.session_state["messages"] = []
    st.session_state["id"] = None
    st.session_state["name"] = None

# --- Main App (Twoj oryginalny kod) ---

def main():
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
        st.session_state["reload_app_state"] = False
        init_openai_client()

    if st.session_state["reload_app_state"]:
        st.session_state["reload_app_state"] = False
        st.rerun()

    if not st.session_state["logged_in"]:
        render_login_page()
    else:
        st.set_page_config(page_title="Chatbot", page_icon=":speech_balloon:", layout="wide")

        if "messages" not in st.session_state or not st.session_state["id"]:
            load_current_conversation()

        # SIDEBAR (Twoj oryginalny uklad)
        with st.sidebar:
            st.header(f"Witaj, {st.session_state['username']}")
            st.button("Wyloguj", on_click=logout)
            st.markdown("---")
            
            # --- Zarządzanie konwersacjami ---
            st.header("Konwersacje")

            # Input do tworzenia nowej konwersacji/zmiany nazwy
            st.text_input(
                "Nazwa konwersacji:",
                key="new_conversation_name_input",
                on_change=save_current_conversation_name, # Callback zapisuje po Enter
                label_visibility="collapsed"
            )

            st.button(
                "➕ Nowa konwersacja", 
                on_click=create_new_conversation, 
                use_container_width=True
            )
            st.markdown("---")

            # Lista zapisanych konwersacji (minimalna zmiana na przycisk z callbackiem)
            available_conversations = list_conversations(st.session_state["username"])
            for conv in available_conversations:
                is_active = st.session_state.get("id") == conv['id']
                if is_active:
                    st.markdown(f"**> {conv['name']}**")
                else:
                    # *** POPRAWKA 2: Używamy przycisku z funkcją przełączania ***
                    st.button(
                        conv['name'], 
                        key=f"switch_{conv['id']}", 
                        on_click=switch_conversation, # Używa nowej funkcji switch
                        args=(conv['id'],),
                        use_container_width=True
                    )
            
            st.markdown("---")
            # --- Ustawienia ---
            st.subheader("Ustawienia")

            st.text_area(
                "Osobowość Chatbota (System Prompt)",
                value=st.session_state.get("chatbot_personality", DEFAULT_PERSONALITY),
                key="new_chatbot_personality",
                on_change=save_current_conversation_personality,
                height=150
            )

        # MAIN CONTENT AREA
        st.title(f"{st.session_state.get('name', 'Chatbot')}")

        for message in st.session_state["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if user_prompt := st.chat_input("Napisz coś..."):
            
            st.session_state["messages"].append({"role": "user", "content": user_prompt})
            with st.chat_message("user"):
                st.markdown(user_prompt)

            client = st.session_state.openai_client
            with st.spinner("Myślę..."):
                reply = chatbot_reply(user_prompt, st.session_state["messages"], client)

            st.session_state["messages"].append(reply)
            with st.chat_message("assistant"):
                st.markdown(reply["content"])
            
            save_current_conversation_messages()


if __name__ == "__main__":
    main()