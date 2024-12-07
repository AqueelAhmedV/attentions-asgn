import streamlit as st
import asyncio
from datetime import datetime
from memory import MemoryAgent
# from agents import WeatherAgent
# from config import OPENWEATHER_API_KEY  # Add this to your config.py

# Initialize agents in session state
async def init_agents():
    """Initialize session state variables and agents"""
    if 'memory_agent' not in st.session_state:
        memory_agent = MemoryAgent()
        await memory_agent.initialize()
        st.session_state.memory_agent = memory_agent
        
        # Initialize WeatherAgent
        # weather_agent = WeatherAgent(
        #     memory_agent=memory_agent,
        #     weather_api_key=OPENWEATHER_API_KEY
        # )
        # st.session_state.weather_agent = weather_agent

def init_session_state():
    """Initialize session state variables"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'session_token' not in st.session_state:
        st.session_state.session_token = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'current_chat' not in st.session_state:
        st.session_state.current_chat = []

def handle_login(username: str, password: str):
    """Handle login logic"""
    session_token = st.session_state.memory_agent.authenticate_user(username, password)
    if session_token:
        st.session_state.authenticated = True
        st.session_state.username = username
        st.session_state.session_token = session_token
        return True
    return False

def handle_signup(username: str, password: str):
    """Handle signup logic"""
    success = st.session_state.memory_agent.create_user(username, password)
    if success:
        return handle_login(username, password)
    return False

async def process_message(message: str) -> str:
    """Process user message and return AI response"""
    username = st.session_state.username
    
    # Detect if it's a weather-related query
    weather_keywords = ['weather', 'temperature', 'rain', 'sunny', 'forecast', 
                       'hot', 'cold', 'climate', 'humidity']
    
    is_weather_query = any(keyword in message.lower() for keyword in weather_keywords)
    
    if is_weather_query:
        response = await st.session_state.weather_agent.process_weather_query(
            username=username,
            query=message
        )
    else:
        # Default response for non-weather queries
        preferences = await st.session_state.memory_agent.get_user_preferences(username)
        response = f"I understand your message. Based on your preferences: {preferences}"
    
    return response

def login_page():
    """Display login form"""
    st.title("Tour Planning Assistant")
    
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    
    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
            
            if submit and username and password:
                if handle_login(username, password):
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")

    with tab2:
        with st.form("signup_form"):
            new_username = st.text_input("Choose Username")
            new_password = st.text_input("Choose Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            submit = st.form_submit_button("Sign Up")
            
            if submit:
                if new_password != confirm_password:
                    st.error("Passwords don't match")
                elif not new_username or not new_password:
                    st.error("Please fill all fields")
                else:
                    if handle_signup(new_username, new_password):
                        st.success("Account created successfully!")
                        st.rerun()
                    else:
                        st.error("Error creating account")

def chat_interface():
    """Main chat interface"""
    # Validate session
    if not st.session_state.memory_agent.validate_session(st.session_state.session_token):
        st.session_state.authenticated = False
        st.rerun()
        return

    st.title(f"Welcome {st.session_state.username}!")
    
    # Add logout button in sidebar
    with st.sidebar:
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.session_token = None
            st.rerun()
    
        st.title("Previous Chats")
        if st.button("Start New Chat"):
            st.session_state.current_chat = []
            st.rerun()
        
        # Display previous chat sessions
        for idx, chat in enumerate(st.session_state.chat_history):
            if st.button(f"Chat {idx + 1} - {chat['date']}", key=f"chat_{idx}"):
                st.session_state.current_chat = chat['messages']
                st.rerun()
    
    # Main chat area
    chat_container = st.container()
    
    # Display current chat messages
    with chat_container:
        for message in st.session_state.current_chat:
            with st.chat_message(message["role"]):
                st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("What's your travel plan?"):
        # Add user message to chat
        st.session_state.current_chat.append({"role": "user", "content": prompt})
        
        # Process message and get AI response
        with st.spinner("Thinking..."):
            ai_response = asyncio.run(process_message(prompt))
        
        # Add AI response to chat
        st.session_state.current_chat.append({"role": "assistant", "content": ai_response})
        
        # Save chat to history if it's new
        if len(st.session_state.current_chat) == 2:  # First interaction in this chat
            st.session_state.chat_history.append({
                "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "messages": st.session_state.current_chat.copy()
            })
        st.rerun()

def main():
    # Run async initialization
    if 'memory_agent' not in st.session_state:
        asyncio.run(init_agents())
    
    init_session_state()
    
    if not st.session_state.authenticated:
        login_page()
    else:
        chat_interface()

if __name__ == "__main__":
    main()
