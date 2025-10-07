import streamlit as st

from ui.pages.login_page import login


def check_authentication(func):
    async def wrapper(*args, **kwargs):
        if st.session_state.get("authentication_status"):

            return await func(*args, **kwargs)
        else:
            await login()

    return wrapper


