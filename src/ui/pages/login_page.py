import asyncio

import streamlit as st
from ui.utils.auth_streamlit import authorization
from utils import logger
import hydralit_components as hc


async def login():

    with hc.HyLoader('', hc.Loaders.standard_loaders, index=3):
        columns = st.columns(spec=[1,8,1])
        with columns[1]:
            if "status_code" not in st.session_state:
                st.session_state['status_code'] = None
            if "current_page" not in st.session_state:
                st.session_state['current_page'] = 1

            authenticator = authorization()

            name, authentication_status, user = authenticator.login(
                fields={
                    "Form name": "Авторизация",
                    "Username": "Username",
                    "Password": "Password",
                    "Login": "Войти",
                },
            )


            if authentication_status:
                st.session_state.authentication_status = True
                st.markdown(
                    """
                    <style>
                    #root > div:nth-child(1) > div.withScreencast > div > div > div > section.main.st-emotion-cache-bm2z3a.ea3mdgi8 > div.block-container.st-emotion-cache-1jicfl2.ea3mdgi5 > div > div > div > div.st-emotion-cache-ixecyn.e10yg2by1 > div > div > div > div:nth-child(2) {
                        display: none;
                    }
                    </style>
                    <script>
                    document.addEventListener("DOMContentLoaded", function() {
                        var element = document.querySelector("#root > div:nth-child(1) > div.withScreencast > div > div > div > section.main.st-emotion-cache-bm2z3a.ea3mdgi8 > div.block-container.st-emotion-cache-1jicfl2.ea3mdgi5 > div > div > div > div.st-emotion-cache-ixecyn.e10yg2by1 > div > div > div > div:nth-child(2)");
                        if (element) {
                            element.remove();
                        }
                    });
                    </script>
                    """,
                    unsafe_allow_html=True
                )
                st.rerun()
            elif authentication_status is False:
                st.session_state.authentication_status = False
            elif authentication_status is None:
                st.session_state.authentication_status = None
            else:
                # TODO: Find proper actions
                logger.debug(authentication_status)


if __name__ == "__main__":
    asyncio.run(login())
