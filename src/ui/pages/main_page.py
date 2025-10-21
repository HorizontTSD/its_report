import asyncio
import streamlit as st

from ui.utils.greating import greatings

from ui.utils.data_processing import check_authentication
from markdown_content import intnrodaction


hide_sidebar = """
    <style>
    [data-testid="stSidebar"] {display: none;}
    [data-testid="stSidebarNav"] {display: none;}
    [data-testid="collapsedControl"] {display: none;}
    </style>
    """
st.markdown(hide_sidebar, unsafe_allow_html=True)

@check_authentication
async def main():

    st.set_page_config(layout="wide")
    st.markdown(intnrodaction)


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()



