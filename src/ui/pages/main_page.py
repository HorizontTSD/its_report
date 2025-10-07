import asyncio
import streamlit as st

from ui.utils.greating import greatings

from ui.utils.data_processing import check_authentication


@check_authentication
async def main():
    st.sidebar.write('# Здесь будут всякие параметры')

    if st.button('Скажи привет!'):
        greatings()
        st.sidebar.write('## Привет Саша')


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()



