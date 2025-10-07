import streamlit_authenticator as stauth

password = stauth.Hasher(['admin']).generate()
print(password)
