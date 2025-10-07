import os
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth


def authorization():
    path_to_yaml = os.path.join("src/auth_config.yaml")

    with open(path_to_yaml) as file:
        config = yaml.load(file, Loader=SafeLoader)

    authenticator = stauth.Authenticate(
        credentials=config["credentials"],
        cookie_name=config["cookie"]["name"],
        cookie_key=config["cookie"]["key"],
        # cookie_name='',
        # cookie_key='',
        cookie_expiry_days=config["cookie"]["expiry_days"],
        pre_authorized=config["preauthorized"],
    )

    return authenticator
