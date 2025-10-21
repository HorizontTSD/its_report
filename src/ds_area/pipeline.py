import os
import pandas as pd
from collections import defaultdict
import requests
from datetime import datetime, timezone
from typing import List, Dict


home_path = os.getcwd()
init_data_path = os.path.join(home_path, "src", "init_data")
files = [f for f in os.listdir(init_data_path) if os.path.isfile(os.path.join(init_data_path, f))]
preprocess_data_path = os.path.join(home_path, "src", "preprocess_data")
