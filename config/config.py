import json

def load_config() -> dict:
    with open('../resources/config.json') as f:
        data = json.load(f)
    return data
