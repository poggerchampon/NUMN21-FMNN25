import numpy as np
import json

from .room import Room

def load_apartment_layout(filename='models/apartment.json'):
    """
    Loads apartment layout from a JSON file and initializes Room objects.

    Parameters:
    - filename: The path to the JSON file containing the apartment configuration. Defaults to 'models/apartment.json'.

    Returns:
    - list[Room]: A list of Room objects initialized from the JSON file.

    """
    try:
        with open(filename, 'r') as f:
            apart_config = json.load(f)

        # Ensure that the 'rooms' key is present in the configuration
        if 'rooms' not in apart_config:
            raise KeyError(f"Expected key 'rooms' not found in the JSON file {filename}")

        rooms = [Room(**room_config) for room_config in apart_config['rooms']]

        return rooms

    except FileNotFoundError:
        raise FileNotFoundError(f"The file {filename} was not found.")
    except json.JSONDecodeError:
        raise json.JSONDecodeError(f"Invalid JSON format in {filename}.")
