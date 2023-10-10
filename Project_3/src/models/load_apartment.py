import numpy as np
import json

from .room import Room

def load_apartment_layout(filename='models/apartment.json'):
	with open(filename, 'r') as f:
		apart_config = json.load(f)
	
	rooms = [Room(**room_config) for room_config in apart_config['rooms']]
	
	return rooms
