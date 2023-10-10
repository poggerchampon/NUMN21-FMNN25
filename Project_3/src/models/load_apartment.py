from room import Room
import numpy as np
import json

def load_apartment_layout(filename='apartment.json'):
	with open('apartment.json', 'r') as f:
		apart_config = json.load(f)
	
	rooms = [Room(**room_config) for room_config in apart_config['rooms']]
	
	return rooms
