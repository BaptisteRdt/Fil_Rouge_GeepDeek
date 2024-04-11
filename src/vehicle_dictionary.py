from math import hypot
import cv2

types = {0: "car", 1: "motorcycle", 2: "truck"}


def get_vehicles_dict(frame, old_vehicles_dict: dict = None) -> dict:
    vehicles = {}
    for bbox in frame[0].boxes:
        xl, yt, xr, yb = bbox.xyxy[0]
        vehicles[int(bbox.id)] = {'type': types[int(bbox.cls)], 'xl': int(xl), 'yt': int(yt),
                                  'xr': int(xr), 'yb': int(yb), 'confidence': float(bbox.conf),
                                  'center': {'x': int(xl + xr) / 2, 'y': int(yt + yb) / 2},
                                  'is_track': bbox.is_track, 'id': int(bbox.id), 'is_moving': False,
                                  'direction': {'Top': 0, 'Bottom': 0, 'Left': 0, 'Right': 0}}
        if old_vehicles_dict is not None and int(bbox.id) in old_vehicles_dict:
            direction = direction_between_vehicles(old_vehicles_dict[int(bbox.id)], vehicles[int(bbox.id)])
            vehicles[int(bbox.id)]['direction'] = direction
            vehicles[int(bbox.id)]['length'] = get_car_length(vehicles[int(bbox.id)])
            vehicles[int(bbox.id)]['is_moving'] = (direction != {'Top': 0, 'Bottom': 0, 'Left': 0, 'Right': 0})

    vehicles = get_closest_vehicles(vehicles)
    return vehicles


def condition_closest_vehicles(vehicle, other_vehicle):
    return (vehicle['id'] != other_vehicle['id'] and other_vehicle['is_moving'] and
            other_vehicle['is_track'] and vehicle['direction'] == other_vehicle['direction'])


def get_closest_vehicles(vehicles: dict) -> dict:
    for vehicle in vehicles.values():
        vehicle['id_closest_vehicle'] = None
        if vehicle['is_track'] and vehicle['is_moving']:
            min_distance = 100000
            for other_vehicle in vehicles.values():
                if condition_closest_vehicles(vehicle, other_vehicle):
                    distance = hypot(abs(vehicle['center']['x'] - other_vehicle['center']['x']),
                                     abs(vehicle['center']['y'] - other_vehicle['center']['y']))
                    if distance < min_distance:
                        min_distance = distance
                        vehicle['id_closest_vehicle'] = other_vehicle['id']
    return vehicles


def get_car_length(vehicle: dict) -> int:
    if vehicle['direction']['Top'] == 1 or vehicle['direction']['Bottom'] == 1:
        length = vehicle['yt'] - vehicle['yb']
    else:
        length = vehicle['xr'] - vehicle['xl']
    return length


def direction_between_vehicles(vehicle_1, vehicle_2) -> dict:
    diff_x = vehicle_1['center']['x'] - vehicle_2['center']['x']
    diff_y = vehicle_1['center']['y'] - vehicle_2['center']['y']
    direction = {'Top': 0, 'Bottom': 0, 'Left': 0, 'Right': 0}

    if abs(diff_x) < 10 and abs(diff_y) < 10:
        return direction
    
    if diff_y > diff_x:
        if diff_y < 0.0:
            direction['Top'] = 1
        else:
            direction['Bottom'] = 1
    else:
        if diff_x < 0.0:
            direction['Left'] = 1
        else:
            direction['Right'] = 1
    return direction
