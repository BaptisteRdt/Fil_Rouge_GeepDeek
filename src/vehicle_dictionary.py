from math import hypot

types = {0: "car", 1: "motorcycle", 2: "truck"}


def get_vehicles_dict(frame, old_vehicles_dict: dict = None) -> dict:
    vehicles = {}
    for bbox in frame[0].boxes:
        xl, yt, xr, yb = bbox.xyxyn[0]
        vehicles[int(bbox.id)] = {'type': types[int(bbox.cls)], 'xl': float(xl), 'yt': float(yt),
                                  'xr': float(xr), 'yb': float(yb), 'confidence': float(bbox.conf),
                                  'center': {'x': float(xl + xr) / 2, 'y': float(yt + yb) / 2},
                                  'is_track': bbox.is_track, 'id': int(bbox.id), 'is_moving': False,
                                  'direction': {'Top': 0, 'Bottom': 0, 'Left': 0, 'Right': 0}}
        if old_vehicles_dict is not None and int(bbox.id) in old_vehicles_dict:
            vehicles[int(bbox.id)]['direction'] = direction_between_vehicles(old_vehicles_dict[int(bbox.id)],
                                                                             vehicles[int(bbox.id)])
            vehicles[int(bbox.id)]['is_moving'] = True

    vehicles = get_closest_vehicles(vehicles)
    return vehicles


def get_closest_vehicles(vehicles: dict) -> dict:
    for vehicle in vehicles.values():
        vehicle['id_closest_vehicle'] = None
        if vehicle['is_track'] and vehicle['is_moving']:
            min_distance = 1
            for other_vehicle in vehicles.values():
                if vehicle['id'] != other_vehicle['id'] or not other_vehicle['is_moving'] or not other_vehicle['is_track']:
                    distance = hypot(abs(vehicle['center']['x'] - other_vehicle['center']['x']),
                                     abs(vehicle['center']['y'] - other_vehicle['center']['y']))
                    if distance < min_distance:
                        min_distance = distance
                        vehicle['id_closest_vehicle'] = other_vehicle['id']
    return vehicles


def direction_between_vehicles(vehicle_1, vehicle_2) -> dict:
    diff_x = vehicle_1['center']['x'] - vehicle_2['center']['x']
    diff_y = vehicle_1['center']['y'] - vehicle_2['center']['y']

    direction = {'Top': 0, 'Bottom': 0, 'Left': 0, 'Right': 0}
    if diff_y < 0.0:
        direction['Top'] = 1
    else:
        direction['Bottom'] = 1
    if diff_x < 0.0:
        direction['Left'] = 1
    else:
        direction['Right'] = 1
    return direction