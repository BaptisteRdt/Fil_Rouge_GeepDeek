from math import hypot

types = {0: "car", 1: "motorcycle", 2: "truck"}


def get_vehicles_dict(results) -> dict:
    vehicles = {}
    for bbox in results[0].boxes:
        xl, yt, xr, yb = bbox.xyxyn[0]
        vehicles[int(bbox.id)] = {'type': types[int(bbox.cls)], 'xl': float(xl), 'yt': float(yt),
                                  'xr': float(xr), 'yb': float(yb), 'confidence': float(bbox.conf),
                                  'center': {'x': float(xl + xr) / 2, 'y': float(yt + yb) / 2},
                                  'is_track': bbox.is_track, 'id': int(bbox.id)}
    return vehicles


def get_closest_vehicles(vehicles: dict) -> dict:
    for vehicle in vehicles.values():
        if vehicle['is_track']:
            min_distance = 1
            for other_vehicle in vehicles.values():
                if vehicle['id'] != other_vehicle['id']:
                    distance = hypot(abs(vehicle['center']['x'] - other_vehicle['center']['x']),
                                     abs(vehicle['center']['y'] - other_vehicle['center']['y']))
                    if distance < min_distance:
                        min_distance = distance
                        vehicle['id_closest_vehicle'] = other_vehicle['id']
    return vehicles


def direction_of_closest_vehicle(vehicle, closest_vehicle) -> dict:
    diff_x = vehicle['center']['x'] - closest_vehicle['center']['x']
    diff_y = vehicle['center']['y'] - closest_vehicle['center']['y']

    direction = {'Top': 0,'Bottom': 0,'Left': 0,'Right': 0}
    if diff_y < 0.0:
        direction['Top'] = 1
    else:
        direction['Bottom'] = 1
    if diff_x < 0.0:
        direction['Left'] = 1
    else:
        direction['Right'] = 1
    return direction


def get_distance_from_closest_vehicle(vehicle, closest_vehicle, direction) -> float:
    if direction['Top'] == 1:
        distance_y = vehicle['yt'] - closest_vehicle['yt']
    if direction['Bottom'] == 1:
        distance_y = vehicle['yb'] - closest_vehicle['yb']
    if direction['Left'] == 1:
        distance_x = vehicle['xl'] - closest_vehicle['xl']
    if direction['Right'] == 1:
        distance_x = vehicle['xr'] - closest_vehicle['xr']

    return (distance_x + distance_y) / 2


def traffic_jam_condition(vehicles: dict, threshold: float) -> bool:
    close_vehicles = 0
    for vehicle in vehicles.values():
        if vehicle['id_closest_vehicle'] is not None:
            closest_vehicle = vehicles[vehicle['id_closest_vehicle']]
            direction_of_vehicle = direction_of_closest_vehicle(vehicle, closest_vehicle)
            distance = get_distance_from_closest_vehicle(vehicle, closest_vehicle, direction_of_vehicle)
            if distance < threshold:
                close_vehicles += 1
    if close_vehicles >= len(vehicles)*0.8:
        return True
    else:
        return False


def traffic_jam(results, threshold: float) -> bool:
    vehicles = get_vehicles_dict(results)
    vehicles = get_closest_vehicles(vehicles)

    return traffic_jam_condition(vehicles, threshold)
