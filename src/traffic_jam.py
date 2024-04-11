from vehicle_dictionary import direction_between_vehicles


def get_distance_from_closest_vehicle(vehicle, closest_vehicle, direction) -> float:
    distance = 0
    if direction['Top'] == 1:
        distance = vehicle['yt'] - closest_vehicle['yb']
    if direction['Bottom'] == 1:
        distance = vehicle['yb'] - closest_vehicle['yt']
    if direction['Left'] == 1:
        distance = vehicle['xl'] - closest_vehicle['xr']
    if direction['Right'] == 1:
        distance = vehicle['xr'] - closest_vehicle['xl']

    return distance


def traffic_jam(vehicles) -> bool:
    close_vehicles = 0
    moving_vehicles = 0
    for vehicle in vehicles.values():
        if vehicle['is_moving']:
            moving_vehicles += 1
        if vehicle['id_closest_vehicle'] is not None:
            closest_vehicle = vehicles[vehicle['id_closest_vehicle']]
            direction_between_vehicle = direction_between_vehicles(vehicle, closest_vehicle)
            distance = get_distance_from_closest_vehicle(vehicle, closest_vehicle, direction_between_vehicle)
            if abs(distance) < vehicle['length']:
                close_vehicles += 1

    if close_vehicles >= moving_vehicles*0.7:
        return True
    else:
        return False