from vehicle_dictionary import direction_between_vehicles


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
            direction_between_vehicle = direction_between_vehicles(vehicle, closest_vehicle)
            distance = get_distance_from_closest_vehicle(vehicle, closest_vehicle, direction_between_vehicle)
            if distance < threshold:
                close_vehicles += 1
    if close_vehicles >= len(vehicles)*0.8:
        return True
    else:
        return False


def traffic_jam(vehicles, threshold: float) -> bool:
    return traffic_jam_condition(vehicles, threshold)
