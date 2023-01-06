import math
import numpy
import scipy
import folium
import logging
import webbrowser
import osmnx as ox
import pandas as pd
import networkx as nx
import geopandas as gpd

from pathlib import Path
from pandas import Series
from shapely import affinity
from folium import Map, GeoJson, GeoJsonPopup, Choropleth
from networkx import MultiDiGraph
from geopandas import GeoDataFrame, GeoSeries
from shapely.geometry import Point, Polygon, LineString

standard_crs = 4326
block_id_label = 'block_id'
node_id_label = 'node_id'
node_distance_label = 'node_distance'


def remove_objects_outside_blocks_and_set_block_id(blocks: GeoDataFrame, objects: GeoDataFrame) -> GeoDataFrame:
    inside_objects = objects.sjoin(df=blocks[['id', 'geometry']].rename(columns={'id': 'id_right'}), how='left')
    if block_id_label in inside_objects.columns:
        logging.warning(f' objects geoDataFrame already contains {block_id_label} column, renamed to {block_id_label}_(2)')
        inside_objects = inside_objects.rename(columns={block_id_label: f'{block_id_label}_(2)'})
    inside_objects = inside_objects.rename(columns={'id_right': block_id_label})
    inside_objects = inside_objects[inside_objects[block_id_label].notna()].reset_index(drop=True)
    inside_objects = inside_objects.drop(columns=['index_right'])
    inside_objects = inside_objects.astype(dtype={block_id_label: int}, copy=False)
    logging.info(f' {len(objects) - len(inside_objects)} objects outside the blocks were deleted')
    return inside_objects


def get_geometry_angle(geometry: Polygon):
    simplified_coords = geometry.minimum_rotated_rectangle.boundary.coords
    sides = [LineString(coordinates=[c1, c2]) for c1, c2 in zip(simplified_coords, simplified_coords[1:])]
    longest_side = max(sides, key=lambda x: x.length)
    point1, point2 = longest_side.coords
    return math.degrees(math.atan2(point2[1] - point1[1], point2[0] - point1[0]))


def transform_point_objects_into_square_polygons_with_median_area(blocks: GeoDataFrame, objects: GeoDataFrame) -> GeoDataFrame:
    points = objects[objects.geometry.geom_type == 'Point'].copy()
    if points.empty:
        return objects

    polygons = objects[objects.geometry.geom_type == 'Polygon']
    median_area = polygons.geometry.area.median()
    square_length = math.sqrt(median_area)
    angles = list(blocks.geometry.apply(func=lambda x: get_geometry_angle(x)))
    points.geometry = points.apply(
        func=lambda x: generate_rotated_square_from_point(point=x.geometry, length=square_length, angle=angles[x[block_id_label]]), axis=1)
    return pd.concat(objs=[polygons, points]).reset_index()


def generate_rotated_square_from_point(point: Point, length: float, angle: float) -> Polygon:
    return affinity.rotate(geom=point.buffer(distance=length / 2.0, cap_style=3), angle=angle, origin='centroid')


def remove_contained_points_in_objects_from_geodataframe(data: GeoDataFrame, blocks: GeoDataFrame, objects: GeoDataFrame) -> GeoDataFrame:
    polygons = data[data.geometry.geom_type == 'Polygon']
    points = data[data.geometry.geom_type == 'Point']
    indexes = []
    for _, block in blocks.iterrows():
        block_points = points[points[block_id_label] == block['id']]
        if not block_points.empty:
            geometry_1 = polygons[polygons[block_id_label] == block['id']].geometry
            geometry_2 = objects[objects[block_id_label] == block['id']].geometry
            all_geometry = pd.concat([geometry_1, geometry_2], ignore_index=True).unary_union
            contained_points = block_points[block_points.within(all_geometry)]
            indexes.extend(contained_points.index)
    return data.drop(indexes, axis=0)


def load_walk_graph(blocks: GeoDataFrame, to_crs) -> MultiDiGraph:
    graph = ox.graph_from_polygon(polygon=blocks.to_crs(crs=standard_crs).geometry.unary_union.convex_hull, network_type='walk', simplify=True)
    return ox.project_graph(G=graph, to_crs=to_crs)


def get_utm_crs(geometry) -> str:
    mean_longitude = geometry.representative_point().x.mean()
    utm_zone = int(math.floor((mean_longitude + 180.0) / 6.0) + 1.0)
    return f'+proj=utm +zone={utm_zone} +ellps=WGS84 +datum=WGS84 +units=m +no_defs'


def graph_to_polygon(graph: MultiDiGraph) -> Polygon:
    node_points = [Point((data['x'], data['y'])) for node, data in graph.nodes(data=True)]
    return GeoSeries(node_points).unary_union.convex_hull


def polygon_to_x_y_coords(polygon: Polygon) -> [float, float]:
    centroid = polygon.centroid
    return centroid.x, centroid.y


def geodataframe_to_x_y_coords(objects: GeoDataFrame) -> [Series, Series]:
    x_coords, y_coords = zip(*objects.geometry.apply(func=lambda x: polygon_to_x_y_coords(x)))
    return x_coords, y_coords


def get_gravity_for_building_and_playground(area_playground: float, distance: float) -> float:
    return area_playground / ((distance + 50.0) ** 2.0)


def set_nearest_nodes_to_objects(objects: GeoDataFrame, walk_graph: MultiDiGraph) -> None:
    x_coords, y_coords = geodataframe_to_x_y_coords(objects=objects)
    objects[node_id_label], objects[node_distance_label] = ox.nearest_nodes(G=walk_graph, X=x_coords, Y=y_coords, return_dist=True)


def distribute_people_to_playgrounds(living_buildings: GeoDataFrame, playgrounds: GeoDataFrame, walk_graph: MultiDiGraph,
                                     max_meters_to_playground: int) -> None:
    set_nearest_nodes_to_objects(playgrounds, walk_graph)
    set_nearest_nodes_to_objects(living_buildings, walk_graph)
    for i, living_building in living_buildings.iterrows():
        subgraph = nx.ego_graph(G=walk_graph, n=living_building[node_id_label], radius=max_meters_to_playground, distance='length')
        available_playgrounds = playgrounds[playgrounds[node_id_label].isin(subgraph.nodes)].reset_index(drop=True)
        if available_playgrounds.empty:
            living_building['undistributed'] = living_building['population']
            continue

        gravity = []
        for _, playground in available_playgrounds.iterrows():
            distance = nx.shortest_path_length(G=subgraph, source=living_building[node_id_label], target=playground[node_id_label], weight='length')
            distance += living_building[node_distance_label] + playground[node_distance_label]
            gravity.append(get_gravity_for_building_and_playground(area_playground=playground['area'], distance=distance))

        whole_gravity = sum(gravity)
        distribution = list([living_building['population'] * g / whole_gravity for g in gravity])
        for j, playground in available_playgrounds.iterrows():
            playground['living_buildings_neighbours'].append((i, distribution[j]))


def calculate_playgrounds_over_capacity(playgrounds: GeoDataFrame, living_buildings: GeoDataFrame) -> None:
    undistributed = list(living_buildings['undistributed'])
    fullness = []
    for _, playground in playgrounds.iterrows():
        real_capacity = sum(population for _, population in playground['living_buildings_neighbours'])
        fullness.append(real_capacity / playground['capacity'])
        over_capacity = real_capacity - playground['capacity']
        if over_capacity <= 0:
            continue

        for i, population in playground['living_buildings_neighbours']:
            undistributed[i] += population / real_capacity * over_capacity

    playgrounds['fullness'] = fullness
    living_buildings['undistributed'] = undistributed
    living_buildings['undistributed'] /= living_buildings['population']


def get_blocks(filename: Path) -> GeoDataFrame:
    blocks = gpd.read_file(filename=filename).set_crs(crs=standard_crs)
    utm_crs = get_utm_crs(blocks.geometry)
    blocks = blocks.explode(ignore_index=True).to_crs(crs=utm_crs)
    blocks['id'] = blocks.index
    return blocks


def get_buildings(filename: Path, blocks: GeoDataFrame, to_crs) -> GeoDataFrame:
    buildings = gpd.read_file(filename=filename).set_crs(crs=standard_crs).to_crs(crs=to_crs)
    buildings = remove_objects_outside_blocks_and_set_block_id(blocks=blocks, objects=buildings)
    return buildings


def get_living_buildings(buildings: GeoDataFrame) -> GeoDataFrame:
    living_buildings = buildings[buildings['population'] > 0].reset_index(drop=True)
    living_buildings['undistributed'] = [0.0] * len(living_buildings)
    return living_buildings


def get_playgrounds(filename: Path, blocks: GeoDataFrame, buildings: GeoDataFrame, to_crs) -> GeoDataFrame:
    playground_area_per_people = 0.5
    playgrounds = gpd.read_file(filename=filename).set_crs(crs=standard_crs).to_crs(crs=to_crs)
    playgrounds = playgrounds.explode(ignore_index=True)
    playgrounds = remove_objects_outside_blocks_and_set_block_id(blocks=blocks, objects=playgrounds)
    playgrounds = remove_contained_points_in_objects_from_geodataframe(data=playgrounds, blocks=blocks, objects=buildings)
    playgrounds = transform_point_objects_into_square_polygons_with_median_area(blocks=blocks, objects=playgrounds)
    playgrounds['area'] = round(playgrounds.geometry.area, 2)
    playgrounds['population'] = [0] * len(playgrounds)
    playgrounds['living_buildings_neighbours'] = [[] for _ in range(len(playgrounds))]
    playgrounds['capacity'] = playgrounds['area'] / playground_area_per_people
    return playgrounds


def get_walk_graph(filename: Path, blocks: GeoDataFrame, to_crs) -> MultiDiGraph:
    if filename.exists():
        walk_graph = ox.load_graphml(filepath=filename)
    else:
        walk_graph = load_walk_graph(blocks=blocks, to_crs=to_crs)
        ox.save_graphml(G=walk_graph, filepath=filename)
    return walk_graph


def save_map_and_open_in_browser(filename: Path, folium_map) -> None:
    open_in_new_tab = 2
    folium_map.save(filename)
    webbrowser.open(url=str(filename), new=open_in_new_tab)


def draw_result_map(filename: Path, blocks: GeoDataFrame, playgrounds: GeoDataFrame) -> None:
    location = blocks.to_crs(standard_crs).geometry.unary_union.convex_hull.centroid
    folium_map = Map(location=(location.y, location.x), zoom_start=13, tiles=None)

    GeoJson(data=blocks[['geometry', 'id']],
            name='Кварталы',
            popup=GeoJsonPopup(fields=['id'], aliases=['Номер квартала']),
            style_function=lambda x: {'fillColor': '#f5f5f5', 'lineColor': '#ffffbf', 'weight': '2'},
            highlight_function=lambda x: {'weight': '4'},
            smooth_factor=0.1
            ).add_to(parent=folium_map)

    Choropleth(geo_data=playgrounds[['geometry', 'fullness']],
               name='Детские площадки (хороплет)',
               data=playgrounds['fullness'],
               key_on='feature.id',
               fill_color='PuRd',
               line_opacity=0.5,
               legend_name="Загруженность площадок",
               bins=[0.0, 0.25, 0.5, 0.75, 1.0, 1.5, max(playgrounds['fullness'])]
               ).add_to(folium_map)

    GeoJson(data=playgrounds[['geometry', 'area', 'capacity', 'fullness']],
            name='Детские площадки (подсказки)',
            style_function=lambda x: {'color': 'black', 'fillColor': 'transparent', 'weight': 0.5},
            popup=GeoJsonPopup(
                fields=['area', 'capacity', 'fullness'],
                aliases=['Площадь м²', 'Вместимость', 'Загруженность']
            ),
            highlight_function=lambda x: {'weight': 2},
            ).add_to(folium_map)

    folium.TileLayer('cartodbpositron', overlay=False, name='Светлая тема').add_to(folium_map)
    folium.TileLayer('cartodbdark_matter', overlay=False, name='Темная тема').add_to(folium_map)
    folium.LayerControl(collapsed=False).add_to(folium_map)

    save_map_and_open_in_browser(filename=filename, folium_map=folium_map)


def main():
    test_folder = Path() / 'data' / 'test'
    blocks = get_blocks(filename=test_folder / 'blocks.geojson')
    utm_crs = blocks.crs
    buildings = get_buildings(filename=test_folder / 'input_buildings.geojson', blocks=blocks, to_crs=utm_crs)
    living_buildings = get_living_buildings(buildings=buildings)
    playgrounds = get_playgrounds(filename=Path('playgrounds.geojson'), blocks=blocks, buildings=buildings, to_crs=utm_crs)
    # walk_graph = get_walk_graph(filename=test_folder / 'walk_graph.graphml', blocks=blocks, to_crs=utm_crs)
    # distribute_people_to_playgrounds(living_buildings=living_buildings, playgrounds=playgrounds, walk_graph=walk_graph, max_meters_to_playground=500)
    # calculate_playgrounds_over_capacity(playgrounds=playgrounds, living_buildings=living_buildings)

    draw_result_map(filename=Path('map.html'), blocks=blocks, playgrounds=playgrounds)

    # playgrounds.drop(columns=['living_buildings_neighbours']).to_crs(standard_crs).to_file(filename='playgrounds.geojson', driver='GeoJSON')
    # living_buildings.to_crs(standard_crs).to_file(filename='living_buildings.geojson', driver='GeoJSON')


main()
