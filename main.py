import math
import scipy
import folium
import shapely
import logging
import webbrowser
import numpy as np
import osmnx as ox
import pandas as pd
import networkx as nx
import geopandas as gpd
import matplotlib.pyplot as plt

from pathlib import Path
from pandas import Series
from shapely import affinity
from matplotlib import ticker
from networkx import MultiDiGraph
from matplotlib.pyplot import figure
from geopandas import GeoDataFrame, GeoSeries
from plotly.graph_objects import Figure, Table
from shapely.geometry import Point, Polygon, LineString
from folium import Map, GeoJson, GeoJsonPopup, Choropleth

standard_crs = 4326
non_gravity_distance = 50.0

pd.options.mode.chained_assignment = None  # default='warn'


def get_blocks(filename: Path) -> GeoDataFrame:
    blocks = gpd.read_file(filename=filename).set_crs(crs=standard_crs)
    utm_crs = get_utm_crs(blocks.geometry)
    blocks = blocks.explode(ignore_index=True).to_crs(crs=utm_crs)
    blocks['id'] = blocks.index
    return blocks


def get_utm_crs(geometry) -> str:
    mean_longitude = geometry.representative_point().x.mean()
    utm_zone = int(math.floor((mean_longitude + 180.0) / 6.0) + 1.0)
    return f'+proj=utm +zone={utm_zone} +ellps=WGS84 +datum=WGS84 +units=m +no_defs'


def get_buildings(filename: Path, blocks: GeoDataFrame, to_crs) -> GeoDataFrame:
    buildings = gpd.read_file(filename=filename).set_crs(crs=standard_crs).to_crs(crs=to_crs)
    buildings = remove_objects_outside_blocks_and_set_block_id(blocks=blocks, objects=buildings)
    return buildings


def remove_objects_outside_blocks_and_set_block_id(blocks: GeoDataFrame, objects: GeoDataFrame) -> GeoDataFrame:
    inside_objects = objects.sjoin(df=blocks[['id', 'geometry']].rename(columns={'id': 'id_right'}), how='left')
    if 'block_id' in inside_objects.columns:
        logging.warning(' objects geoDataFrame already contains block_id column, renamed to block_id_(2)')
        inside_objects = inside_objects.rename(columns={'block_id': 'block_id_(2)'})
    inside_objects = inside_objects.rename(columns={'id_right': 'block_id'})
    inside_objects = inside_objects[inside_objects['block_id'].notna()].reset_index(drop=True)
    inside_objects = inside_objects.drop(columns=['index_right'])
    inside_objects = inside_objects.astype(dtype={'block_id': int}, copy=False)
    logging.info(f' {len(objects) - len(inside_objects)} objects outside the blocks were deleted')
    return inside_objects


def get_living_buildings(buildings: GeoDataFrame) -> GeoDataFrame:
    living_buildings = buildings[buildings['population'] > 0].reset_index(drop=True)
    return living_buildings


def get_playgrounds(filename: Path, blocks: GeoDataFrame, buildings: GeoDataFrame, playground_area_per_people: float, to_crs) -> GeoDataFrame:
    playgrounds = gpd.read_file(filename=filename).set_crs(crs=standard_crs).to_crs(crs=to_crs)
    playgrounds = playgrounds.explode(ignore_index=True)
    playgrounds = remove_objects_outside_blocks_and_set_block_id(blocks=blocks, objects=playgrounds)
    playgrounds = remove_contained_points_in_objects_from_geodataframe(data=playgrounds, blocks=blocks, objects=buildings)
    playgrounds = transform_point_objects_into_square_polygons_with_median_area(blocks=blocks, objects=playgrounds)
    playgrounds['area'] = round(playgrounds.geometry.area, 2)
    playgrounds['population'] = [0] * len(playgrounds)
    playgrounds['living_buildings_neighbours'] = [[] for _ in range(len(playgrounds))]
    playgrounds['capacity'] = playgrounds['area'] / playground_area_per_people
    playgrounds['fullness'] = [0] * len(playgrounds)
    return playgrounds


def remove_contained_points_in_objects_from_geodataframe(data: GeoDataFrame, blocks: GeoDataFrame, objects: GeoDataFrame) -> GeoDataFrame:
    polygons = data[data.geometry.geom_type == 'Polygon']
    points = data[data.geometry.geom_type == 'Point']
    indexes = []
    for _, block in blocks.iterrows():
        block_points = points[points['block_id'] == block['id']]
        if not block_points.empty:
            geometry_1 = polygons[polygons['block_id'] == block['id']].geometry
            geometry_2 = objects[objects['block_id'] == block['id']].geometry
            all_geometry = pd.concat([geometry_1, geometry_2], ignore_index=True).unary_union
            contained_points = block_points[block_points.within(all_geometry)]
            indexes.extend(contained_points.index)
    return data.drop(indexes, axis=0)


def transform_point_objects_into_square_polygons_with_median_area(blocks: GeoDataFrame, objects: GeoDataFrame) -> GeoDataFrame:
    points = objects[objects.geometry.geom_type == 'Point'].copy()
    if points.empty:
        return objects

    polygons = objects[objects.geometry.geom_type == 'Polygon']
    mean_area = polygons.geometry.area.mean()
    square_length = math.sqrt(mean_area)
    angles = list(blocks.geometry.apply(func=lambda x: get_geometry_angle(x)))
    points.geometry = points.apply(
        func=lambda x: generate_rotated_square_from_point(point=x.geometry, length=square_length, angle=angles[x['block_id']]), axis=1)
    return pd.concat(objs=[polygons, points]).reset_index(drop=True)


def get_geometry_angle(geometry: Polygon):
    simplified_coords = geometry.minimum_rotated_rectangle.boundary.coords
    sides = [LineString(coordinates=[c1, c2]) for c1, c2 in zip(simplified_coords, simplified_coords[1:])]
    longest_side = max(sides, key=lambda x: x.length)
    point1, point2 = longest_side.coords
    return math.degrees(math.atan2(point2[1] - point1[1], point2[0] - point1[0]))


def generate_rotated_square_from_point(point: Point, length: float, angle: float) -> Polygon:
    square_form = 3
    return affinity.rotate(geom=point.buffer(distance=length / 2.0, cap_style=square_form), angle=angle, origin='centroid')


def get_graph(filename: Path, blocks: GeoDataFrame, network_type: str, to_crs) -> MultiDiGraph:
    if filename.exists():
        walk_graph = ox.load_graphml(filepath=filename)
    else:
        walk_graph = load_graph(blocks=blocks, network_type=network_type, to_crs=to_crs)
        ox.save_graphml(G=walk_graph, filepath=filename)
    return walk_graph


def load_graph(blocks: GeoDataFrame, network_type: str, to_crs) -> MultiDiGraph:
    graph = ox.graph_from_polygon(polygon=blocks.to_crs(crs=standard_crs).unary_union.convex_hull, network_type=network_type, simplify=False)
    return ox.project_graph(G=graph, to_crs=to_crs)


def distribute_people_to_playgrounds(living_buildings: GeoDataFrame, playgrounds: GeoDataFrame, walk_graph: MultiDiGraph, max_meters_to_playground: float) -> None:
    set_nearest_nodes_to_objects(playgrounds, walk_graph)
    set_nearest_nodes_to_objects(living_buildings, walk_graph)
    living_buildings['undistributed_population'] = [0.0] * len(living_buildings)
    is_isochrone_cached = 'isochrone_graph' in living_buildings.columns
    isochrone_graphs = []
    gravities = []
    for i, living_building in living_buildings.iterrows():
        isochrone_graph = living_building['isochrone_graph'] if is_isochrone_cached else nx.ego_graph(G=walk_graph, n=living_building['node_id'], radius=max_meters_to_playground,
                                                                                                      distance='length')
        isochrone_graphs.append(isochrone_graph)
        available_playgrounds = playgrounds[playgrounds['node_id'].isin(isochrone_graph.nodes)].reset_index(drop=True)
        gravity = []
        gravities.append(gravity)
        if available_playgrounds.empty:
            living_building['undistributed_population'] = living_building['population']
            continue

        for _, playground in available_playgrounds.iterrows():
            distance = nx.shortest_path_length(G=isochrone_graph, source=living_building['node_id'], target=playground['node_id'], weight='length')
            distance += living_building['node_distance'] + playground['node_distance']
            gravity.append(get_gravity_for_building_and_playground(area_playground=playground['area'], distance=distance))

        whole_gravity = sum(gravity)
        distribution = list([living_building['population'] * g / whole_gravity for g in gravity])
        for j, playground in available_playgrounds.iterrows():
            playground['living_buildings_neighbours'].append((i, distribution[j]))

    living_buildings['gravity'] = gravities
    living_buildings['isochrone_graph'] = isochrone_graphs


def set_nearest_nodes_to_objects(objects: GeoDataFrame, walk_graph: MultiDiGraph) -> None:
    x_coords, y_coords = geodataframe_to_x_y_coords(objects=objects)
    objects['node_id'], objects['node_distance'] = ox.nearest_nodes(G=walk_graph, X=x_coords, Y=y_coords, return_dist=True)


def geodataframe_to_x_y_coords(objects: GeoDataFrame) -> [Series, Series]:
    x_coords, y_coords = zip(*objects.geometry.apply(func=lambda x: polygon_to_x_y_coords(x)))
    return x_coords, y_coords


def polygon_to_x_y_coords(polygon: Polygon) -> [float, float]:
    centroid = polygon.centroid
    return centroid.x, centroid.y


def get_gravity_for_building_and_playground(area_playground: float, distance: float) -> float:
    return area_playground / ((distance + non_gravity_distance) ** 2.0)


def calculate_playgrounds_fullness(playgrounds: GeoDataFrame, living_buildings: GeoDataFrame) -> None:
    undistributed_population = list(living_buildings['undistributed_population'])
    fullness = []
    for _, playground in playgrounds.iterrows():
        real_capacity = sum(population for _, population in playground['living_buildings_neighbours'])
        fullness.append(real_capacity / playground['capacity'])
        if real_capacity == 0.0:
            continue

        over_capacity_proportion = (real_capacity - playground['capacity']) / real_capacity
        if over_capacity_proportion <= 0:
            continue

        for i, population in playground['living_buildings_neighbours']:
            undistributed_population[i] += population * over_capacity_proportion

    playgrounds['fullness'] = fullness
    living_buildings['undistributed_population'] = undistributed_population
    living_buildings['undistributed_proportion'] = living_buildings['undistributed_population'] / living_buildings['population']


def save_output_data_before(folder: Path, blocks: GeoDataFrame, playgrounds: GeoDataFrame, living_buildings: GeoDataFrame, gdf_walk_graph: GeoDataFrame) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    playgrounds = playgrounds.drop(columns=['living_buildings_neighbours'])
    playgrounds.to_crs(standard_crs).to_file(filename=str(folder / 'output_playgrounds_before.geojson'), driver='GeoJSON')
    blocks.to_crs(standard_crs).to_file(filename=str(folder / 'output_blocks.geojson'), driver='GeoJSON')
    living_buildings = living_buildings.drop(columns=['isochrone_graph', 'gravity'])
    living_buildings.to_crs(standard_crs).to_file(filename=str(folder / 'output_buildings_before.geojson'), driver='GeoJSON')
    gdf_walk_graph.to_crs(standard_crs).to_file(filename=str(folder / 'walk_graph.geojson'), driver='GeoJSON')


def draw_map(filename: Path, blocks: GeoDataFrame = None, playgrounds: GeoDataFrame = None, living_buildings: GeoDataFrame = None, gdf_walk_graph: GeoDataFrame = None,
             gdf_drive_graph: GeoDataFrame = None, physical_objects: GeoDataFrame = None, free_areas: GeoDataFrame = None, new_playgrounds: GeoDataFrame = None) -> None:
    location = blocks.to_crs(standard_crs).unary_union.convex_hull.centroid
    folium_map = Map(location=(location.y, location.x), zoom_start=13, tiles=None)

    if blocks is not None:
        GeoJson(data=blocks[['geometry', 'id']],
                name='Кварталы',
                popup=GeoJsonPopup(fields=['id'], aliases=['Номер квартала']),
                style_function=lambda x: {'fillColor': 'transparent', 'weight': '2'},
                highlight_function=lambda x: {'weight': '4'}
                ).add_to(parent=folium_map)

    if gdf_walk_graph is not None:
        GeoJson(data=gdf_walk_graph[['geometry', 'length']],
                name='Пешеходный граф',
                popup=GeoJsonPopup(fields=['length'], aliases=['Длина в метрах']),
                style_function=lambda x: {'weight': '1'},
                highlight_function=lambda x: {'weight': '2'},
                show=False
                ).add_to(parent=folium_map)

    if free_areas is not None:
        GeoJson(data=free_areas[['geometry', 'area']],
                name='Свободная площадь',
                popup=GeoJsonPopup(fields=['area'], aliases=['Площадь м²']),
                style_function=lambda x: {'color': 'green', 'weight': '1'},
                highlight_function=lambda x: {'weight': '2'},
                show=False
                ).add_to(parent=folium_map)

    if gdf_drive_graph is not None:
        GeoJson(data=gdf_drive_graph[['geometry', 'length']],
                name='Автомобильный граф',
                popup=GeoJsonPopup(fields=['length'], aliases=['Длина в метрах']),
                style_function=lambda x: {'color': 'orange', 'weight': '1'},
                highlight_function=lambda x: {'weight': '2'},
                show=False
                ).add_to(parent=folium_map)

    if physical_objects is not None:
        GeoJson(data=physical_objects[['geometry']],
                name='Физические объекты',
                style_function=lambda x: {'color': 'grey', 'weight': '1'},
                highlight_function=lambda x: {'weight': '2'},
                show=False
                ).add_to(parent=folium_map)

    if living_buildings is not None:
        Choropleth(geo_data=living_buildings[['geometry', 'undistributed_proportion']],
                   name='Жилые здания (хороплет)',
                   data=living_buildings['undistributed_proportion'],
                   key_on='feature.id', fill_color='OrRd',
                   fill_opacity=1.0, line_opacity=0.5,
                   bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                   legend_name='Доля нераспределенных жителей'
                   ).add_to(folium_map)

        living_buildings['undistributed%'] = round((living_buildings['undistributed_proportion'] * 100.0), 1).apply(func=lambda x: f'{x}%')
        GeoJson(data=living_buildings[['geometry', 'population', 'undistributed%']],
                name='Жилые здания (подсказки)',
                style_function=lambda x: {'color': 'black', 'fillColor': 'transparent', 'weight': 0.5},
                popup=GeoJsonPopup(fields=['population', 'undistributed%'], aliases=['Жителей', 'Нераспределенные жители']),
                highlight_function=lambda x: {'weight': 2}
                ).add_to(folium_map)

    if playgrounds is not None:
        Choropleth(geo_data=playgrounds[['geometry', 'fullness']],
                   name='Детские площадки (хороплет)',
                   data=playgrounds['fullness'],
                   key_on='feature.id', fill_color='PuRd',
                   fill_opacity=1.0, line_opacity=0.5,
                   legend_name='Загруженность площадок',
                   bins=[0.0, 0.25, 0.5, 0.75, 1.0, 1.5, max(playgrounds['fullness'])]
                   ).add_to(folium_map)

        playgrounds['fullness%'] = round((playgrounds['fullness'] * 100.0), 1).apply(func=lambda x: f'{x}%')
        GeoJson(data=playgrounds[['geometry', 'area', 'capacity', 'fullness%']],
                name='Детские площадки (подсказки)',
                style_function=lambda x: {'color': 'black', 'fillColor': 'transparent', 'weight': 0.5},
                popup=GeoJsonPopup(fields=['area', 'capacity', 'fullness%'], aliases=['Площадь м²', 'Вместимость людей', 'Загруженность']),
                highlight_function=lambda x: {'weight': 2},
                ).add_to(folium_map)

    if new_playgrounds is not None:
        GeoJson(data=new_playgrounds[['geometry']],
                name='Новые детские площадки (обводка)',
                style_function=lambda x: {'color': 'red', 'fillColor': 'transparent', 'weight': 1},
                ).add_to(folium_map)

    folium.TileLayer('cartodbpositron', overlay=False, name='Светлая тема').add_to(folium_map)
    folium.TileLayer('cartodbdark_matter', overlay=False, name='Темная тема').add_to(folium_map)
    folium.LayerControl(collapsed=False).add_to(folium_map)

    save_map_and_open_in_browser(filename=filename, folium_map=folium_map)


def save_map_and_open_in_browser(filename: Path, folium_map) -> None:
    open_in_new_tab = 2
    folium_map.save(filename)
    webbrowser.open(url=str(filename), new=open_in_new_tab)


def get_physical_objects(filename: Path, blocks: GeoDataFrame, to_crs) -> GeoDataFrame:
    physical_objects = gpd.read_file(filename=filename).set_crs(crs=standard_crs).to_crs(crs=to_crs)
    physical_objects = remove_objects_outside_blocks_and_set_block_id(blocks=blocks, objects=physical_objects)
    return physical_objects


def get_free_areas(blocks: GeoDataFrame, physical_objects: GeoDataFrame, gdf_drive_graph: GeoDataFrame, buildings: GeoDataFrame, playgrounds: GeoDataFrame,
                   living_buildings: GeoDataFrame, min_free_area: float, min_undistributed_population: float) -> GeoDataFrame:
    min_meters_to_roads = 20
    geometry_1 = gdf_drive_graph.geometry.buffer(min_meters_to_roads).geometry.set_crs(blocks.crs)

    min_meters_to_buildings = 10
    geometry_2 = physical_objects.geometry.buffer(min_meters_to_buildings).geometry.set_crs(blocks.crs)
    geometry_3 = buildings.geometry.buffer(min_meters_to_buildings).geometry.set_crs(blocks.crs)

    undistributed_living_buildings = living_buildings[living_buildings['undistributed_population'] >= min_undistributed_population]
    if undistributed_living_buildings.empty:
        return undistributed_living_buildings

    free_areas = GeoSeries(data=graphs_to_polygons(graphs=undistributed_living_buildings['isochrone_graph']).unary_union)
    free_areas = free_areas.intersection(other=blocks.unary_union)
    free_areas = free_areas.difference(other=pd.concat([geometry_1, geometry_2, geometry_3, playgrounds.geometry], ignore_index=True).unary_union)
    free_areas = free_areas.explode(ignore_index=True).set_crs(blocks.crs)
    free_areas = free_areas[free_areas.area >= min_free_area]
    free_areas = gpd.GeoDataFrame(data={'area': round(free_areas.area, 2)}, geometry=free_areas, crs=blocks.crs)
    return free_areas


def graphs_to_polygons(graphs: list[MultiDiGraph]) -> GeoSeries:
    polygons = []
    for graph in graphs:
        node_points = [Point((data['x'], data['y'])) for node, data in graph.nodes(data=True)]
        polygons.append(GeoSeries(node_points).unary_union.convex_hull)
    return GeoSeries(data=polygons)


def geodataframe_to_polygons_grid(objects: GeoDataFrame, cell_width: float, cell_height: float):
    polygons_cells = []
    for polygon in objects.geometry:
        min_x, min_y, max_x, max_y = polygon.bounds

        columns = list(np.arange(min_x, max_x + cell_width, cell_width))
        rows = list(np.arange(min_y, max_y + cell_height, cell_height))

        polygon_cells = []
        for y in rows[:-1]:
            for x in columns[:-1]:
                polygon_cells.append(shapely.box(x, y, x + cell_width, y + cell_height))
        truncated_cells = GeoSeries(polygon_cells).intersection(polygon)
        truncated_cells = truncated_cells[~truncated_cells.is_empty]
        polygons_cells.extend(list(truncated_cells))
    grid = gpd.GeoDataFrame(geometry=polygons_cells, crs=objects.crs)
    grid['area'] = round(grid.geometry.area, 2)
    grid = grid[grid['area'] >= 1.0]
    return grid


def generate_playgrounds(undistributed_living_buildings: GeoDataFrame, free_areas: GeoDataFrame, walk_graph: MultiDiGraph, playground_area_per_people: float) -> Series:
    set_nearest_nodes_to_objects(free_areas, walk_graph)
    new_playgrounds = []
    for _, building in undistributed_living_buildings.sort_values(by='undistributed_proportion', ascending=False).iterrows():
        undistributed_area = building['undistributed_population'] * playground_area_per_people
        isochrone_graph = building['isochrone_graph']
        available_free_areas = free_areas[free_areas['node_id'].isin(isochrone_graph.nodes)]
        whole_gravity = sum(building['gravity'])
        population = building['population']
        optimal_distance = get_optimal_distance_for_playground(playground_area=undistributed_area, gravity=whole_gravity, building_population=population,
                                                               playground_population=undistributed_area / playground_area_per_people)
        distances = []
        for node_id in available_free_areas['node_id']:
            distances.append(nx.shortest_path_length(G=isochrone_graph, source=building['node_id'], target=node_id, weight='length') - optimal_distance)
        available_free_areas['distance'] = abs(np.array(distances) + building['node_distance'] + available_free_areas['node_distance'])
        start_point = min(available_free_areas.iterrows(), key=lambda x: x[1]['distance'])[1].geometry.centroid
        available_free_areas['distance'] = available_free_areas.apply(func=lambda x: x.geometry.centroid.distance(start_point), axis=1)
        available_free_areas = available_free_areas.sort_values(by='distance')

        areas_count = 0
        for _, available_area in available_free_areas.iterrows():
            undistributed_area -= available_area['area']
            areas_count += 1
            if undistributed_area <= 0:
                break

        selected_areas = available_free_areas.iloc[:areas_count]
        new_playgrounds.extend(list(selected_areas.geometry))
        free_areas = free_areas.drop(selected_areas.index)

    return GeoSeries(data=GeoSeries(data=new_playgrounds).unary_union).explode(ignore_index=True)


def get_optimal_distance_for_playground(playground_area: float, gravity: float, building_population: float, playground_population: float) -> float:
    if gravity == 0.0:
        return 0.0

    b = non_gravity_distance * 2.0
    c = ((building_population - playground_population) * playground_area / playground_population / gravity) - (non_gravity_distance ** 2.0)
    d = math.sqrt((b ** 2.0) + 4.0 * c)

    return (max(-b + d, -b - d)) / 2.0


def save_output_data_after(folder: Path, playgrounds: GeoDataFrame, living_buildings: GeoDataFrame, gdf_drive_graph: GeoDataFrame, new_playgrounds: GeoDataFrame) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    playgrounds = playgrounds.drop(columns=['living_buildings_neighbours'])
    playgrounds.to_crs(standard_crs).to_file(filename=str(folder / 'output_playgrounds_after.geojson'), driver='GeoJSON')
    living_buildings = living_buildings.drop(columns=['isochrone_graph', 'gravity'])
    living_buildings.to_crs(standard_crs).to_file(filename=str(folder / 'output_buildings_after.geojson'), driver='GeoJSON')
    gdf_drive_graph.to_crs(standard_crs).to_file(filename=str(folder / 'drive_graph.geojson'), driver='GeoJSON')
    new_playgrounds.to_crs(standard_crs).to_file(filename=str(folder / 'new_playgrounds.geojson'), driver='GeoJSON')


def show_statistics(statistics: dict) -> None:
    values = [[key, *value] for key, value in statistics.items()]
    fig = Figure(data=[Table(header=dict(values=['Показатель', 'До', 'После']), cells=dict(values=values))])
    fig.show()


def save_and_show_graphics(folder: Path, undistributed_proportion_before: np.array, undistributed_proportion_after: np.array, fullness_before: np.array,
                           fullness_after: np.array) -> None:
    plt.rcParams['font.size'] = 18

    figure(figsize=(20, 13), dpi=80)
    plt.hist([undistributed_proportion_before * 100.0, undistributed_proportion_after * 100.0], bins=30, density=True)
    plt.title('Распределение необеспеченных жилых домов')
    plt.legend(['До', 'После'])
    plt.ylabel('Доля домов')
    plt.xlabel('Процент необеспеченности')
    plt.savefig(folder / 'undistributed_proportion.png')
    plt.show()

    figure(figsize=(20, 13), dpi=80)
    plt.hist([fullness_before * 100.0, fullness_after * 100.0], bins=30, density=True)
    plt.title('Распределение нагрузки на площадки')
    plt.legend(['До', 'После'])
    plt.ylabel('Доля площадок')
    plt.xlabel('Процент нагрузки')
    plt.gca().xaxis.set_major_formatter(ticker.PercentFormatter())
    plt.savefig(folder / 'fullness.png')
    plt.show()


def main():
    max_meters_to_playground = 500.0
    playground_area_per_people = 0.5
    min_playground_area = 25.0
    min_undistributed_proportion = 0.2
    min_undistributed_population = min_playground_area / playground_area_per_people

    input_data_folder, output_data_folder = Path() / 'data' / 'input', Path() / 'data' / 'output'
    blocks = get_blocks(filename=input_data_folder / 'input_blocks.geojson')
    utm_crs = blocks.crs

    buildings = get_buildings(filename=input_data_folder / 'input_buildings.geojson', blocks=blocks, to_crs=utm_crs)
    living_buildings = get_living_buildings(buildings=buildings)
    playgrounds = get_playgrounds(filename=input_data_folder / 'input_playgrounds.geojson', blocks=blocks, buildings=buildings,
                                  playground_area_per_people=playground_area_per_people, to_crs=utm_crs)

    walk_graph = get_graph(filename=input_data_folder / 'walk_graph.graphml', blocks=blocks, network_type='walk', to_crs=utm_crs)
    distribute_people_to_playgrounds(living_buildings=living_buildings, playgrounds=playgrounds, walk_graph=walk_graph, max_meters_to_playground=max_meters_to_playground)
    calculate_playgrounds_fullness(playgrounds=playgrounds, living_buildings=living_buildings)

    undistributed_proportion_before = living_buildings['undistributed_proportion']
    fullness_before = playgrounds['fullness']
    statistics = {'Количество площадок': [len(playgrounds)], 'Общая площадь м²': [sum(playgrounds['area'])],
                  'Нераспределенных жителей': [round(sum(living_buildings['undistributed_population']))]}
    gdf_walk_graph = ox.graph_to_gdfs(G=walk_graph, nodes=False)[['geometry', 'length']].reset_index(drop=True)
    save_output_data_before(folder=output_data_folder, blocks=blocks, playgrounds=playgrounds, living_buildings=living_buildings, gdf_walk_graph=gdf_walk_graph)
    draw_map(filename=output_data_folder / 'map_before.html', blocks=blocks, playgrounds=playgrounds, living_buildings=living_buildings, gdf_walk_graph=gdf_walk_graph)

    drive_graph = get_graph(filename=input_data_folder / 'drive_graph.graphml', blocks=blocks, network_type='drive', to_crs=utm_crs)
    physical_objects = get_physical_objects(filename=input_data_folder / 'input_physical_objects.geojson', blocks=blocks, to_crs=utm_crs)
    gdf_drive_graph = ox.graph_to_gdfs(G=drive_graph, nodes=False)[['geometry', 'length']].reset_index(drop=True)
    free_areas = get_free_areas(blocks=blocks, gdf_drive_graph=gdf_drive_graph, physical_objects=physical_objects, buildings=buildings, playgrounds=playgrounds,
                                living_buildings=living_buildings, min_free_area=min_playground_area, min_undistributed_population=min_undistributed_population)

    free_areas = geodataframe_to_polygons_grid(objects=free_areas, cell_width=math.sqrt(min_playground_area), cell_height=math.sqrt(min_playground_area))
    undistributed_living_buildings = living_buildings[(living_buildings['undistributed_population'] >= min_undistributed_population) |
                                                      (living_buildings['undistributed_proportion'] >= min_undistributed_proportion)]
    new_playgrounds = generate_playgrounds(undistributed_living_buildings=undistributed_living_buildings, free_areas=free_areas, walk_graph=walk_graph,
                                           playground_area_per_people=playground_area_per_people)
    new_playgrounds = GeoDataFrame(data={'area': round(new_playgrounds.area, 2), 'capacity': round(new_playgrounds.area, 2) / playground_area_per_people}, geometry=new_playgrounds,
                                   crs=utm_crs)
    playgrounds = pd.concat([playgrounds, new_playgrounds], ignore_index=True)
    playgrounds['living_buildings_neighbours'] = [[] for _ in range(len(playgrounds))]

    distribute_people_to_playgrounds(living_buildings=living_buildings, playgrounds=playgrounds, walk_graph=walk_graph, max_meters_to_playground=max_meters_to_playground)
    calculate_playgrounds_fullness(playgrounds=playgrounds, living_buildings=living_buildings)
    draw_map(filename=output_data_folder / 'map_after.html', blocks=blocks, playgrounds=playgrounds, living_buildings=living_buildings, gdf_walk_graph=gdf_walk_graph,
             gdf_drive_graph=gdf_drive_graph, physical_objects=physical_objects, free_areas=free_areas, new_playgrounds=new_playgrounds)
    save_output_data_after(folder=output_data_folder, playgrounds=playgrounds, living_buildings=living_buildings, gdf_drive_graph=gdf_drive_graph, new_playgrounds=new_playgrounds)

    statistics['Количество площадок'].append(len(playgrounds))
    statistics['Общая площадь м²'].append(sum(playgrounds['area']))
    statistics['Нераспределенных жителей'].append(round(sum(living_buildings['undistributed_population'])))
    show_statistics(statistics=statistics)

    save_and_show_graphics(folder=output_data_folder, undistributed_proportion_before=undistributed_proportion_before, fullness_before=fullness_before,
                           undistributed_proportion_after=living_buildings['undistributed_proportion'], fullness_after=playgrounds['fullness'])


main()
