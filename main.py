import math
import scipy
import pandana
import logging
import osmnx as ox
import pandas as pd
import networkx as nx
import geopandas as gpd

from pathlib import Path
from pandas import Series
from pandana import Network
from shapely import affinity
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
    inside_objects = inside_objects[inside_objects[block_id_label].notna()]
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
    polygons = objects[objects.geometry.geom_type == 'Polygon']
    median_area = polygons.geometry.area.median()
    square_length = math.sqrt(median_area)
    points = objects[objects.geometry.geom_type == 'Point'].copy()
    angles = list(blocks.geometry.apply(func=lambda x: get_geometry_angle(x)))
    points.geometry = points.apply(
        func=lambda x: generate_rotated_square_from_point(point=x.geometry, length=square_length, angle=angles[x[block_id_label]]), axis=1)
    return pd.concat(objs=[polygons, points])


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


def get_walk_graph(blocks: GeoDataFrame, to_crs=None) -> Network:
    graph = ox.graph_from_polygon(polygon=blocks.to_crs(crs=standard_crs).geometry.unary_union.convex_hull, network_type='walk', simplify=True)
    nodes, edges = ox.graph_to_gdfs(G=ox.project_graph(graph, to_crs=to_crs), nodes=True, edges=True)
    edges = edges.reset_index()
    return Network(nodes['x'], nodes['y'], edges['u'], edges['v'], edges[['length']])


def get_utm_crs(geometry) -> str:
    mean_longitude = geometry.representative_point().x.mean()
    utm_zone = int(math.floor((mean_longitude + 180.0) / 6.0) + 1.0)
    return f'+proj=utm +zone={utm_zone} +ellps=WGS84 +datum=WGS84 +units=m +no_defs'


def get_isochrone_walk_graph(start_point: Point, meters: int, walk_graph: MultiDiGraph) -> MultiDiGraph:
    return nx.ego_graph(G=walk_graph, n=ox.nearest_nodes(G=walk_graph, X=start_point.x, Y=start_point.y), radius=meters, distance='length')


def graph_to_polygon(graph: MultiDiGraph) -> Polygon:
    node_points = [Point((data['x'], data['y'])) for node, data in graph.nodes(data=True)]
    return GeoSeries(node_points).unary_union.convex_hull


def polygon_to_x_y_coords(polygon: Polygon) -> [float, float]:
    centroid = polygon.centroid
    return centroid.x, centroid.y


def geodataframe_to_x_y_coords(objects: GeoDataFrame) -> [Series, Series]:
    x_coords, y_coords = zip(*objects.geometry.apply(func=lambda x: polygon_to_x_y_coords(x)))
    return Series(x_coords), Series(y_coords)


def get_gravity_for_building_and_playground(area_playground: float, distance: float) -> float:
    return area_playground / (distance + 50.0) ** 2.0


def set_nearest_nodes_to_objects(objects: GeoDataFrame, walk_graph: Network) -> None:
    x_coords, y_coords = geodataframe_to_x_y_coords(objects=objects)
    objects[node_id_label] = walk_graph.get_node_ids(x_col=x_coords, y_col=y_coords)
    # objects[node_distance_label] =


def distribute_people_to_playgrounds(blocks: GeoDataFrame, living_buildings: GeoDataFrame, playgrounds: GeoDataFrame, walk_graph: Network) -> None:
    set_nearest_nodes_to_objects(playgrounds, walk_graph)
    set_nearest_nodes_to_objects(living_buildings, walk_graph)
    max_meters_to_playground = 500

    print(walk_graph.nodes_in_range([living_buildings.iloc[0][node_id_label]], radius=500))

    # for _, block in blocks.iterrows():
    #     block_living_buildings = living_buildings[living_buildings[block_id_label] == block['id']]
    #
    #     walk_graph.nodes_in_range()
    #     for _, living_building in block_living_buildings.iterrows():
    #
    #         graph = get_walk_isochrone_graph(start_point=living_building.geometry.centroid, meters=max_meters_to_playground, walk_graph=walk_graph)
    #         isochrone = graph_to_polygon(graph)
    #         available_playgrounds = playgrounds[playgrounds.intersects(other=isochrone)]
    #         for _, playground in available_playgrounds.iterrows():
    #
    #         # playgrounds_gravity = available_playgrounds.apply(func=lambda x: get_gravity_for_building_and_playground(x))

def main():
    test_folder = Path() / 'data' / 'test'

    blocks = gpd.read_file(filename=test_folder / 'blocks.geojson').set_crs(crs=standard_crs)
    utm_crs = get_utm_crs(blocks.geometry)

    blocks = blocks.explode(ignore_index=True).to_crs(crs=utm_crs)
    blocks['id'] = blocks.index

    buildings = gpd.read_file(filename=test_folder / 'input_buildings.geojson').set_crs(crs=standard_crs).to_crs(crs=utm_crs)
    buildings = remove_objects_outside_blocks_and_set_block_id(blocks=blocks, objects=buildings)

    living_buildings = buildings[buildings['population'] > 0]

    playgrounds = gpd.read_file(filename=test_folder / 'input_playgrounds.geojson').set_crs(crs=standard_crs).to_crs(crs=utm_crs)
    playgrounds = playgrounds.explode(ignore_index=True)
    playgrounds = remove_objects_outside_blocks_and_set_block_id(blocks=blocks, objects=playgrounds)
    playgrounds = remove_contained_points_in_objects_from_geodataframe(data=playgrounds, blocks=blocks, objects=buildings)
    playgrounds['area'] = round(playgrounds.geometry.area, 2)
    playgrounds['population'] = [0] * len(playgrounds)
    playgrounds = transform_point_objects_into_square_polygons_with_median_area(blocks=blocks, objects=playgrounds)

    nodes_file_path, edges_file_path = test_folder / 'nodes.csv', test_folder / 'edges.csv'
    if nodes_file_path.exists() and edges_file_path.exists():
        nodes, edges = pd.read_csv(nodes_file_path, index_col=0), pd.read_csv(edges_file_path, index_col=0)
        walk_graph = pandana.Network(nodes['x'], nodes['y'], edges['from'], edges['to'], GeoDataFrame({'distance': edges['length']}))
    else:
        walk_graph = get_walk_graph(blocks=blocks, to_crs=utm_crs)
        walk_graph.nodes_df.to_csv(nodes_file_path), walk_graph.edges_df.to_csv(edges_file_path)

    distribute_people_to_playgrounds(blocks, living_buildings, playgrounds, walk_graph)

    # walk_graph.get_node_ids(x_col=playgrounds, y_col=)
    # graph = get_isochrone_walk_graph(start_point=living_buildings.iloc[0].geometry.centroid, meters=500, walk_graph=walk_graph)
    # polygon = graph_to_polygon(graph=graph)
    # available_playgrounds = playgrounds[playgrounds.intersects(other=polygon)]
    # for _, playground in available_playgrounds.iterrows():
    #     start_point = living_buildings.iloc[0].geometry.centroid
    #     start_node = ox.nearest_nodes(G=walk_graph, X=start_point.x, Y=start_point.y)
    #
    #     end_point = playground.geometry.centroid
    #     end_node = ox.nearest_nodes(G=walk_graph, X=end_point.x, Y=end_point.y)
    #
    #     distance = ox.shortest_path(G=walk_graph, orig=start_node, dest=end_node, weight='length')
    #     print(distance)
    

main()
