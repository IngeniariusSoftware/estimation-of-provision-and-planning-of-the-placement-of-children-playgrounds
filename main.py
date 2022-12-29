import math
import logging
import pandas as pd
import geopandas as gpd

from pathlib import Path
from shapely import affinity
from geopandas import GeoDataFrame
from shapely.geometry import Point, Polygon, LineString

block_id_label = 'block_id'
standard_crs = 4326
equal_area_meter_crs = 6933
equal_ratio_meter_crs = 3857


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


def get_square_meters_area(objects: GeoDataFrame) -> list[float]:
    return round(objects.to_crs(crs=equal_area_meter_crs).geometry.area, 2)


def get_geometry_angle(geometry: Polygon):
    simplified_coords = geometry.minimum_rotated_rectangle.boundary.coords
    sides = [LineString(coordinates=[c1, c2]) for c1, c2 in zip(simplified_coords, simplified_coords[1:])]
    longest_side = max(sides, key=lambda x: x.length)
    point1, point2 = longest_side.coords
    return math.degrees(math.atan2(point2[1] - point1[1], point2[0] - point1[0]))


def transform_point_objects_into_square_polygons_with_median_area(blocks: GeoDataFrame, objects: GeoDataFrame) -> GeoDataFrame:
    objects = objects.to_crs(equal_ratio_meter_crs)
    polygons = objects[objects.geometry.geom_type == 'Polygon']
    median_area = polygons.geometry.area.median()
    square_length = math.sqrt(median_area)
    points = objects[objects.geometry.geom_type == 'Point'].copy()
    angles = list(blocks.to_crs(equal_ratio_meter_crs).geometry.apply(func=lambda x: get_geometry_angle(x)))
    points.geometry = points.apply(
        func=lambda x: generate_rotated_square_from_point(point=x.geometry, length=square_length, angle=angles[x[block_id_label]]), axis=1)
    return pd.concat([polygons, points]).set_crs(equal_ratio_meter_crs).to_crs(standard_crs)


def generate_rotated_square_from_point(point: Point, length: float, angle: float) -> Polygon:
    return affinity.rotate(geom=point.buffer(distance=length / 2.0, cap_style=3), angle=angle, origin='centroid')


def remove_contained_points_in_objects_from_geodataframe(data: GeoDataFrame, objects: GeoDataFrame) -> GeoDataFrame:
    polygons = data[data.geometry.geom_type == 'Polygon']
    points = data[data.geometry.geom_type == 'Point']
    contained_points = points[points.within(pd.concat([polygons.geometry, objects.geometry], ignore_index=True).unary_union)]
    return data.drop(contained_points.index, axis=0)


def main():
    blocks = gpd.read_file(filename=Path.cwd() / 'data' / 'test' / 'blocks.geojson').set_crs(crs=standard_crs)
    blocks = blocks.explode(ignore_index=True)
    blocks['id'] = blocks.index

    buildings = gpd.read_file(filename=Path.cwd() / 'data' / 'test' / 'input_buildings.geojson').set_crs(crs=standard_crs)
    buildings = remove_objects_outside_blocks_and_set_block_id(blocks=blocks, objects=buildings)

    playgrounds = gpd.read_file(filename=Path.cwd() / 'data' / 'test' / 'input_playgrounds.geojson').set_crs(crs=standard_crs)
    playgrounds = playgrounds.explode(ignore_index=True)
    playgrounds = remove_contained_points_in_objects_from_geodataframe(data=playgrounds, objects=buildings)
    playgrounds = remove_objects_outside_blocks_and_set_block_id(blocks=blocks, objects=playgrounds)
    playgrounds['area'] = get_square_meters_area(playgrounds)
    playgrounds = transform_point_objects_into_square_polygons_with_median_area(blocks=blocks, objects=playgrounds)
    playgrounds.to_file(filename='result.geojson', driver='GeoJSON')

main()
