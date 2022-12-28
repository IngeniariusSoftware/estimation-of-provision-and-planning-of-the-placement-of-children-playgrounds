import logging
import geopandas as gpd
from pathlib import Path


def remove_objects_outside_blocks(blocks: gpd.GeoDataFrame, objects: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    block_id_label = 'block_id'
    inside_objects = objects.sjoin(df=blocks[['id', 'geometry']].rename(columns={'id': 'id_right'}), how='left')
    if block_id_label in inside_objects.columns:
        logging.warning(f' objects geoDataFrame already contains {block_id_label} column, renamed to {block_id_label}_(2)')
        inside_objects.rename(columns={block_id_label: f'{block_id_label}_(2)'}, inplace=True)
    inside_objects.rename(columns={'id_right': block_id_label}, inplace=True)
    inside_objects.drop(columns=['index_right'], inplace=True)
    inside_objects = inside_objects[inside_objects[block_id_label].notna()]

    logging.info(f' {len(objects) - len(inside_objects)} objects outside the blocks were deleted')
    return inside_objects


def main():
    crs = 4326

    blocks = gpd.read_file(Path.cwd() / 'data' / 'test' / 'blocks.geojson').set_crs(crs)
    buildings = gpd.read_file(filename=Path.cwd() / 'data' / 'test' / 'input_buildings.geojson').set_crs(crs)
    output_buildings = remove_objects_outside_blocks(blocks=blocks, objects=buildings)

   # # playgrounds = gpd.read_file(Path.cwd() / 'data' / 'test' / 'playgrounds.geojson').set_crs(crs)
    result = remove_objects_outside_blocks(blocks=blocks, objects=buildings)
    result.to_file(filename='output_buildings.geojson', driver='GeoJSON')

    # result = playgrounds.sjoin(df=blocks[['id', 'geometry']], how='left')
    # result.rename(columns={'id_right': 'block_id'}, inplace=True)
    # result = result[result['block_id'].notna()]
    #
    #
    # result.to_file(filename='result.geojson', driver='GeoJSON')

main()