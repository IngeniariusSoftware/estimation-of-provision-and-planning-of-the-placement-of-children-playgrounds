import main
import geopandas as gpd
from pathlib import Path


def test_remove_buildings_outside_blocks():
    crs = 4326
    blocks = gpd.read_file(filename=Path.cwd() / 'data' / 'test' / 'blocks.geojson').set_crs(crs=crs)
    input_buildings = gpd.read_file(filename=Path.cwd() / 'data' / 'test' / 'input_buildings.geojson').set_crs(crs=crs)
    output_buildings = main.remove_objects_outside_blocks(blocks=blocks, objects=input_buildings)
    output_buildings.reset_index(drop=True, inplace=True)
    expected_buildings = gpd.read_file(filename=Path.cwd() / 'data' / 'test' / 'output_buildings.geojson').set_crs(crs=crs)
    assert all(output_buildings.sort_index().sort_index(axis=1) == expected_buildings.sort_index().sort_index(axis=1))
