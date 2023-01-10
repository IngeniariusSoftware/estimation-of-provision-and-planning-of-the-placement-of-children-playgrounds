### Description 

1. This program first calculates the provision of residential buildings with playgrounds according to the established standard (0.5 m²).
   
2. Then, using polygons of free space, it places new playgrounds for houses with undistributed population, recalculates indicators and gives the result.

### Input

1. Calculation playgrounds buildings coverage and population distribution

    > Some of the playgrouns can be marked with points. In this case, they are restored to square polygons with the average area of the remaining polygon playgrounds.

    Name | Geometry | Fields
    --- | --- | ---
    Blocks | `Polygon` | -
    Playgrounds | `Polygon\|Point` | -
    Buildings | `Polygon\|Point` | populaiton: int

2. Generating new playgrounds
   
    Same data from step 1.

    Name | Geometry | Fields
    --- | --- | ---
    Physical objects \| Free areas| `Polygon`  | -


### Method and intermediate data

1. Calculation playgrounds buildings coverage and population distribution

    To calculate the playgrounds coverage of a residential building, an isochron is used to obtain the nearest playgrounds. The distribution of population is calculated using a gravity model: $G_{playground}=S_{playground} / (R_{playground} + 50_{meters})^2$

    Name | Geometry\Type | Fields
    --- | --- | ---
    Walk graph | `MultiDiGraph` | -
    Playgrounds | `Polygon` | area: float, living_buildings_neighbours: list[id: int, distribution: float]
    Living buildings | `Polygon\|Point` | population: int, gravity: list[float], isochrone_graph: MultiDiGraph

2. Generating new playgrounds

    To obtain a free area, all buffer zones around all physical objects and drive graph are cut out.

    To generate new playgrounds, the free area is divided into polygons with an area equal to the minimum area of the playgrounds. Then, for each high undistributed living buildings, the optimal placement distance of the new playground is calculated from the above formula and the nearest polygon at this distance is selected. Then the other polygons closest in the radius from this starting polygon get to the required area.

    Name | Geometry\Type | Fields
    --- | --- | ---
    Drive graph | `GeoDataFrame` | -
    Free areas | `Polygon` | -


### Output

1. Calculation playgrounds buildings coverage and population distribution

    Name | Geometry | Fields
    --- | --- | ---
    Blocks | `Polygon` | -
    Playgrounds | `Polygon` | area: float, fullness: float
    Buildings | `Polygon\|Point` | population: int, undistributed_proportion: float


2. Generating new playgrounds


    Name | Geometry | Fields
    --- | --- | ---
    Blocks | `Polygon` | -
    Playgrounds | `Polygon` | area: float, fullness: float
    Buildings | `Polygon\|Point` | population: int, undistributed_proportion; float
    New playgrounds | `Polygon` | area: float, fullness: float



### Screenshots

#### 1. Screenshot of first step result

![screenshot_before](data/screenshots/screenshot_before.png)

#### 2. Screenshot after generating new playgrounds

![screenshot_after](data/screenshots/screenshot_after.png)

#### 3. Comparative histogram of the undistributed population before and after

![screenshot_after](data/screenshots/undistributed_proportion.png)

### Result demonstration

1. [First step map](https://github.com/IngeniariusSoftware/estimation-of-provision-and-planning-of-the-placement-of-children-playgrounds/releases/download/1.0.0/map_before.html)

2. [Map after generating new playgorouns](https://github.com/IngeniariusSoftware/estimation-of-provision-and-planning-of-the-placement-of-children-playgrounds/releases/download/1.0.0/map_after.html)