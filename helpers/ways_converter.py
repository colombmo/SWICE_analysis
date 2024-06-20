# -*- coding: utf-8 -*-
'''
# From https://gist.github.com/jorgehatccrma/799ad3411293fd76add6a3230ee65640
'''
import logging
from collections import defaultdict

import overpy
from shapely.geometry.polygon import Polygon
from shapely.geometry.multipolygon import MultiPolygon
import geopandas as gpd


def ways2poly(ways):
    """
    Given an iterable of `ways`, combined them into one or more polygons.
    
    Args:
        ways: iterable of `overpy.Way` that form the desired polygon:w
        
    Return:
        polys, incomplete: `polys` is a list of list of  (long, lat) coords describing 
            valid (i.e. closed) polygons; `incomplete` is a list of list of (long, lat) 
            coords describing "incomplete polygons" (i.e. LineString)
    """
    w = set(ways)
    polys = []
    incomplete = []
    current = None
    while True:
        if not current:
            if len(w) > 0:
                current = w.pop().nodes
            else:
                break
        if current[0].id == current[-1].id:
            polys.append(current)
            current = None
            continue
        else:
            if len(w) < 1:
                incomplete.append(current)
                break
            to_remove = set()
            for n in w:
                if n.nodes[0].id == current[-1].id:
                    current += n.nodes
                elif n.nodes[0].id == current[0].id:
                    current = list(reversed(n.nodes)) + current
                elif n.nodes[-1].id == current[0].id:
                    current = n.nodes + current
                elif n.nodes[-1].id == current[-1].id:
                    current += list(reversed(n.nodes))
                else:
                    continue
                to_remove.add(n)
            if len(to_remove) == 0:
                incomplete.append(current)
                current = None
                continue
            w -= to_remove

    return polys, incomplete


if __name__ == "__main__":

    queries = dict()
    queries['PF'] = '''
        (rel[boundary="administrative"]["name:en"="French Polynesia"][admin_level="3"];>;);
        way(r);(._;>;);out;
    '''
    queries['PR'] = '''
        (rel[boundary="administrative"][name="Puerto Rico"][admin_level="4"];>;);
        way(r);(._;>;);out;
    '''

    # Use overpass to get boundary data (ways) from OSM
    api = overpy.Overpass()
    ways = defaultdict(list)
    for c in queries:
        print('Getting data for {}'.format(c))
        ways[c] = api.query(queries[c])
        print('next')
    print('done')

    # build polygons from the collected data
    boundaries = defaultdict(list)
    for k in ways:
        polys, incmp = ways2poly(ways[k].get_ways())
        boundaries[k] = {'polygons': polys, 'incomplete': incmp}

        if len(polys) > 0 and len(incmp) == 0:
            outcome = 'OK'
        else:
            outcome = 'ERROR'

        print("{}: {:>2} polygons, {:>2} incomplete ({})".format(
            k, len(polys), len(incmp), outcome))

        # I only care about complete polygons, but you could process incomplete
        # ones as (Multi)LineString if needed
        if outcome == 'OK':
            boundaries[k]['shape'] = MultiPolygon(
                [Polygon([(n.lon, n.lat) for n in p]) for p in polys])

    # create GeoDataFrame with the boundaries
    gdf = gpd.GeoDataFrame(
        pd.DataFrame(
            [(k, boundaries[k]['shape']) for k in boundaries
             if 'shape' in boundaries[k].keys()],
            columns=['cc', 'boundary']),
        geometry='boundary')

    # Save a shapefile
    gdf.to_file("missing_boundaries.shp")

    # plot them
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    base = world.plot(facecolor='none', edgecolor='gray')
    gdf.plot(ax=base, cmap='tab20', alpha=0.5)