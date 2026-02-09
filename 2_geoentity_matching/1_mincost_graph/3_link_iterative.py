import os 
import networkx as nx
import numpy as np
from geopy.distance import great_circle
import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx
from shapely.geometry import Point
from sklearn.cluster import DBSCAN
import argparse
import json
import pandas as pd 
import glob 
import pdb 


HALF_DEMAND = True
MAX_ITER = 10  # Maximum number of refinement iterations
CENTER_TOLERANCE = 0.001  # Threshold for center stability (in degrees)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Script to merge map data.')

    parser.add_argument('--input_folder', type=str, default='outputs_paper/ngmdb_nickel_0114_geonames_plotfiltered_step1/')
    parser.add_argument('--output_folder', type=str, default='outputs_paper/ngmdb_nickel_0114_geonames_plotfiltered_scalecleaned_step2/')

    

    parser.add_argument('--metadata_path', type = str, default = '') # '../data/ngmdb/nickel_meta_combine.csv'
    parser.add_argument('--map_name_col_name', type=str, default='cog_id') # filename
    parser.add_argument('--scale_col_name', type=str, default='scale_clean') # scale

    return parser.parse_args()



def plot_dbscan_clusters(gdf, largest_cluster_id, output_path='dbscan_clusters.png'):
    """
    Plot DBSCAN clusters on a geographic basemap and save the figure as a PNG.

    Parameters:
        gdf (GeoDataFrame): GeoDataFrame with 'cluster' labels and lat/lon points in EPSG:4326.
        largest_cluster_id (int): ID of the largest cluster to highlight.
        output_path (str): File path to save the PNG plot.
    """
    # Project to Web Mercator (EPSG:3857) for basemap compatibility
    gdf_proj = gdf.to_crs(epsg=3857)

    # Separate noise and clusters
    noise = gdf_proj[gdf_proj['cluster'] == -1]
    clustered = gdf_proj[gdf_proj['cluster'] != -1]

    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot noise
    if not noise.empty:
        noise.plot(ax=ax, color='lightgray', markersize=10, label='Noise')

    # Plot clusters
    for cluster_id in clustered['cluster'].unique():
        cluster_points = clustered[clustered['cluster'] == cluster_id]
        if cluster_id == largest_cluster_id:
            cluster_points.plot(ax=ax, color='blue', markersize=20, label='Largest Cluster')
        else:
            cluster_points.plot(ax=ax, markersize=10, label=f'Cluster {cluster_id}')

    # Add basemap
    # ctx.add_basemap(ax, source=ctx.providers.Stamen.TonerLite, crs=gdf_proj.crs)
    # ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=5)

    # # Set extent to continental US
    # ax.set_xlim(-14000000, -7000000)
    # ax.set_ylim(3000000, 6500000)

    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, crs=gdf_proj.crs, zoom=5)

    # After plotting the points...

    # # Zoom to cluster extent (roughly state-sized)
    # xmin, ymin, xmax, ymax = gdf_proj.total_bounds
    # buffer = 100000  # 100 km padding
    # ax.set_xlim(xmin - buffer, xmax + buffer)
    # ax.set_ylim(ymin - buffer, ymax + buffer)

    # # Add basemap
    # ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, crs=gdf_proj.crs, zoom=7)




    ax.set_title("DBSCAN Clusters on Basemap")
    ax.legend()
    ax.axis('off')

    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Cluster plot saved to {output_path}")



def get_center(latitudes, longitudes, eps=1.0, min_samples=5, plot_path=None):
    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(geometry=[Point(lon, lat) for lon, lat in zip(longitudes, latitudes)], crs='EPSG:4326')

    # Perform DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')

    
    clusters = dbscan.fit_predict(np.vstack([longitudes, latitudes]).T)



    gdf['cluster'] = clusters

    print('clusters', clusters)
    # Filter out noise points
    valid_clusters = gdf[gdf['cluster'] != -1]
    print('valid_clusters', valid_clusters)

    if valid_clusters.empty:
        return None, None, None

    # Find the largest cluster
    cluster_counts = valid_clusters['cluster'].value_counts()
    largest_cluster_id = cluster_counts.idxmax()
    largest_cluster = valid_clusters[valid_clusters['cluster'] == largest_cluster_id]


    if plot_path:
        plot_dbscan_clusters(gdf, largest_cluster_id, output_path=plot_path)

    # Compute centroid
    centroid_x = largest_cluster.geometry.x.mean()
    centroid_y = largest_cluster.geometry.y.mean()

    return largest_cluster, centroid_y, centroid_x  # Return lat, lon

def build_graph(place_names, reference_point):
    G = nx.DiGraph()
    source, sink = "source", "sink"

    if HALF_DEMAND:
        node_demand = len(place_names) // 2
    else:
        node_demand = len(place_names)

    G.add_node(source, demand=-node_demand)
    G.add_node(sink, demand=node_demand)

    place_id_map = {place: i for i, place in enumerate(place_names.keys())}
    id_place_map = {i: place for place, i in place_id_map.items()}

    geo_nodes = {}

    for place_id in place_id_map.values():
        G.add_edge(source, place_id, capacity=1, weight=0)

    for place, entry in place_names.items():
        place_id = place_id_map[place]

        for candidate in entry['candidate_matches']:
            geo_id = candidate['index']
            lat = candidate['lat']
            lon = candidate['lon']
            similarity = candidate['score']

            geo_nodes[geo_id] = geo_id
            distance_cost = int(great_circle((lat, lon), reference_point).km)
            similarity_cost = int(100 * (1 - similarity))
            total_cost = distance_cost // 100 + similarity_cost
            G.add_edge(place_id, geo_id, capacity=1, weight=total_cost)

    for geo in geo_nodes:
        G.add_edge(geo, sink, capacity=1, weight=0)

    return G, place_id_map, id_place_map, geo_nodes


def find_flow_solution(input_folder, output_folder, map_name, eps=1.0, min_samples=5, if_plot_cluster=False, if_center_from_all_cands = False):
    with open(f'{input_folder}/{map_name}_matched_sites.json', 'r') as f:
        raw_place_names = json.load(f)

    if len(raw_place_names) == 0:
        print(f"-- No place name found for {map_name} -- ")
        return None, None 
    elif len(raw_place_names) < 3:
        print(f"-- Not enough place names (<3) found for {map_name} -- ")
        return None, None 

    # Filter out empty matches
    place_names = {key: value for key, value in raw_place_names.items() if len(value['candidate_matches']) != 0}

    if if_center_from_all_cands:
        # Initial approximate center using all candidate geocoordinates
        latitudes, longitudes = [], []
        for entry in place_names.values():
            candidates = entry['candidate_matches']
            for candidate in candidates:
                lat = candidate['lat']
                lon = candidate['lon']
                latitudes.append(lat)
                longitudes.append(lon)
    else:
        # Initial approximate center using one candidate from one place name
        latitudes, longitudes = [], []
        for entry in place_names.values():
            candidates = entry['candidate_matches']
            candidate = candidates[0] # choose the first one 
            lat = candidate['lat']
            lon = candidate['lon']
            latitudes.append(lat)
            longitudes.append(lon)


    # Get initial center
    if if_plot_cluster:
        os.makedirs(os.path.join(output_folder, map_name), exist_ok=True)
        _, centroid_lat, centroid_lon = get_center(latitudes, longitudes, eps=eps, min_samples=min_samples, plot_path=os.path.join(output_folder, map_name, 'dbscan_iter0.png'))
    else:
        _, centroid_lat, centroid_lon = get_center(latitudes, longitudes, eps=eps, min_samples=min_samples)

    if centroid_lat is None or centroid_lon is None:
        print(f"Initial clustering failed. Using centroid for map {map_name}.")
        centroid_lat = np.mean(latitudes)
        centroid_lon = np.mean(longitudes)

    reference_point = (centroid_lat, centroid_lon)
    print(f"Initial reference point: {reference_point}")

    for iteration in range(MAX_ITER):
        print(f"=== Iteration {iteration} ===")
        G, place_id_map, id_place_map, geo_nodes = build_graph(place_names, reference_point)
        flow_cost, flow_dict = nx.network_simplex(G)

        # Extract assignments
        assignments = {}
        for place_id in place_id_map.values():
            for geo in geo_nodes:
                if flow_dict[place_id].get(geo, 0) > 0:
                    place_name = id_place_map[place_id]
                    assigned_geo = next(
                        candidate for candidate in place_names[place_name]['candidate_matches'] if candidate['index'] == geo
                    )

                    assignment = {
                        "place_name": place_name,
                        "geo_id": geo,
                        "geo_name": assigned_geo['name'],
                        "lat": assigned_geo['lat'],
                        "lon": assigned_geo['lon'],
                        "feature_class": assigned_geo.get('feature_class', ''),
                        "vertices_list": list(place_names[place_name]["vertices_list"])
                    }

                    # Optional fields depending on dataset (only add if they exist)
                    if 'feature_code' in assigned_geo:
                        assignment['feature_code'] = assigned_geo['feature_code']

                    if 'state_name' in assigned_geo:
                        assignment['state_name'] = assigned_geo['state_name']

                    if 'county_name' in assigned_geo:
                        assignment['county_name'] = assigned_geo['county_name']


                    assignments[place_id] = assignment

        # Compute new center
        assigned_lats = [info['lat'] for info in assignments.values()]
        assigned_lons = [info['lon'] for info in assignments.values()]

        if if_plot_cluster:
            os.makedirs(os.path.join(output_folder, map_name), exist_ok=True)
            _, new_centroid_lat, new_centroid_lon = get_center(assigned_lats, assigned_lons, eps=eps, min_samples=min_samples, plot_path=os.path.join(output_folder, map_name, f'dbscan_iter{iteration + 1}.png'))
        else:
            _, new_centroid_lat, new_centroid_lon = get_center(assigned_lats, assigned_lons, eps=eps, min_samples=min_samples)

        if new_centroid_lat is None or new_centroid_lon is None:
            print("No valid cluster found. Stopping iterations.")
            break

        new_reference_point = (new_centroid_lat, new_centroid_lon)
        lat_diff = abs(new_reference_point[0] - reference_point[0])
        lon_diff = abs(new_reference_point[1] - reference_point[1])
        print(f"New reference point: {new_reference_point} (lat_diff={lat_diff}, lon_diff={lon_diff})")

        if lat_diff < CENTER_TOLERANCE and lon_diff < CENTER_TOLERANCE:
            print("Reference point stabilized. Stopping iterations.")
            break

        reference_point = new_reference_point

    with open(f"{output_folder}/{map_name}_graph_output.json", "w") as f:
        json.dump(assignments, f, indent=4)

    return reference_point, assignments



def filter_assignments(output_folder, reference_point, assignments, distance_threshold = 10):
    # Filter assignments to only those within 10 km of the reference point
    filtered_assignments = {
        place_id: data
        for place_id, data in assignments.items()
        if great_circle(reference_point, (data["lat"], data["lon"])).kilometers <= distance_threshold
    }




    with open(f"{output_folder}/{map_name}_graph_output_scalecleaned.json", "w") as f:
        json.dump(filtered_assignments, f, indent=4)

    return filtered_assignments


def determine_distance_threshold(map_name, metadata_path, map_name_col_name, scale_col_name):
    meta_df = pd.read_csv(metadata_path, dtype=str) # , dtype={'id': str})
    
    map_scale = meta_df[meta_df[map_name_col_name] == map_name][scale_col_name].values[0]


    scale_to_distance_dict = {
      "10000": 7.0,
      "12000": 7.1,
      "20000": 9.9,
      "24000": 17.5,
      "25000": 18.5,
      "31680": 23.5,
      "48000": 36.4,
      "50000": 36.4,
      "62500": 36.4,
      "63360": 47.0,
      "96000": 71.0,
      "100000": 77.8,
      "125000": 84.9,
      "126720": 102.3,
      "128000": 100.0,
      "250000": 188.7,
      "500000": 377.4,
      "1000000": 538.5,
    }

    if map_scale not in scale_to_distance_dict: 
        pdb.set_trace()
        return -1 

    ret_distance = scale_to_distance_dict[map_scale] 
    return ret_distance



if __name__ == '__main__':
    args = parse_arguments()
    print(args)

    os.makedirs(args.output_folder, exist_ok=True)


    # map_list = ['2516015', '2554002','4472002','11640099']
    # # map_list = ['11640099']

    # map_list = [
    #     "6e0627f218de28de12de19da46da46da659a541e54195a19485a4d1826cb20e3", 
    #     "273333b333b333b327b32f132b931bb30db38c188c3a2f3a3fb380318e13b012", 
    #     "643a1e320732133206b30eb215b20d3209b235b22cf624762932293223f222f3", 
    #     "4988298f0e3126b331233893061304a302570643032b0a3303830d0304110a58", 
    #     "00000c0f3b1f3e273e0d3e0f3e373e273f27362f3f2b3f0737271727000b0008", 
    #     "559844da265a1b1a209b2c9b449a20da249806da30da24dc265a827af238433b", 
    #     "687272e4366438e437645d745fe46be46de44f6444e446e54865567076f40716", 
    #     "471d57983cf867721373233246ba16fa3718339a169a4f9a26de20dc88d84c79", 
    #     "7c1b501b659b659b64bb156b1a7b5adb136b065b032b792f7ceb1ceb129d21a0", 
    #     "a4023a721af22bf333ba2a3a3a1e24362e6306a322462c6e364c231a2d9aa044"
    # ]


    map_list = glob.glob(args.input_folder + '/*_matched_sites.json')
    map_list = sorted([os.path.basename(a).split('_matched_sites')[0] for a in map_list])

    # map_list = ["606066c2529ee32f6b236321db295aa112c01ea55b0552a377235f334f4c0846"]


    print(f"number of maps to process: {len(map_list)}")



    for idx  in range(0, len(map_list)):
        map_name = map_list[idx]

        print(f"Processing {idx}th: {map_name}...")
        # find_flow_solution(args.input_folder, args.output_folder, map_name,  eps = 1.0, min_samples = 5, if_plot_cluster = True)

        reference_point, assignments = find_flow_solution(args.input_folder, args.output_folder, map_name,  eps = 0.2, min_samples = 5, if_plot_cluster = True, if_center_from_all_cands= False)
        
        if reference_point is None: # failed because no place name found 
            continue 
        
        if os.path.isfile(args.metadata_path):
            print(f'Using the scale info in metadata file {args.metadata_path} to refine linking output' )

            distance_threshold_km = determine_distance_threshold(map_name, args.metadata_path, map_name_col_name = args.map_name_col_name, scale_col_name = args.scale_col_name)
            # distance_threshold_km = 50
            if distance_threshold_km == -1:
                distance_threshold_km = 50
        else:
            distance_threshold_km = 50 
            
        print(distance_threshold_km)
        filtered_assignments = filter_assignments(args.output_folder, reference_point, assignments, distance_threshold = distance_threshold_km ) 

        print('\n\n\n')





# python3 3_link_iterative.py --input_folder='outputs_paper/ngmdb_nickel_0114_geonames_plotfiltered_step1/' --output_folder='outputs_paper/ngmdb_nickel_0114_geonames_plotfiltered_scalecleaned_step2/' --metadata_path='../data/ngmdb/nickel_meta_combine.csv'

#############################################################################3

# python3 3_link_iterative.py --input_folder='outputs/geoclek_step1_0930_kg' --output_folder='outputs/geoclek_step2_0930_kg' 

# python3 3_link_iterative.py --input_folder='outputs/geoclek_step1_0927' --output_folder='outputs/geoclek_step2_0927' 

# python3 3_link_iterative.py --input_folder='outputs/rio_geonames_08010_step1' --output_folder='outputs/rio_geonames_step2_08010' 


#  python3 2_link_iterative.py --input_folder='outputs/ngmdb_nickel_0802_gnis_step1' --output_folder='outputs/ngmdb_nickel_0802_gnis_step2' --metadata_path='../data/ngmdb/nickel_meta_combine.csv' 


# python3 2_link_iterative.py --input_folder='outputs/ngmdb_nickel_0706' --output_folder='outputs/ngmdb_nickel_0706_graph' --metadata_path='../data/ngmdb/nickel_meta_combine.csv'

# python3 2_link_iterative.py --input_folder='outputs/rumsey_0427' --output_folder='outputs/rumsey_0427_iter'


# python3 2_link_iterative.py --input_folder='outputs/ngmdb_0531' --output_folder='outputs/ngmdb_0531'

# python3 2_link_iterative.py --input_folder='outputs/ngmdb_0531' --output_folder='outputs/ngmdb_0608_quarter_demand'

# python3 2_link_iterative.py --input_folder='outputs/ngmdb_0531' --output_folder='outputs/ngmdb_0608_candidates_geonames'
