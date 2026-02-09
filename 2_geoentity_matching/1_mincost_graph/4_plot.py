import os
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import contextily as ctx
import argparse
import matplotlib.patches as mpatches
import json 
import glob
import pdb 

def parse_arguments():
    parser = argparse.ArgumentParser(description='Script to merge map data.')

    parser.add_argument('--input_candidates_folder', type=str, default = 'outputs/rumsey_output') # stores the *_matched_sites.json
    parser.add_argument('--input_linking_folder', type=str, default = 'outputs/rumsey_output') # stores the *_graph_output.json

    parser.add_argument('--output_png_folder', type=str, default = 'outputs/rumsey_output') # output png visualization files
    parser.add_argument('--method', type=str, default = 'graph', choices = ['graph','ilp']) # output png visualization files



    parser.add_argument('--plot_candidate', action='store_true') 
    parser.add_argument('--plot_target', action='store_true') 
    parser.add_argument('--with_feature_class', action='store_true') 
    parser.add_argument('--with_text_label', action='store_true') 
    parser.add_argument('--filtered_suffix', action='store_true') 


    args = parser.parse_args()

    if not os.path.isdir(args.output_png_folder):
        os.makedirs(args.output_png_folder)

    return args




def plot_candidate_locations(args, map_name, if_show_place_name=False):
    with open(f'{args.input_candidates_folder}/{map_name}_matched_sites.json', 'r') as f:
        data = json.load(f)

    # Build DataFrame
    df = []
    for place_name, entry in data.items():
        key_name = 'candidate_matches'
        print(f"Number of candidates for {place_name}: {len(entry[key_name])}")
        for value in entry[key_name]:
            lat = value['lat']
            lon = value['lon']
            df.append([place_name, Point(lon, lat)])


    # --- Handle empty data ---
    if not df:
        print(f"No valid candidate coordinates found for {map_name}. Skipping plot.")
        return
    # --------------------------

    gdf = gpd.GeoDataFrame(
        df, columns=["place_name", "geometry"], crs="EPSG:4326"
    ).to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot all points in a single style
    gdf.plot(
        ax=ax,
        color="red",
        markersize=100,
        alpha=0.7,
        edgecolor="black"
    )

    # Add text labels if requested
    if if_show_place_name:
        for idx, row in gdf.iterrows():
            ax.text(
                row.geometry.x + 5000,  # small offset to the right
                row.geometry.y,
                row.place_name,
                fontsize=8,
                ha="left",
                va="center",
                bbox=dict(facecolor="white", alpha=0.5, edgecolor="none", pad=1)
            )

    # --- Ensure extent covers at least 500 km x 500 km ---
    xmin, ymin, xmax, ymax = gdf.total_bounds
    width = xmax - xmin
    height = ymax - ymin

    # Convert 500 km to meters in EPSG:3857
    min_extent_m = 600_000  

    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2

    half_width = max(width / 2, min_extent_m / 2)
    half_height = max(height / 2, min_extent_m / 2)

    ax.set_xlim(cx - half_width, cx + half_width)
    ax.set_ylim(cy - half_height, cy + half_height)
    # -----------------------------------------------------


    # Basemap
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.title("Distribution of candidates")

    plt.savefig(f"{args.output_png_folder}/{map_name}_candidates.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_candidate_locations_with_feature_class(args, map_name, if_color_by_feature_class = True, if_show_place_name = False):
    with open(f'{args.input_candidates_folder}/{map_name}_matched_sites.json', 'r') as f:
        data = json.load(f)

    # Build DataFrame
    df = []
    for place_name, entry in data.items():
        # pdb.set_trace()
        # if "candidate_matches" in entry:
        #     key_name = "candidate_matches"
        # elif "geoname_matches" in entry:
        #     key_name = "geoname_matches"
        # else:
        #     raise NotImplementedError

        # print(key_name)
        key_name = 'candidate_matches'
        print(f"Number of candidates for {place_name}: {len(entry[key_name])}")
        for value in entry[key_name]:
            lat = value['lat']
            lon = value['lon']

            match_site_feature_class = value['feature_class']
            match_site_feature_code = value['feature_code']
            df.append([place_name, Point(lon, lat), match_site_feature_class, match_site_feature_code])

    gdf = gpd.GeoDataFrame(
        df, columns=["place_name", "geometry", "feature_class", "feature_code"], crs="EPSG:4326"
    ).to_crs(epsg=3857)

    # Map feature class to description
    feature_class_descriptions = {
        "A": "country, state, region,...",
        "H": "stream, lake,...",
        "L": "parks, area,...",
        "P": "city, village,...",
        "R": "road, railroad",
        "S": "spot, building, farm",
        "T": "mountain, hill, rock,...",
        "U": "undersea",
        "V": "forest, heath,..."
    }

    fig, ax = plt.subplots(figsize=(10, 8))

    if if_color_by_feature_class:
        # Per-class coloring
        unique_classes = gdf["feature_class"].unique()
        color_map = plt.get_cmap("tab10")
        class_colors = {cls: color_map(i % 10) for i, cls in enumerate(unique_classes)}

        for cls, color in class_colors.items():
            subset = gdf[gdf["feature_class"] == cls]
            subset.plot(
                ax=ax,
                color=color,
                markersize=100,
                alpha=0.7,
                edgecolor="black",
                label=cls
            )

        # Legend
        handles = []
        for cls, color in class_colors.items():
            description = feature_class_descriptions.get(cls, "unknown")
            label = f"{cls}: {description}"
            handles.append(mpatches.Patch(color=color, label=label))
        ax.legend(handles=handles, title="Feature Class", loc='best', fontsize='small')

    else:
        # Single color for all
        gdf.plot(
            ax=ax,
            color="red",
            markersize=100,
            alpha=0.7,
            edgecolor="black"
        )

    # Add text labels if requested
    if if_show_place_name:
        for idx, row in gdf.iterrows():
            ax.text(
                row.geometry.x + 5000,  # small offset to the right
                row.geometry.y,
                row.place_name,
                fontsize=8,
                ha="left",
                va="center",
                bbox=dict(facecolor="white", alpha=0.5, edgecolor="none", pad=1)
            )

    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.title("Distribution of candidates")

    plt.savefig(f"{args.output_png_folder}/{map_name}_candidates.png", dpi=300, bbox_inches="tight")
    plt.close()




def plot_target_locations(args, map_name, method ='graph', if_filtered = False, if_show_place_name = False):
    # method choose from 'graph' or 'ilp'

    if if_filtered:
        with open(f'{args.input_linking_folder}/{map_name}_{method}_output_filtered.json', 'r') as f:
            data = json.load(f)

    else:

        with open(f'{args.input_linking_folder}/{map_name}_{method}_output.json', 'r') as f:
            data = json.load(f)

    # Convert to GeoDataFrame
    df = []
    for key, value in data.items():

        # place_name, _, geo_name, lat, lon, match_site_feature_class, match_site_feature_code, =  value
        place_name = value["place_name"]
        geo_name = value["geo_name"]
        lat = value["lat"]
        lon = value["lon"]
        match_site_feature_class = value["feature_class"]
        match_site_feature_code = value.get("feature_code",None)


        df.append([key, place_name, geo_name, lon, lat, Point(lon, lat), match_site_feature_class, match_site_feature_code])


        # place_name, _, geo_name, lat, lon = value
        # df.append([key, place_name, geo_name, lon, lat, Point(lon, lat)])

    gdf = gpd.GeoDataFrame(df, columns=["id", "place_name", "geo_name", "lon", "lat", "geometry", "feature_class", "feature_code"], crs="EPSG:4326")

    # Convert to Web Mercator (EPSG:3857) for contextily
    gdf = gdf.to_crs(epsg=3857)

    # Calculate bounding box of points
    minx, miny, maxx, maxy = gdf.total_bounds
    padding_x = (maxx - minx) * 0.1  # 10% padding
    padding_y = (maxy - miny) * 0.1  # 10% padding

    # Load US state boundaries
    states = gpd.read_file('../data/ne_110m_admin_1_states_provinces/ne_110m_admin_1_states_provinces.shp')
    
    states = states.to_crs(epsg=3857)


    # Plot the locations
    fig, ax = plt.subplots(figsize=(10, 8))

    states.boundary.plot(ax=ax, edgecolor='black', linewidth=1)


    # Map feature class to description
    feature_class_descriptions = {
        "A": "country, state, region,...",
        "H": "stream, lake,...",
        "L": "parks, area,...",
        "P": "city, village,...",
        "R": "road, railroad",
        "S": "spot, building, farm",
        "T": "mountain, hill, rock,...",
        "U": "undersea",
        "V": "forest, heath,..."
    }

    # Define colors for each feature_class
    unique_classes = gdf["feature_class"].unique()
    color_map = plt.get_cmap("tab10")
    class_colors = {cls: color_map(i % 10) for i, cls in enumerate(unique_classes)}

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    for cls, color in class_colors.items():
        subset = gdf[gdf["feature_class"] == cls]
        subset.plot(
            ax=ax, 
            color=color, 
            markersize=100, 
            alpha=0.7, 
            edgecolor="black", 
            label=cls
        )

    # Legend with descriptions
    handles = []
    for cls, color in class_colors.items():
        description = feature_class_descriptions.get(cls, "unknown")
        label = f"{cls}: {description}"
        handles.append(mpatches.Patch(color=color, label=label))

    ax.legend(handles=handles, title="Feature Class", loc='best', fontsize='small')



    # gdf.plot(ax=ax, color='red', markersize=100, alpha=0.7, edgecolor="black")


    if if_filtered and not gdf.empty:
        # Calculate bounding box of points
        minx, miny, maxx, maxy = gdf.total_bounds

        # Handle edge case where all points are identical (zero width/height bbox)
        if maxx > minx and maxy > miny:
            padding_x = (maxx - minx) * 1
            padding_y = (maxy - miny) * 1
        else:
            padding_x = 1_000  # default padding in meters
            padding_y = 1_000

        # Apply padded limits
        ax.set_xlim(minx - padding_x, maxx + padding_x)
        ax.set_ylim(miny - padding_y, maxy + padding_y)



    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, crs=gdf.crs, zoom=5)


    if if_show_place_name:
        # Add text labels
        for x, y, place_name, geo_name in zip(gdf.geometry.x, gdf.geometry.y, gdf.place_name, gdf.geo_name):
            ax.text(x, y, f"{place_name}\n({geo_name})", fontsize=10, ha="right", color="black") 

    # Set the limits to zoom in around the points
    ax.set_xlim(minx - padding_x, maxx + padding_x)
    ax.set_ylim(miny - padding_y, maxy + padding_y)

    # Remove axes for better visualization
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")

    plt.title("Geospatial Locations of Place Names")

    if if_filtered:
        plt.savefig(f"{args.output_png_folder}/{map_name}_visualization_filtered.png", dpi=300, bbox_inches="tight")
    else:

        # Save the figure before displaying it
        plt.savefig(f"{args.output_png_folder}/{map_name}_visualization.png", dpi=300, bbox_inches="tight")
    plt.close()

    # # Show the plot
    # plt.show()



if __name__ == '__main__':
    args = parse_arguments()
    print(args)
    # first_k = 20

    # with open('../data/rumsey/linking_results_maptext_competition_rumsey.json', 'r') as f:
    #     data = json.load(f)

    # for image_data in data[0:first_k]:
    #     image_name = image_data['image']
    #     map_name = os.path.basename(image_name).split('_')[0]

    map_list = ['2516015', '2554002','4472002','11640099']

    map_list = [
        "6e0627f218de28de12de19da46da46da659a541e54195a19485a4d1826cb20e3", 
        "273333b333b333b327b32f132b931bb30db38c188c3a2f3a3fb380318e13b012", 
        "643a1e320732133206b30eb215b20d3209b235b22cf624762932293223f222f3", 
        "4988298f0e3126b331233893061304a302570643032b0a3303830d0304110a58", 
        "00000c0f3b1f3e273e0d3e0f3e373e273f27362f3f2b3f0737271727000b0008", 
        "559844da265a1b1a209b2c9b449a20da249806da30da24dc265a827af238433b", 
        "687272e4366438e437645d745fe46be46de44f6444e446e54865567076f40716", 
        "471d57983cf867721373233246ba16fa3718339a169a4f9a26de20dc88d84c79", 
        "7c1b501b659b659b64bb156b1a7b5adb136b065b032b792f7ceb1ceb129d21a0", 
        "a4023a721af22bf333ba2a3a3a1e24362e6306a322462c6e364c231a2d9aa044"
    ]

    map_list=[
        "643a1e320732133206b30eb215b20d3209b235b22cf624762932293223f222f3", # sorted by rmse 
        "8024103436321eb20da34433033289b291b2913681b487308db00630653041b4",
        "2c854d1d4c2d132f48df102d2a4d3aed678e638e5996749e3e6e5e4e420ea1e2",
        "18233602320256061e035a031b031e07761b76037a0b3b234a0b0f230f2f9308",
        "0d231b00cd3c591e593619127903190b3f3b3f3b5d0b9d23393f5f334b0f1f0c"
    ]

    # map_list = [
    #     "25946d9651a703ad1dad35ad45a54dad67ad5dad7dad67b439c84d6c10fe10d3",
    #     "29c803eb66b34cd919915cb139335d3364bb24ad2c2b30cf22d913d031d82188",
    #     "88e14ee9368972892a39309930c93adb769b6899295b252b2adb809b6309030d",
    #     "4a8c56dd499d59e4378f068f5a0d0c4d260c1f1d1f8f0d8d0d8d15de41c8618d"
    # ]

    # # map_list = ['geologymapnorth','test']
    # map_list = ['geology_map_south_h0_w0', 'Teacup_pluton_alt_map_h0_w0', '101130GES013811_3_h0_w0']

    # map_list = ['geo_map_filtered', 'geology_map_south_filtered', 'Teacup_pluton_alt_map_filtered', '101130GES013811_3_filtered']

    # map_list = [
    #     "526008d60ad703cf22df22d520cb22cb20c624c632c78f27608429662fc73f0f",
    #     "88e14ee9368972892a39309930c93adb769b6899295b252b2adb809b6309030d"
    # ]

    # map_list = [
    #     '1960760-8-14-30',
    # ]

    # map_list = [
    #     '1612797-56-27-53',
    #     '1784751-1-0-0',
    #     '1786056-1-0-3',
    #     '1787745-1-34-35',
    #     '1875189-8-42-20',
    #     '1960760-8-14-30',
    #     '1977748-31-9-9',
    #     '1977748-44-9-9',
    #     '1983023-14-16-26',
    #     '1983037-4-9-9',
    #     '1983037-5-9-10',
    #     '1983040-8-0-19',
    #     '1983055-3-13-22',
    #     '1984286-5-11-10',
    #     '2041094-4-6-30'
    # ]


    input_folder = '../code/outputs/ngmdb_nickel_1123_gnis_step1'
    map_list = glob.glob(input_folder + '/*_matched_sites.json')
    map_list = sorted([os.path.basename(a).split('_')[0] for a in map_list])


    for map_name in map_list:

        if args.plot_candidate:
            if args.with_feature_class:
                plot_candidate_locations_with_feature_class(args, map_name, if_color_by_feature_class = True, if_show_place_name = args.with_text_label)
            else:
                plot_candidate_locations(args, map_name, if_show_place_name = args.with_text_label)

        try:
            if args.plot_target:
                plot_target_locations(args, map_name, method = args.method, if_filtered = args.filtered_suffix, if_show_place_name = args.with_text_label)
        except Exception as e:
            print(e)

# without compatibility cost

# python 4_plot.py --input_linking_folder='outputs/ngmdb_nickel_1123_gnis_step2ilp_notype'  --output_png_folder='outputs_visual/ngmdb_nickel_1123_gnis_step2ilp_notype' --plot_target --method='ilp' --with_text_label

# with type compatibiltiy cost
# python 4_plot.py --input_candidates_folder='../code/outputs/ngmdb_nickel_1123_gnis_step1'  --output_png_folder='outputs_visual/ngmdb_nickel_1123_gnis_step2ilp' --plot_candidate
# python 4_plot.py --input_linking_folder='outputs/ngmdb_nickel_1123_gnis_step2ilp'  --output_png_folder='outputs_visual/ngmdb_nickel_1123_gnis_step2ilp' --plot_target --method='ilp' --with_text_label


# python 4_plot.py --input_candidates_folder='outputs/geoclek_step1_0930_ci'  --output_png_folder='analyze/plot_geoclek_0930_ci' --plot_candidate


# python 4_plot.py --input_candidates_folder='outputs/geoclek_step1_0930_kg'  --output_png_folder='analyze/plot_geoclek_0930_kg' --plot_candidate

# python 4_plot.py --input_candidates_folder='outputs/geoclek_step1_0927'  --output_png_folder='analyze/plot_geoclek_0927' --plot_candidate
# python 4_plot.py --input_linking_folder='outputs/geoclek_step2_0927'  --output_png_folder='analyze/plot_geoclek_0927' --plot_target


# python 4_plot.py --input_candidates_folder='outputs/ngmdb_nickel_0908_geonames_step1'  --output_png_folder='analyze/plot_nickel_0908_geonames' --plot_candidate


# python 4_plot.py --input_candidates_folder='outputs/ngmdb_nickel_0802_gnis_step1'  --output_png_folder='analyze/plot_nickel_0802' --plot_candidate
# python 4_plot.py --input_linking_folder='outputs/ngmdb_nickel_0802_gnis_step2'  --output_png_folder='analyze/plot_nickel_0802' --plot_target --filtered_suffix


# python 4_plot.py --input_candidates_folder='outputs/rio_geonames_08010_step1'  --output_png_folder='analyze/plot_rio_0810' 

# python 4_plot.py --input_candidates_folder='outputs/usarandom_geonames_step1_08010'  --output_png_folder='analyze/plot_usarandom_0810'



# python 4_plot.py --input_candidates_folder='outputs/ngmdb_nickel_0706' --input_linking_folder='outputs/ngmdb_nickel_0706_graph' --output_png_folder='analyze/plot_0706'


# python 4_plot.py --input_candidates_folder='outputs/ngmdb_0616' --input_linking_folder='outputs/ngmdb_0616' --output_png_folder='outputs/ngmdb_0616'



# python 4_plot.py --input_candidates_folder='outputs/ngmdb_0531' --input_linking_folder='outputs/ngmdb_0608' --output_png_folder='outputs/ngmdb_0608'

# python 4_plot.py --input_candidates_folder='outputs/ngmdb_0531' --input_linking_folder='outputs/ngmdb_0608_candidates_geonames' --output_png_folder='outputs/ngmdb_0608_candidates_geonames'


