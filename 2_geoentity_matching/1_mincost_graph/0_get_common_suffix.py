
import os 
import pandas as pd
import argparse
import json 
import pdb 


def parse_arguments():
    parser = argparse.ArgumentParser(description='Script to merge map data.')

    parser.add_argument('--reference', type=str, default = 'geonames', choices = ['mrds', 'roi','kg', 'geonames','gnis'], help = 'reference dataset')
    parser.add_argument('--output_path', type=str, default = 'geonames_suffix.csv', help = 'output csv file')
    

    return parser.parse_args()


def get_common_suffixes(df, namekey, top_k=200):
    """
    Given a DataFrame and the column name containing place names,
    return the most frequent single-word suffixes.
    """

    # Extract last word from each name
    suffixes = (
        df[namekey]
        .astype(str)
        .str.strip()
        .str.split()
        .str[-1]
        .str.lower()
    )

    # Count frequency
    suffix_counts = suffixes.value_counts()

    # Return top_k as list of (suffix, count)
    return suffix_counts.head(top_k)


def save_suffixes(suffix_series, out_path):
    """
    Save suffix frequency list to a file.
    Accepts a Pandas Series (index = suffix, value = count).
    """

    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Save as CSV
    suffix_series.to_csv(out_path, header=['count'])

    
    print(f"Suffix list written to: {out_path}")




def main(args):


    first_k = 200

    if args.reference == 'mrds':
        df = pd.read_csv('../data/mrds.csv')  # Replace with your file path
    elif args.reference == 'roi':
        df = pd.read_csv('../data/tungsten_target_region.csv')  # Replace with your file path
    elif args.reference == 'kg':
        with open('../data/minmod_kg_merge.json', 'r') as f:
            data = json.load(f)

        df = pd.DataFrame(data)
        # df = df.drop_duplicates()
    elif args.reference == 'geonames':
        
        # Define column names
        columns = [
            "geonameid", "name", "asciiname", "alternatenames", "latitude", "longitude",
            "feature_class", "feature_code", "country_code", "cc2", "admin1_code",
            "admin2_code", "admin3_code", "admin4_code", "population", "elevation",
            "dem", "timezone", "modification_date"
        ]

        # Load the text file into a DataFrame
        file_path = '../data/geonames_US.txt'  # Replace with full allcountries path
        # file_path = '../data/geonames_CI.txt'  # 

        df = pd.read_csv(file_path, sep='\t', header=None, names=columns)
        df['name'] = df['name'].astype(str)

    elif args.reference == 'gnis':
        gnis_txt_path = '../data/GNIS/DomesticNames_National_Text/Text/DomesticNames_National.txt'
        df = pd.read_csv(gnis_txt_path, sep='|')
        df['feature_name'] = df['feature_name'].astype(str)
    else:
        raise NotImplementedError


    # Display the first few rows of the DataFrame
    print('First few rows of the reference database:')
    print(df.head())



    print(f'Number of records {len(df)}' )



    if args.reference == 'kg':
        namekey = 'ms_name' 
    elif args.reference == 'geonames':
        namekey = 'name'
    elif args.reference == 'gnis':
        namekey = 'feature_name'
    elif args.reference == 'roi' or args.reference == 'mrds':
        namekey = 'site_name'
    else:
        raise NotImplementedError


    # --- Compute suffixes ---
    top_suffixes = get_common_suffixes(df, namekey, top_k=50)
    print("\nTop suffixes:")
    print(top_suffixes)

    save_suffixes(top_suffixes, args.output_path)


if __name__ == '__main__':
    args = parse_arguments()
    print(args)

    main(args)
