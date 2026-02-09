import pandas as pd
import numpy as np
import json
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns

# --- Paths ---
gnis_txt_path = '../data/GNIS/DomesticNames_National_Text/Text/DomesticNames_National.txt'
gnis_to_geonames_mapping_json_path = '../data/GNIS/gnis_to_geonames_mapping.json'


# --- Data loading ---
def read_names_and_types(gnis_txt_path, mapping_json_path):
    df = pd.read_csv(gnis_txt_path, sep='|')

    with open(mapping_json_path, 'r') as f:
        gnis_to_geonames = json.load(f)

    def map_to_geonames(feature_class):
        return gnis_to_geonames.get(feature_class, {}).get("primary", "UNKNOWN")

    df['geonames_feature_class'] = df['feature_class'].map(map_to_geonames)

    map_to_feature_classes = df.groupby('map_name')['feature_class'].apply(lambda x: sorted(x.unique()))
    map_to_geonames_types = df.groupby('map_name')['geonames_feature_class'].apply(lambda x: sorted(x.unique()))

    return map_to_feature_classes, map_to_geonames_types


# --- Occurrence counting ---
def count_occurrence_frequency(map_to_types, level="feature"):
    """
    Count how many maps each feature type or superclass appears in.

    level: "feature" (full type) or "superclass" (part before '.')
    Returns DataFrame sorted by count descending.
    """
    counter = Counter()
    for types in map_to_types.values:
        if level == "superclass":
            items = set(t.split('.')[0] for t in types)
        else:
            items = set(types)
        counter.update(items)
    df = pd.DataFrame(counter.items(), columns=["item", "count"]).sort_values("count", ascending=False).reset_index(drop=True)
    return df


# --- Co-occurrence dictionary builder ---
def count_co_occurrence_frequency(map_to_types, level="feature"):
    """
    Build co-occurrence counts between types or superclasses.

    Returns nested dict {item1: {item2: count}}.
    """
    co_occurrence = defaultdict(lambda: defaultdict(int))
    for types in map_to_types.values:
        if level == "superclass":
            items = list(set(t.split('.')[0] for t in types))
        else:
            items = list(set(types))
        for i, a in enumerate(items):
            for j, b in enumerate(items):
                if i != j:
                    co_occurrence[a][b] += 1
    return dict(co_occurrence)


# --- Matrix conversion and normalization ---
def co_occurrence_dict_to_matrix(co_dict, occurrence_df=None, normalize=None):
    """
    Convert co-occurrence dict to matrix.

    If normalize=True, do simple proportion P(B|A) = co_occurrence(A,B) / occurrence(A).
    """
    all_items = sorted(set(co_dict.keys()) | {k for v in co_dict.values() for k in v})
    matrix = pd.DataFrame(0.0, index=all_items, columns=all_items, dtype=float)

    for a, neighbors in co_dict.items():
        for b, count in neighbors.items():
            matrix.at[a, b] = count

    if normalize:
        if normalize == 'conditional':
            if occurrence_df is None:
                raise ValueError("occurrence_df required for normalization")
            occ_series = occurrence_df.set_index("item")["count"]
            for a in matrix.index:
                denom = occ_series.get(a, 0)
                if denom > 0:
                    matrix.loc[a] /= denom
                else:
                    matrix.loc[a] = 0.0
        elif normalize == 'max':
            max_val = matrix.values.max()
            if max_val > 0:
                matrix /= max_val
            else:
                matrix[:] = 0.0
        else:
            raise NotImplementedError 

    np.fill_diagonal(matrix.values, 1.0)

    return matrix


# --- Plotting functions ---
# def plot_occurrence_frequency(df, title="Occurrence Frequency", top_n=20):
#     df_top = df.head(top_n)
#     plt.figure(figsize=(10, 6))
#     sns.barplot(data=df_top, x="count", y="item", palette="Blues_d")
#     plt.title(title)
#     plt.xlabel("Number of Maps Containing Item")
#     plt.ylabel("")
#     plt.tight_layout()
#     plt.show()
def plot_occurrence_frequency(df, title="Occurrence Frequency"):
    plt.figure(figsize=(10, max(6, len(df)*0.3)))  # Dynamically adjust height based on number of items
    sns.barplot(data=df, x="count", y="item", palette="Blues_d")
    plt.title(title)
    plt.xlabel("Number of Maps Containing Item")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()


def plot_co_occurrence_matrix(matrix, title="Co-occurrence Matrix", vmax=None):
    plt.figure(figsize=(14, 12))
    sns.heatmap(matrix, cmap="Blues", square=True, cbar=True,
                linewidths=0.5, linecolor='gray', vmax=vmax)
    plt.title(title)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


# --- Main workflow ---
def main():
    map_to_feature_classes, map_to_geonames_types = read_names_and_types(gnis_txt_path, gnis_to_geonames_mapping_json_path)

    for level, label in [("feature", "GeoNames Feature Types"), ("superclass", "GeoNames Superclasses")]:
        # Occurrence counts & plot
        occ_df = count_occurrence_frequency(map_to_geonames_types, level=level)
        plot_occurrence_frequency(occ_df, title=f"{label} Occurrence Frequency")

        # Co-occurrence counts
        co_dict = count_co_occurrence_frequency(map_to_geonames_types, level=level)

        # Raw counts matrix & plot
        matrix_counts = co_occurrence_dict_to_matrix(co_dict)
        plot_co_occurrence_matrix(matrix_counts, title=f"{label} Co-occurrence Counts")

        # Proportion matrix & plot
        matrix_condition_norm = co_occurrence_dict_to_matrix(co_dict, occurrence_df=occ_df, normalize='conditional')
        plot_co_occurrence_matrix(matrix_condition_norm, title=f"{label} Co-occurrence Conditional Probability", vmax=1)

        # Proportion matrix & plot normalized by max count
        matrix_max_norm = co_occurrence_dict_to_matrix(co_dict, occurrence_df=occ_df, normalize='max')
        plot_co_occurrence_matrix(matrix_max_norm, title=f"{label} Co-occurrence Max Normalization", vmax=1)

    # Also do GNIS feature classes (non-GeoNames) for comparison
    gnis_occ_df = count_occurrence_frequency(map_to_feature_classes, level="feature")
    plot_occurrence_frequency(gnis_occ_df, title="GNIS Feature Classes Occurrence Frequency")

    gnis_co_dict = count_co_occurrence_frequency(map_to_feature_classes, level="feature")
    gnis_matrix_counts = co_occurrence_dict_to_matrix(gnis_co_dict)
    plot_co_occurrence_matrix(gnis_matrix_counts, title="GNIS Feature Classes Co-occurrence Counts")

    gnis_matrix_condition_norm = co_occurrence_dict_to_matrix(gnis_co_dict, occurrence_df=gnis_occ_df, normalize='conditional')
    plot_co_occurrence_matrix(gnis_matrix_condition_norm, title="GNIS Feature Classes Co-occurrence Conditional Probability", vmax=1)

    gnis_matrix_max_norm = co_occurrence_dict_to_matrix(gnis_co_dict, occurrence_df=gnis_occ_df, normalize='max')
    plot_co_occurrence_matrix(gnis_matrix_max_norm, title="GNIS Feature Classes Co-occurrence Max Normalization", vmax=1)

    print(gnis_matrix_max_norm)
    gnis_matrix_max_norm.to_csv("gnis_matrix_norm.csv")


if __name__ == "__main__":
    main()
