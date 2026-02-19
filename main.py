import pandas as pd
import geopandas as gpd
import multiprocessing
from quackosm import PbfFileReader
from pathlib import Path
import matplotlib.pyplot as plt

ELEPHANT_LOCATIONS = 'elephants.csv'
NEPAL_PBF = 'nepal-210101.osm.pbf'
OUTPUT_CSV = 'output/elephants_with_distances.csv'
OUTPUT_BOXPLOT = 'output/boxplot.png'

TAGS = {
    "trunk_roads": ["trunk"],
    "primary_roads_and_above": ["trunk", "primary"],
    "secondary_roads_and_above": ["trunk", "secondary"],
    "tertiary_roads_and_above": ["trunk", "tertiary"],
    "all_roads": ["trunk", "tertiary", "unclassified"]
}

def get_min_distances(points_gdf, lines_gdf, dist_col_name):
    if len(lines_gdf) == 0:
        print(f"Warning: No lines for {dist_col_name} – all distances will be NaN")
        distances = pd.Series([pd.NA] * len(points_gdf), index=points_gdf.index)
        return distances

    joined = points_gdf.sjoin_nearest(lines_gdf, how="left", distance_col=dist_col_name)
    joined = joined.reset_index(names="original_idx")
    min_rows = joined.loc[joined.groupby("original_idx")[dist_col_name].idxmin()]
    distances = min_rows.set_index("original_idx")[dist_col_name]
    distances = distances.reindex(points_gdf.index)
    
    return distances

if __name__ == '__main__':
    multiprocessing.freeze_support()

    gdfs = {}
    distances = {}

    ele_df = pd.read_csv(ELEPHANT_LOCATIONS)
    ele_df = ele_df.reset_index(drop=True)
    ele_gdf = gpd.GeoDataFrame(
        ele_df,
        geometry=gpd.points_from_xy(ele_df['longitude'], ele_df['latitude']),
        crs="EPSG:4326"
    ).to_crs('EPSG:32645')

    reader = PbfFileReader(tags_filter={"highway": True})
    gpq_path = reader.convert_pbf_to_parquet(NEPAL_PBF)

    all_roads_gdf = gpd.read_parquet(gpq_path)
    print(f"Loaded {len(all_roads_gdf):,} highways")

    if 'tags' in all_roads_gdf.columns and 'highway' not in all_roads_gdf.columns:
        all_roads_gdf['highway'] = all_roads_gdf['tags'].apply(lambda t: t.get('highway') if isinstance(t, dict) else None)

    all_roads_gdf = all_roads_gdf[all_roads_gdf['geometry'].is_valid & ~all_roads_gdf['geometry'].is_empty]
    result_df = ele_df.copy()

    for name, tag in TAGS.items():
        gdfs[name] = all_roads_gdf[all_roads_gdf['highway'].isin(tag)][['geometry']].copy()
        print(f"{name}: {len(gdfs[name]):,}")
        gdfs[name] = gdfs[name].to_crs("EPSG:32645")
        result_df[f'dist_to_{name}'] = get_min_distances(ele_gdf, gdfs[name], f'dist_to_{name}')

    result_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved to: {OUTPUT_CSV}")

    dist_cols = [f'dist_to_{col}' for col in TAGS.keys() if f'dist_to_{col}' in result_df.columns]
    data = [result_df[col].dropna().values for col in dist_cols]
    tick_labels = [col.replace('dist_to_', '').replace('_', '\n') for col in dist_cols]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.boxplot(data, tick_labels=tick_labels, patch_artist=True,
               boxprops=dict(facecolor='steelblue', alpha=0.7),
               medianprops=dict(color='red', linewidth=2))
    ax.set_title('Elephant Distance to Road Types')
    ax.set_xlabel('Road Type')
    ax.set_ylabel('Distance (metres)')
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(OUTPUT_BOXPLOT, dpi=150)
    plt.show()
    print(f"Boxplot saved to {OUTPUT_BOXPLOT}")

    for col in TAGS.keys():
        if f'dist_to_{col}' in result_df.columns:
            s = result_df[f'dist_to_{col}']
            print(f"\n{col}:")
            print(f"Average: {s.mean():.1f}")
            print(f"Median:  {s.median():.1f}")
            print(f"Max:     {s.max():.1f}")
            print(f"Min:     {s.min():.1f}")
