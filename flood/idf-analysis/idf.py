import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import geopandas as gpd
import pandas as pd
import typer
import xarray as xr
from idf_analysis.idf_class import IntensityDurationFrequencyAnalyse
from shapely.geometry import Point
from tqdm.auto import tqdm


def idf(
    rainfall_nc_folder: Path = Path("data"),
    shape_file: Path = Path("data/mwi_admbnda_adm3_nso_20181016_new.shp"),
    admin_column: str = "ADM3_PCODE",
    output_folder: Path = Path("output"),
    melt_df: bool = typer.Option(
        False, help="When true, intensity will be melted into 1 column."
    ),
    debug: bool = typer.Option(
        False, "--debug", help="When true, more output will be printed."
    ),
):
    """Performs IDF Analysis.

    rainfall nc files should:
    - contain the dimensions: [longitude, latitude, time]
    - contain the variable: [tp] containing the rainfall in meters

    shapefile should:
    - contain polygongons as geometry
    - contain a column with the name set in admin_column
    """

    gdf = gpd.read_file(shape_file)

    # Load nc files into xarray dataset
    xd = xr.concat(
        [xr.open_dataset(p) for p in rainfall_nc_folder.glob("*.nc")], dim="time"
    )

    # Create points for intersecting
    xd = xd.stack(point=["longitude", "latitude"])
    points = [Point(long, lat) for long, lat in xd.point.values]

    total = pd.DataFrame()

    param_folder = output_folder / "idf_parameters"
    param_folder.mkdir(parents=True, exist_ok=True)

    # Calculate IDF
    progress_bar = tqdm(gdf.iterrows(), desc="Performing IDF Analysis")
    for _, row in progress_bar:
        polygon = row.geometry
        admin_code = row[admin_column]
        progress_bar.set_description(f"Performing IDF Analysis for region {admin_code}")

        points_inside_polygon = [(p.x, p.y) for p in points if polygon.intersects(p)]

        if not points_inside_polygon:
            if debug:
                print(f"No Points found inside Polygon for zone: {admin_code}")
            continue

        # Get all rainfall data for relevant points
        df = xd.loc[{"point": points_inside_polygon}].to_dataframe()

        # Average over all points in polygon
        df = df.groupby(level=0).mean()

        # Convert rainfall from m to mm
        df["tp"] = df["tp"] * 1000.0

        # Perform IDF based on https://markuspic.github.io/intensity_duration_frequency_analysis/example/example_python_api.html
        idf = IntensityDurationFrequencyAnalyse(extended_durations=True)
        idf._parameter.durations = [
            i * 60 for i in [1, 2, 3, 4, 6, 9, 12, 18, 24, 36, 48, 72]
        ]
        idf.set_series(df["tp"])

        # Store idf parameters so they can be reused
        idf.auto_save_parameters(param_folder / f"idf_parameters_{admin_code}.yaml")
        idf_df = idf.result_table(return_periods=[2, 5, 10, 20, 50, 100]).round(2)

        # clean up result df
        idf_df["duration (hour)"] = idf_df.index / 60.0
        idf_df[admin_column] = admin_code
        if melt_df:
            idf_df = pd.melt(
                idf_df,
                id_vars=[admin_column, "duration (hour)"],
                var_name="return period (year)",
                value_name="intensity (mm)",
            )
        else:
            idf_df = idf_df.set_index([admin_column, "duration (hour)"])
            idf_df.columns = idf_df.columns.map(
                lambda col: f"intensity (mm) for return period {col} years"
            )

        total = pd.concat([total, idf_df])

    total.to_csv(output_folder / "intensity_duration_frequency.csv")
    return total


if __name__ == "__main__":
    typer.run(idf)
