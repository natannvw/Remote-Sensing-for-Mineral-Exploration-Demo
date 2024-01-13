import geopandas as gpd
from pyproj.crs import CRS
from rasterio.io import MemoryFile
from rasterio.mask import mask


def reproject_gdf(
    gdf: gpd.GeoDataFrame, dst_crs=CRS.from_epsg(4326)
) -> gpd.GeoDataFrame:
    """
    This function takes a GeoDataFrame and a target CRS as input,
    checks if the CRS of the GeoDataFrame is the same as the target CRS,
    if not, reprojects the GeoDataFrame to the target CRS.
    """
    # Reproject the GeoDataFrame if the CRS does not match
    if CRS(gdf.crs) != dst_crs:
        print(f"Reprojecting GeoDataFrame from {gdf.crs} to {dst_crs}")
        gdf = gdf.to_crs(dst_crs)

    return gdf


def clip_raster(raster: Raster, polygon: gpd.GeoDataFrame) -> Raster:
    polygon = reproject_gdf(polygon, dst_crs=raster.profile["crs"])

    with MemoryFile() as memfile:  # the mask() function expects a Rasterio dataset as the src argument
        with memfile.open(**raster.profile) as src:
            src.write(raster.datacube)

            clipped_data, clipped_transform = mask(src, polygon.geometry, crop=True)

    # Update the profile
    clipped_profile = raster.profile.copy()
    clipped_profile.update(
        {
            "height": clipped_data.shape[1],
            "width": clipped_data.shape[2],
            "transform": clipped_transform,
        }
    )

    # Assign the clipped data to the object
    raster.datacube = clipped_data
    raster.profile = clipped_profile

    return raster
