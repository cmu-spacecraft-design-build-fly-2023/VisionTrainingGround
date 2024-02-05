"""
Google Earth Engine data downloader.

Author: Kyle McCleary
"""

import argparse
import os
import shutil
from multiprocessing import Pool,cpu_count
import ee
import requests
from retry import retry
import numpy as np
from getMGRS import getMGRS
import pyproj
from tqdm.contrib.concurrent import process_map  # Import process_map from tqdm
from tqdm import tqdm  # Make sure to import tqdm

def get_region_filter_from_bounds(bounds, get_rect=True):
    """
    Creates a filter for a given geographical rectangle defined by longitude and latitude bounds.

    Parameters:
    bounds (list): A list of four elements [left, top, right, bottom] defining the geographical rectangle in longitude and latitude.
    get_rect (bool): A flag to determine whether to return the rectangle geometry along with the region filter.

    Returns:
    ee.Filter: A filter that selects images intersecting with the defined rectangle.
    ee.Geometry.Rectangle (optional): The rectangle geometry, returned if getRect is True.
    """
    region_left, region_top, region_right, region_bottom = bounds
    rect_from_bounds = ee.Geometry.Rectangle([region_left, region_top, region_right, region_bottom])
    
    # If sensor is Sentinel 2, add 500 km buffer to bounds of grid region to avoid black sections of images.
    if args.sensor == 's2':
        out_rect = rect_from_bounds.buffer(500000)
    # If sensor is Landsat, do not add buffer.
    else:
        out_rect = rect_from_bounds
    region_filter_from_bounds = ee.Filter.bounds(out_rect)
    if get_rect:
        return region_filter_from_bounds, rect_from_bounds
    return region_filter_from_bounds

def get_date_filter(i_date, f_date):
    """
    Creates a date filter for selecting images within a specified date range.

    Parameters:
    i_date (str): Initial date of the date range in a format recognizable by the Earth Engine API. (e.g. '2022-01-01')
    f_date (str): Final date of the date range in a format recognizable by the Earth Engine API. (e.g. '2023-01-01')

    Returns:
    ee.Filter: A date filter for the specified date range.
    """
    ee_date_filter = ee.Filter.date(i_date, f_date)
    return ee_date_filter

def get_collection(sensor, ee_region_filter, ee_date_filter, ee_bands = None, cloud_cover_min = 0.0, cloud_cover_max = 30.0, date_sort=True):
    """
    Retrieves a filtered collection of Landsat images based on the specified parameters.

    Parameters:
    sensor (str): The sensor to pull images from. Options are l8, l9, or s2.
    ee_region_filter (ee.Filter): The geographical region filter.
    ee_date_filter (ee.Filter): The date range filter.
    ee_bands (list): A list of band names to include in the collection. Default is ['B4', 'B3', 'B2'].
    cloud_cover_min (float): The minimum cloud cover percentage for the images. Default is 0.0.
    cloud_cover_max (float): The maximum cloud cover percentage for the images. Default is 30.0.
    date_sort (bool): Flag to sort the collection by acquisition date.

    Returns:
    ee.ImageCollection: A collection of Landsat images filtered by the specified parameters.
    """
    if sensor == 'l8':
        collection_string = 'LANDSAT/LC08/C02/T1_TOA'
        cloud_string = 'CLOUD_COVER'
    elif sensor == 'l9':
        collection_string = 'LANDSAT/LC09/C02/T1_TOA'
        cloud_string = 'CLOUD_COVER'
    elif sensor == 's2':
        collection_string = 'COPERNICUS/S2_HARMONIZED'
        cloud_string = 'CLOUDY_PIXEL_PERCENTAGE'

    if ee_bands is None:
        ee_bands = ['B4', 'B3', 'B2']

    ee_collection = ee.ImageCollection(collection_string)
    ee_collection = ee_collection.filter(ee_region_filter).filter(ee_date_filter)
    ee_collection = ee_collection.filter(ee.Filter.lt(cloud_string, cloud_cover_max))
    ee_collection = ee_collection.filter(ee.Filter.gte(cloud_string, cloud_cover_min))
    ee_collection = ee_collection.select(ee_bands)
    if date_sort:
        ee_collection = ee_collection.sort('DATE_ACQUIRED')
    return ee_collection

def get_points_in_region(ee_region, num_points, pts_scale, pts_seed):
    """
    Selects random points within a specified region, focusing on land areas. Uses MODIS land/water data to filter out water bodies.

    Parameters:
    ee_region (ee.Geometry): The region within which to select points.
    num_points (int): The number of random points to select.
    pts_scale (float): The scale to sample points
    seed (int): A seed number for the random point generation to ensure reproducibility.

    Returns:
    list: A list of randomly selected geographical points (longitude and latitude) within the specified region.
    """
    water_land_data = ee.ImageCollection('MODIS/061/MCD12Q1')
    land = water_land_data.select('LW').first()
    mask = land.eq(2)
    selected_points = land.updateMask(mask).stratifiedSample(region=ee_region, scale = pts_scale,
                                                    classBand = 'LW', numPoints = num_points,
                                                    geometries=True,seed = pts_seed)
    return selected_points.aggregate_array('.geo').getInfo()

def make_rectangle(ee_point, pt_buffer):
    """
    Creates a rectangle geometry around a given point.

    Parameters:
    ee_point (dict): A dictionary containing the 'coordinates' key, which holds the longitude and latitude of the point.
    pt_buffer (float): A float value containing the radius in meters to extend rectangle bounds from the point.

    Returns:
    ee.Geometry.Rectangle: A rectangle geometry centered around the given point with a fixed buffer.
    """
    coords = ee_point['coordinates']

    if args.grid_key[-1] <= 'M':
        proj = "EPSG:327" + args.grid_key[:-1]
    else:
        proj = "EPSG:326" + args.grid_key[:-1]

    transformer = pyproj.Transformer.from_crs("EPSG:4326", proj, always_xy=True)
    transformed_pt = transformer.transform(coords[0], coords[1])
    pt_tl_x = transformed_pt[0] - pt_buffer
    pt_tl_y = transformed_pt[1] + pt_buffer
    pt_br_x = transformed_pt[0] + pt_buffer
    pt_br_y = transformed_pt[1] - pt_buffer
    pt_rect = ee.Geometry.Rectangle([pt_tl_x, pt_br_y, pt_br_x, pt_tl_y], proj, True, False)
    return pt_rect

def get_url(index):
    """
    Generates a download URL for a satellite image from the Earth Engine image collection.

    Parameters:
    index (int): The index of the image in the Earth Engine image list.

    Returns:
    str: A URL string from which the image can be downloaded.
    """
    image = ee.Image(im_list.get(index))
    if args.crs:
        crs = args.crs
    else:
        if args.sensor in ('l8','l9'):
            crs = image.select(0).projection()
        else:
            if args.grid_key[-1] <= 'M':
                crs = "EPSG:327" + args.grid_key[:-1]
            else:
                crs = "EPSG:326" + args.grid_key[:-1]
    if args.sensor in ('l8','l9'):
        image = image.multiply(650).toByte()
        image = image.clip(image.geometry())
    url = image.getDownloadURL({
        'scale':scale,
        'format':out_format,
        'bands':bands,
        'crs':crs})
    #print('URL',index,'done: ', url)
    return url

@retry(tries=20, delay=1, backoff=2)
def get_and_download_url(index):
    """
    Downloads an image from a retrieved URL and saves it to a specified path. This function will retry up to 10 times 
    with increasing delays if the download fails.

    Parameters:
    index (int): The index of the image, used for retrieving the URL and naming the downloaded file.

    Notes:
    The file is saved in either the GeoTIFF or PNG format, depending on the 'out_format' variable.
    The file name is constructed using the sensor, region name, and index.
    """
    url = get_url(index)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
        #print(out_path, 'folder created')
    if out_format == 'GEOTiff':
        ext = '.tif'
    else:
        ext = '.png'
    out_name = args.sensor + '_' + region_name + '_' + str(index).zfill(5) + ext
    r = requests.get(url, stream=True)
    if r.status_code !=200:
        r.raise_for_status()
    with open(os.path.join(out_path,out_name),'wb') as out_file:
        shutil.copyfileobj(r.raw, out_file)
    #print('Download',out_name, 'done')

def argument_parser():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--bounds', nargs='+', type=int, default=[-84, 24, -78, 32])
    parser.add_argument('-g', '--grid_key', type=str)
    parser.add_argument('-i', '--idate',type=str, default='2020')
    parser.add_argument('-f', '--fdate',type=str, default='2023')
    parser.add_argument('-s', '--scale', type = float, default = 328.0)
    parser.add_argument('-m', '--maxims', type = int, default = 10)
    parser.add_argument('-se', '--sensor', choices=['l8', 'l9', 's2'], type=str, default = 'l8')
    parser.add_argument('-o', '--outpath', type=str, default = 'landsat_images')
    parser.add_argument('-r', '--region', type=str, default=None)
    parser.add_argument('-e', '--format', type=str,default = 'GEOTiff')
    parser.add_argument('-sd', '--seed', type=int,default = None)
    parser.add_argument('-c', '--crs', type=str, default = None)
    parser.add_argument('-cc', '--cloud_cover_max',type=float, default = 30.0)
    parser.add_argument('-ccgt', '--cloud_cover_min', type=float, default = 0.0)
    parser.add_argument('-ba','--bands',type=str,nargs='+',default =['B4','B3','B2'])
    parser.add_argument('-bs', '--batch_size', type=int, default = 100)
    parser.add_argument('-si', '--startindex', type=int, default=0)
    parsed_args = parser.parse_args()
    if parsed_args.region is None:
        parsed_args.region = parsed_args.grid_key
    return parsed_args


# Initialize high-volume earthengine api.
ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')

# Get command line arguments
args = argument_parser()
# Assigning parsed arguments to variables
scale = args.scale
max_ims = args.maxims
out_path = args.outpath
out_format = args.format
region_name = args.region
bands = args.bands


# Adjusting bounds if grid key is provided
if args.grid_key:
    grid = getMGRS()
    left, bottom, right, top = grid[args.grid_key]
    args.bounds = [float(left), float(bottom), float(right), float(top)]

if args.region is None:
    args.region = args.grid_key

# Setting seed for random number generation
if args.seed:
    seed = args.seed
else:
    seed = np.random.randint(100000)

region_filter, region_rect = get_region_filter_from_bounds(args.bounds, get_rect=True) # Getting region filter and rectangle from bounds
date_filter = get_date_filter(args.idate, args.fdate) # Getting date filter based on input dates

# Processing image collection based on selected options
collection = get_collection(args.sensor, region_filter, date_filter, 
                            ee_bands=bands, cloud_cover_min = args.cloud_cover_min,
                            cloud_cover_max=args.cloud_cover_max, date_sort=True)

# Process landsat sensor.
if args.sensor in ('l8', 'l9'):
    collection_size = collection.size().getInfo()
    if collection_size < max_ims:
        max_ims = collection_size
    im_list = collection.toList(max_ims)
# Process sentinel sensor.
elif args.sensor == 's2':
    im_list = []
    if args.seed:
        seed = args.seed
        np.random.seed = seed
    else:
        seed = np.random.randint(100000)
    # Select random points in region.
    points = get_points_in_region(region_rect, max_ims, scale, np.random.randint(100000))
    # Reproject collection to correct UTM region.
    for point in points:
        # Create landsat sized rectangle around point and filter collection
        clip_rect = make_rectangle(point, 185000/2)
        collection_with_random_column = collection.filter(clip_rect)
        # Add random value to each image in collection.
        collection_with_random_column = collection.randomColumn('random',np.random.randint(100000))
        # Sort by random value to change mosaic.
        collection_with_random_column = collection_with_random_column.sort('random')
        # Convert feature collection back to image collection.
        collection_with_random_column = ee.ImageCollection(collection_with_random_column)
        # Create mosaic and scale to vis spectrum and byte.
        im = collection_with_random_column.mosaic().multiply(0.0001).divide(0.3).multiply(255).toByte()
        # Clip image to landsat-like rectangle.
        rect_im = im.clip(clip_rect)
        im_list.append(rect_im)
    im_list = ee.List(im_list)
    
if __name__ == '__main__':
    indexes = range(args.startindex,args.startindex+max_ims)
    batches = len(indexes)/args.batch_size
    if batches%1 != 0:
        batches = batches + 1
    batches = int(batches)
    p = Pool(cpu_count())
    with Pool(cpu_count()) as p:
        # Wrap the outer loop with tqdm to track batch progress
        for i in tqdm(range(batches), desc="Processing batches"):
            if i != batches - 1:
                batch_idxs = indexes[i * args.batch_size:(i + 1) * args.batch_size]
            else:
                batch_idxs = indexes[i * args.batch_size:]
            
            # Execute the batch task
            p.starmap(get_and_download_url, zip(batch_idxs))

    p.close()
    p.join()