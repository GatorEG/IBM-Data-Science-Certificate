# ----------------------------------------------------------------------
# IBM Data Science Capstone Project
# Produced by: Ethan Goldstein, ethan@ethangoldstein.ca
# ----------------------------------------------------------------------

import pandas as pd  								# creating and managing dataframes
import requests 									# calling the Foursquare API
import copy  										# deep copying dataframes
import folium  										# for creating maps
from sklearn.cluster import KMeans  				# for clustering
from sklearn.preprocessing import StandardScaler  	# for data normalization
from yellowbrick.cluster import KElbowVisualizer   	# for drawing the elbow chart viz

# Set Display Options
pd.set_option('display.max_columns', 10000)
pd.set_option('display.max_rows', 10000)

capstoneFolder = 'REDACTED'

# Load subway stop latitute and longitude
subwayStops = pd.read_csv(capstoneFolder + 'stopsVenueCount.csv')

# ----------------------------------------------------------------------
# Map subway stations on folium map
# ----------------------------------------------------------------------

# FIGURE 1 - TTC SUBWAY STATIONS
m1 = folium.Map(
    location=[43.653963, -79.387207],  # Toronto
    zoom_start=12,
    tiles='Stamen Terrain')

for stop, lat, lon in zip(subwayStops['station'],
                          subwayStops['avg_lat'],
                          subwayStops['avg_lon']):
    label = '{}'.format(stop)
    label = folium.Popup(label)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='white',
        fill_opacity=0.7).add_to(m1)

# Save map as html file

m1.save(capstoneFolder + 'mappedStations.html')

# FIGURE 2 - TTC SUBWAY STATIONS AND WALKING RADIUS
# Generate base map object
m2 = folium.Map(
    location=[43.653963, -79.387207],  # Toronto
    zoom_start=12,
    tiles='Stamen Terrain')

# Add grey 'walking radius' circle
for stop, lat, lon in zip(subwayStops['station'],
                          subwayStops['avg_lat'],
                          subwayStops['avg_lon']):
    folium.Circle(
        [lat, lon],
        radius=1500,
        weight=1,
        color='grey',
        fill_color='grey',
        fill_opacity=0.25).add_to(m2)

# Add each subway station
for stop, lat, lon in zip(subwayStops['station'],
                          subwayStops['avg_lat'],
                          subwayStops['avg_lon']):
    label = '{}'.format(stop)
    label = folium.Popup(label)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='white',
        fill_opacity=0.7).add_to(m2)

# Save map as html file
m2.save(capstoneFolder + 'mappedStationWalkingRadius.html')

# ----------------------------------------------------------------------
# SCRAPE FOURSQUARE FOR VENUE COUNTS
# ----------------------------------------------------------------------

# Category IDs for the venue categories important to me
# Source: https://developer.foursquare.com/docs/resources/categories
categoryIDs = {'Libraries': '4bf58dd8d48988d12f941735',
               'Gluten Free Restaurants': '4c2cd86ed066bed06c3c5209',
               'Grocery Stores': '4bf58dd8d48988d118951735',
               'Pubs': '4bf58dd8d48988d11b941735',
               'Parks': '4bf58dd8d48988d163941735'}

# Foursquare query data
CLIENT_ID = 'REDACTED'
CLIENT_SECRET = 'REDACTED'
LIMIT = 100
RADIUS = 1500
VERSION = '20180605'
LIMIT = 200

# Format new dataframe
stopsVenueCount = copy.deepcopy(subwayStops)
stopsVenueCount.columns = ['Stop Name',
                           'Latitude',
                           'Longitute']

# Add extra columns for Foursquare venues
for x in ['Libraries', 'Gluten Free Restaurants',
          'Grocery Stores', 'Pubs', 'Parks']:
    stopsVenueCount[x] = ""

stopsVenueCount.set_index(['Stop Name'], inplace=True)

# Pull data from Foursquare
# loop through subway stops
for stop, lat, lng in zip(subwayStops['station'],
                          subwayStops['avg_lat'],
                          subwayStops['avg_lon']):

    #  loop through categories

    for key, cat in categoryIDs.items():
        url = ('https://api.foursquare.com/v2/venues/search?client_id={}'
               '&client_secret={}&v={}&ll={},{}&categoryId={}&radius={}'
               '&limit={}'.format(
                   CLIENT_ID,
                   CLIENT_SECRET,
                   VERSION,
                   lat,
                   lng,
                   cat,
                   RADIUS,
                   LIMIT))

        queryResults = requests.get(url).json()['response']['venues']
        venueCount = pd.DataFrame(queryResults).shape[0]

        # Add data to frame
        stopsVenueCount.at[stop, key] = venueCount

# Save dataframe to cache data
stopsVenueCount.to_csv(path_or_buf=capstoneFolder + 'stopsVenueCount.csv')

# ----------------------------------------------------------------------
# Cluster the subway stations
# ----------------------------------------------------------------------
station_clustering = copy.deepcopy(subwayStops)

# Prepare Dataframe
station_clustering.drop('Longitute', 1)
station_clustering.drop('Latitude', 1)
station_clustering.set_index(keys='Stop Name', inplace=True)

# Normalize data
scaler = StandardScaler()
scaler.fit(station_clustering)
station_transformed = scaler.transform(station_clustering)

kmeans = KMeans(n_clusters=3,
                init="k-means++",
                n_init=20).fit(station_transformed)

# Determine Optimal K value
kmeans_viz = KElbowVisualizer(kmeans, k=(1, 10))
kmeans_viz.fit(station_transformed)
kmeans_viz.show(outpath=capstoneFolder + 'k-elbow.png')

# Add clusters to dataframe and output results
station_clustering.insert(0, "Cluster Labels", kmeans.labels_)
station_clustering.reset_index(inplace=True)
station_clustering.to_csv(capstoneFolder + 'clusteringLabelsD.csv')

# Prepare map with clustered stations
colour_dict = {
    0: 'red',
    1: 'blue',
    2: 'green'}

m3 = folium.Map(
    location=[43.653963, -79.387207],  # Toronto
    zoom_start=12,
    tiles='Stamen Terrain')

for stop, lat, lon, col in zip(station_clustering['Stop Name'],
                               station_clustering['Latitude'],
                               station_clustering['Longitute'],
                               station_clustering['Cluster Labels']):
    print(col, colour_dict[col])
    label = '{}'.format(stop)
    label = folium.Popup(label)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='white',
        fill_opacity=0.7).add_to(m3)

m3.save(capstoneFolder + 'clusteredStations.html')
