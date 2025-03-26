import requests

def get_airport_coordinates(city_name):
    # Replace with your GeoNames username
    username = 'tisha_agrawal_08'
    
    # GeoNames API endpoint for searching airports
    url = f'http://api.geonames.org/searchJSON?name={city_name}&featureClass=S&featureCode=AIRP&maxRows=1&username={username}'
    
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        if data['totalResultsCount'] > 0:
            airport = data['geonames'][0]
            airport_name = airport['name']
            latitude = airport['lat']
            longitude = airport['lng']
            print(f"Nearest Airport: {airport_name}")
            print(f"Latitude: {latitude}, Longitude: {longitude}")
            return latitude, longitude
        else:
            print("No airports found for the given city.")
            return None, None
    else:
        print(f"Error: {response.status_code}")
        return None, None

# Example Usage
city1 = input("Enter a deparuture name: ")
city2 = input("Enter a arrival name: ")
lat, lon = get_airport_coordinates(city1)
lat, lon = get_airport_coordinates(city2)