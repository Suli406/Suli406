import pandas as pd
import requests

pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

url = 'https://randomuser.me/api/?results=100'
response = requests.get(url)

if response.status_code == 200:
    data = response.json()

    names, address, email, city, country = [],[],[],[],[]

    for users in data['results']:
        first_name = users['name']['first']
        last_name = users['name']['last']
        street_number = users['location']['street']['number']
        street_name = users['location']['street']['name']
        cities = users['location']['city']
        countries = users['location']['country']
        emails = users['email']

        names.append(f'{first_name} {last_name}')
        address.append(f'{street_number} {street_name}')
        email.append(emails),
        city.append(cities)
        country.append(countries)

    df = pd.DataFrame({
        'Name': names,
        'Address': address,
        'Email': email,
        'City': city,
        "Country": country
    })
    print(df)
else:
    print('Failed to retrieve data')