

### Script to extract only fields from the source and append to the target.
### Creates a new file if the source doesn't exist


import pandas as pd


#### Company Incs

"""
#Company 1 Source : Fortune 1000
company_d1 = 'company_F1000.xlsx'
data = pd.read_excel(company_d1)
company_1 = pd.DataFrame(data['Company Name'])
company_1 = company_1.rename(index=str,columns={"Company Name":"Name"})

#Company 2 Source: Kaggle
company_d2 = 'company_kaggle.csv'
data = pd.read_csv(company_d2)
company_2 = pd.DataFrame(data['Company Name'].sample(30000,random_state=0))
company_2 = company_2.rename(index=str,columns={"Company Name":"Name"})

#Company 3 Source: Brands Export 
company_d3 = 'company_brands.xlsx'
data = pd.read_excel(company_d3)
company_3 = pd.DataFrame(data['Manufacturer'].sample(9000,random_state=0))
company_3 = company_3.rename(index=str,columns={"Manufacturer":"Name"})

company = pd.concat([company_1,company_2,company_3])


company.to_csv('company_merged.csv')
print('Completed saving company names',company.head())



### Places
# Cities and Countries: https://datahub.io/core/world-cities

city_1 = 'cities_countries.csv'
data = pd.read_csv(city_1)
city_names = pd.DataFrame(data['name'])
city_names = city_names.rename(index=str,columns={"name":"Name"})

country_names = data.drop_duplicates('country')
country_names = city_names.rename(index=str,columns={"country":"Name"})

location = pd.concat([city_names,country_names])
location.to_csv('location_merged.csv')

print('Completed saving location names',location.head())
"""

### Items and Goods


