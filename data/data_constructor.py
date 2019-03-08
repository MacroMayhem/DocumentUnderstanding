

### Script to extract only fields from the source and append to the target.
### Creates a new file if the source doesn't exist
### Takes care of removign special characters and stop words if necessary


import pandas as pd
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords as nltk_stopwords


def process_names(name_set, name, stopwords):
    lower_name = name.lower()
    processed_name = ""
    for char in lower_name:
        if char.isalnum():
            processed_name += char
        elif char == ' ' and len(processed_name) > 0 and processed_name not in stopwords:
            name_set.add(processed_name)
            processed_name = ""

    if len(processed_name) > 0 and processed_name not in stopwords:
        name_set.add(processed_name)

    return name_set

#### Company Incs

def get_company_data():
    stopwords = set(nltk_stopwords.words('english'))
    #Company 1 Source : Fortune 1000
    company_d1 = './data/company_F1000.xlsx'
    data = pd.read_excel(company_d1)
    company_1 = pd.DataFrame(data['Company Name'])
    company_1 = company_1.rename(index=str,columns={"Company Name":"name"})

    #Company 2 Source: Kaggle
    company_d2 = './data/company_kaggle.csv'
    data = pd.read_csv(company_d2)
    company_2 = pd.DataFrame(data['Company Name'].sample(15000,random_state=0))
    company_2 = company_2.rename(index=str,columns={"Company Name":"name"})


    """
    #Company 3 Source: Brands Export 
    company_d3 = 'company_brands.xlsx'
    data = pd.read_excel(company_d3)
    company_3 = pd.DataFrame(data['Manufacturer'].sample(9000,random_state=0))
    company_3 = company_3.rename(index=str,columns={"Manufacturer":"name"})
    """
    company = pd.concat([company_1,company_2],ignore_index=True)

    processed_names = set()

    for index, row in company.iterrows():
        try:
            processed_names = process_names(processed_names,row['name'],stopwords)
        except AttributeError:
            print(index, row, row['name'])

    processed_company = pd.DataFrame(columns=['name'])


    for idx,item in enumerate(processed_names):
        processed_company.loc[idx] = item


    print(processed_company.isnull().values.any())
    #processed_company.to_csv('company_merged.csv')
    print('Completed saving company names:',processed_company.shape)
    return processed_company



### Places
# Cities and Countries: https://datahub.io/core/world-cities

def get_location_data():
    stopwords = set(nltk_stopwords.words('english'))

    city_1 = './data/cities_countries.csv'
    data = pd.read_csv(city_1)
    city_names = pd.DataFrame(data['name']).sample(10000,random_state=0)
    city_names = city_names.rename(index=str,columns={"name":"name"})

    country_names = pd.DataFrame(data['country'].values,columns=['name'])
    country_names.drop_duplicates(inplace=True)

    location = pd.concat([city_names,country_names],ignore_index=True)
    processed_locations = pd.DataFrame(columns = ['name'])

    processed_names = set()

    for index, row in location.iterrows():
        try:
            processed_names = process_names(processed_names,row['name'],stopwords)
        except AttributeError:
            print(index, row, row['name'])


    for idx,item in enumerate(processed_names):
        processed_locations.loc[idx] = item


    print(processed_locations.isnull().values.any())
    #processed_locations.to_csv('location_merged.csv')

    print('Completed saving location names',processed_locations.shape)

    return processed_locations

### Items and Goods
# Source : Flipkart Kaggle

def flipkart_category_break(unique_categories,category_tree):

    categories =  category_tree.split('>>')
    for category in categories:
        lower_category = category.lower()
        processed_category = ""
        for char in lower_category:
            if char.isalnum():
                processed_category += char
            elif char == ' ' and len(processed_category) > 0:
                unique_categories.add(processed_category)
                processed_category = ""

        if len(processed_category) > 0:
            unique_categories.add(processed_category)

    return unique_categories


def get_items_cat():

    stopwords = set(nltk_stopwords.words('english'))

    goods_1 = './data/goods_flipkart.csv'
    data = pd.read_csv(goods_1)
    goods = data['product_category_tree'].drop_duplicates()

    category_set = set()

    for row in goods:
        category_set = flipkart_category_break(category_set,row)

    print('Unique categories of goods found:',len(category_set))

    processed_goods = pd.DataFrame(columns=['name'])

    for idx,item in enumerate(category_set):
        if item not in stopwords:
            processed_goods.loc[idx] = item

    print(processed_goods.isnull().values.any())
    #processed_goods.to_csv('goods.csv')

    return processed_goods