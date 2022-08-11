#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
matplotlib.rcParams['figure.figsize'] = (20,10)
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# Bengaluru_House_Data.csv
Bengaluru_House_Data_df=pd.read_csv('Bengaluru_House_Data.csv')


# In[4]:


Bengaluru_House_Data_df


# In[5]:


Bengaluru_House_Data_df.describe()


# In[6]:


#Visualising all numeric variable
#import libraries
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6,12))
sns.pairplot(Bengaluru_House_Data_df)
plt.show()


# In[7]:


corre = Bengaluru_House_Data_df.corr()
corre

plt.figure(figsize=(6,6))
sns.heatmap(corre,annot=True)


# In[8]:


Bengaluru_House_Data_df.shape


# In[9]:


# checking count means type of categories of flat
Bengaluru_House_Data_df.groupby('area_type')['area_type'].agg('count')


# In[10]:


Bengaluru_House_Data_df.columns


# In[11]:


Bengaluru_House_Data_df.dtypes


# In[12]:


# Removing unused columns
col_to_drops = ['area_type','society','balcony','availability']
col_to_drops


# In[13]:


Bengaluru_House_Data_df.drop(columns = col_to_drops,inplace = True)


# In[14]:


# Now columns count after removing columns with null values
Bengaluru_House_Data_df


# In[15]:


# making temp data frame to show percenage of missing value
na_df = pd.DataFrame({'Column_name':Bengaluru_House_Data_df.columns,
                      'Na_count':Bengaluru_House_Data_df.isnull().sum(),
                      'Na_percentage':Bengaluru_House_Data_df.isnull().sum()/Bengaluru_House_Data_df.shape[0]*100})
# sorting in Descending order 
na_df.sort_values(by='Na_percentage',ascending = False)


# In[16]:


# Droping missing value rows from Dataset
Bengaluru_House_Data_df.dropna(inplace = True)


# In[17]:


Bengaluru_House_Data_df.isnull().sum()


# In[18]:


# Now rows count after removing rows with null values
Bengaluru_House_Data_df.shape


# In[19]:


# Here in size column 3 word indicating no.of room i.e BHK,Bedroom,RK
Bengaluru_House_Data_df['size'].unique()


# In[20]:


# now instead of using 1 bhk now we use i or 2 
Bengaluru_House_Data_df['BHK'] = Bengaluru_House_Data_df['size'].apply(lambda x: int(x.split(' ')[0]))


# In[21]:


Bengaluru_House_Data_df


# In[22]:


# different value of bedroom, hall & kitchen
Bengaluru_House_Data_df['BHK'].unique()


# In[23]:


#we can see 43 bedroom house but there area is very less, its a error bcoz 2400 sfeet is very small for 43 bhk
Bengaluru_House_Data_df[Bengaluru_House_Data_df.BHK>20]


# In[24]:


# we can see some times value in range and some times single value..so take average of range value
Bengaluru_House_Data_df['total_sqft'].unique()


# In[25]:


def is_float(x):
    try:
        float(x)
    except:
        return False
    return True


# In[26]:


# here total_sqft column is not uniform totly unstructured so we take average of Range no.
Bengaluru_House_Data_df[~Bengaluru_House_Data_df['total_sqft'].apply(is_float)].head(15)


# In[27]:


def convert_sqrft_to_no(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None


# In[28]:


Bengaluru_House_Data_df['total_area'] = Bengaluru_House_Data_df['total_sqft'].apply(convert_sqrft_to_no)


# In[29]:


# making new data frame for removing extra extra column we forget to remove i'e size and total_sqft
bng_house_data = Bengaluru_House_Data_df.copy()
bng_house_data.head(5)


# In[30]:


# Removing unused columns
col_to_remove = ['total_sqft','size']

bng_house_data.drop(columns = col_to_remove,inplace = True)


# In[31]:


bng_house_data.head(5)


# In[32]:


# Here we have 2 column price and total_area we all know in realstate price-per-sqrft is very imp
# This feature will help us do some outlier cleaing
# price in lakh
bng_house_data['price_per_sqrft'] = bng_house_data['price']*100000/bng_house_data['total_area']
bng_house_data.head(10)


# In[33]:


# checking area
bng_house_data['location'].unique()


# In[34]:


# checking no of rows per location high no. of location/rows in area
len(bng_house_data['location'].unique())


# In[35]:


# we have lot of rows per area that's tough to handle this's called dimensionlity curse there r 
# techinque to reduce this one the effective techinque is "other category" which mean 
# when you have lot of location like 30405 in which many of 1-2 two area/data point to find this given below


# In[36]:


# strping/remove extra space from location to clean tha data 
bng_house_data.location = bng_house_data.location.apply(lambda x: x.strip())
# checking statics of location
location_stats = bng_house_data.groupby('location')['location'].agg('count').sort_values(ascending = False)
location_stats
# Here some location has many data point/area and some has only one 


# In[37]:


# any location which has less than 10 data point is called other location now checking for no. of location having
# less than 10 data point

len(location_stats[location_stats<=10])


# In[38]:


# now checking for location having less than 10 data point
location_status_less_10 = location_stats[location_stats<=10]
location_status_less_10


# In[39]:


# moving less than 10 data point to other
bng_house_data.location = bng_house_data.location.apply(lambda x: 'other' if x in location_status_less_10 else x)
len(bng_house_data.location.unique())


# In[40]:


bng_house_data.head(10)


# In[41]:


bng_house_data.shape


# In[42]:


# here total_area data is incorrect W.R.T BHK cuz 6 room with 1020 ft and 8 room with 600ft 
# so we romove all these type of data point
bng_house_data[bng_house_data.total_area/bng_house_data.BHK<300].head(10)


# In[43]:


# ccheck no. of rows in data frame
bng_house_data.shape


# In[44]:


# removing some outliers by using nigate nigate use to filter on ur critrea set
# making df  
city_house_data = bng_house_data[~(bng_house_data.total_area/bng_house_data.BHK<300)]
city_house_data.shape


# In[45]:


# checking now price_per_sqrft column using describe method its is used to checking detail about any column
# now can see minimum price in 267 sqrft is worng in banglore
# so we remove these cases based on standred deviation
city_house_data.price_per_sqrft.describe()


# In[46]:


# here we write a function that can remove outliers per location cuz some location have price some have less price
# so we have to find mean and standred deviation and filter any data point which is beyond one standred deviation
def remove_pps_outliers(Bengaluru_House_Data_df):
    df_out = pd.DataFrame()
    for key, subdf in Bengaluru_House_Data_df.groupby('location'):
        m = np.mean(subdf.price_per_sqrft)
        st = np.std(subdf.price_per_sqrft)
        reduced_df = subdf[(subdf.price_per_sqrft>(m-st)) & (subdf.price_per_sqrft<(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index = True)
    return df_out    
# making df  
town_data = remove_pps_outliers(city_house_data)
town_data.shape


# In[47]:


# again there is problem in room price 2 room price is costlier than 3 room so we need to visualize this
def plot_scatter_chart(Bengaluru_House_Data_df,location):
    bhk2 = Bengaluru_House_Data_df[(Bengaluru_House_Data_df.location==location) & (Bengaluru_House_Data_df.BHK==2)]
    bhk3 = Bengaluru_House_Data_df[(Bengaluru_House_Data_df.location==location) & (Bengaluru_House_Data_df.BHK==3)]
    matplotlib.rcParams["figure.figsize"] = (15,10)
    plt.scatter(bhk2.total_area,bhk2.price,color='blue',label='2 BHK',s=50)
    plt.scatter(bhk3.total_area,bhk3.price,marker= '+',color='green',label='3 BHK',s=50)
    plt.xlabel("Total_squre_feet_area")
    plt.ylabel("price")
    plt.title(location)
    plt.legend
        
plot_scatter_chart(town_data,"Hebbal") 


# In[48]:


def remove_bhk_outlier(Bengaluru_House_Data_df):
    exclude_indices = np.array([])
    for location, location_df in Bengaluru_House_Data_df.groupby('location'):
        bhk_stats = {}
        for BHK,bhk_df in location_df.groupby('BHK'):
            bhk_stats[BHK] = {
                'mean':np.mean(bhk_df.price_per_sqrft),
                'std':np.std(bhk_df.price_per_sqrft),
                'count':bhk_df.shape[0]
            }
        for bhk,bhk_df in location_df.groupby('BHK'):
            
            stats = bhk_stats.get(BHK-1)
            if stats and stats['count']>5:
                
                exclude_indices = np.append(exclude_indices,bhk_df[bhk_df.price_per_sqrft<(stats['mean'])].index.values)
           
    return Bengaluru_House_Data_df.drop(exclude_indices,axis = 'index')
# making new df
land_data_df = remove_bhk_outlier(town_data)

land_data_df.shape        


# In[49]:


plot_scatter_chart(land_data_df,"Hebbal")


# In[50]:


plt.hist(land_data_df.price_per_sqrft,rwidth=0.8)
plt.xlabel('price_per_sqrft')
plt.ylabel('count')


# In[51]:


#checking no. of bathroom
land_data_df.bath.unique()


# In[52]:


# checking bathroom with no. of bhk and area
land_data_df[land_data_df['bath']>10]


# In[53]:


# checking bathrom outlier
land_data_df[land_data_df.bath>land_data_df.BHK+2]


# In[54]:


# removing outlier and making new df
city_land_df = land_data_df[land_data_df.bath<land_data_df.BHK+2]
city_land_df.shape


# In[55]:


# making new df
soil_final = city_land_df.drop('price_per_sqrft',axis='columns')
soil_final.head(5)


# In[56]:


# here location column is text column and meachine learning model can not interpret text data so we convert into no.
# so we use one hot encoding
dummies = pd.get_dummies(soil_final.location)
dummies.head(5)


# In[57]:


# now we concate both data frame soil_final and dummies
data11 = pd.concat([soil_final,dummies],axis = 'columns')
data11


# In[ ]:




