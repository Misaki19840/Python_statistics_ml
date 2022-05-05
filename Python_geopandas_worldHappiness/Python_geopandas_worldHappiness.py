import geopandas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

import warnings

fig, ax = plt.subplots(figsize=(10, 8))

yearName = st.select_slider('Year', options=["2015","2016","2017","2018","2019"])
fpath = ".\\" + yearName + ".csv"
# print(fpath)

world = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))
world_happiness = pd.read_csv(fpath)
world_happiness_rn1 = world_happiness.rename(
    columns={'Happiness Score': 'Score',
             'Happiness.Score': 'Score', 
             'Country': 'Country or region'})
world_happiness_rn2 = world_happiness_rn1.replace('United States', 'United States of America')

world_happiness_final = world.merge(world_happiness_rn2, how="left", left_on=['name'], right_on=['Country or region'])
# print(world_happiness_final.columns)
# print(world_happiness_final.head())
# print(world_happiness_final['name'])

world_happiness_final.plot(column="Score",ax=ax,legend=True,
                    legend_kwds={'label': "Happiness Score",'orientation': "horizontal"})
plt.title("World Happiness Report")

st.pyplot(fig)