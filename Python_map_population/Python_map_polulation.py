import json
import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np
import requests

District_lat_lon = {
"全国": [35.68944,139.69167],
"北海道": [43.06417,141.34694],
"青森": [40.82444,140.74],
"岩手": [39.70361,141.1525],
"宮城": [38.26889,140.87194],
"秋田": [39.71861,140.1025],
"山形": [38.24056,140.36333],
"福島": [37.75,140.46778],
"茨城": [36.34139,140.44667],
"栃木": [36.56583,139.88361],
"群馬": [36.39111,139.06083],
"埼玉": [35.85694,139.64889],
"千葉": [35.60472,140.12333],
"東京": [35.68944,139.69167],
"神奈川": [35.44778,139.6425],
"新潟": [37.90222,139.02361],
"富山": [36.69528,137.21139],
"石川": [36.59444,136.62556],
"福井": [36.06528,136.22194],
"山梨": [35.66389,138.56833],
"長野": [36.65139,138.18111],
"岐阜": [35.39111,136.72222],
"静岡": [34.97694,138.38306],
"愛知": [35.18028,136.90667],
"三重": [34.73028,136.50861],
"滋賀": [35.00444,135.86833],
"京都": [35.02139,135.75556],
"大阪": [34.68639,135.52],
"兵庫": [34.69139,135.18306],
"奈良": [34.68528,135.83278],
"和歌山": [34.22611,135.1675],
"鳥取": [35.50361,134.23833],
"島根": [35.47222,133.05056],
"岡山": [34.66167,133.935],
"広島": [34.39639,132.45944],
"山口": [34.18583,131.47139],
"徳島": [34.06583,134.55944],
"香川": [34.34028,134.04333],
"愛媛": [33.84167,132.76611],
"高知": [33.55972,133.53111],
"福岡": [33.60639,130.41806],
"佐賀": [33.24944,130.29889],
"長崎": [32.74472,129.87361],
"熊本": [32.78972,130.74167],
"大分": [33.23806,131.6125],
"宮崎": [31.91111,131.42389],
"鹿児島": [31.56028,130.55806],
"沖縄": [26.2125,127.68111],
}

df = pd.read_excel(".\\n220200200.xlsx", header=None, skiprows=7, skipfooter=3)
new_columns = ['prefecture','population_h22','population_h27',
'col4','incresed_ratio_h22-h27','population_r2','col7','col8','incresed_ratio_h27-r2']
df.set_axis(new_columns, axis=1, inplace=True)

df["lat"] = df["prefecture"].apply(lambda x: District_lat_lon[x][0])
df["lon"] = df["prefecture"].apply(lambda x: District_lat_lon[x][1])

df2 = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [35.68944,139.69167],
    columns=['lat', 'lon'])

df3 = df[['prefecture', 'lat', 'lon', 'population_r2']][1:]
df3 = df3.astype({'population_r2': int})

st.pydeck_chart(pdk.Deck(
     map_style='mapbox://styles/mapbox/light-v9',
     initial_view_state=pdk.ViewState(
         latitude=35.68944,
         longitude=139.69167,
         zoom=5,
         pitch=50,
     ),
     layers=[
        #  pdk.Layer(
        #     'HexagonLayer',
        #     data=df3,
        #     get_position='[lon, lat]',
        #     get_weight="population_r2", # doesnt work
        #     radius=5000,
        #     elevation_scale=4,
        #     elevation_range=[0, 1000],
        #     pickable=True,
        #     extruded=True,
        #  ),
         pdk.Layer(
            'HeatmapLayer',
            data=df,
            get_position='[lon, lat]',
            get_weight="population_r2",
            opacity=0.8,
            cell_size_pixels=15,
            elevation_scale=4,
            elevation_range=[0, 1000]
        )  
     ],
 ))