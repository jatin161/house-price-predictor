import pickle
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import MinMaxScaler
st.write(
    """
    <div style="margin-align:center;">
    <H1>House Value Prediction(Ames Iowa) </H1>""",
    unsafe_allow_html=True,
)
st.write(
    """
    <div style="margin-align:center;">
    <H3>Steps To use</H3>
    1) Fill out the Basic details about the your dream houses<br>
    2) Click on button at the bottom <br>
    <br>
    </div>
    """,
    unsafe_allow_html=True,
)
base = pd.read_excel("base.xlsx")
def get_scalar_value(metric,input_value,base):
    value=base[(base['metric']==metric) & (base['input_value']==input_value)]
    value=value.reset_index()
    value=value['value'][0]
    return value


def df_creator(GrLivArea,OverallQuality,Neighborhood,GarageCars,KitchenQual,s1stFlrSF,base):
    data=pd.DataFrame(data=[{"GrLivArea":GrLivArea,
                            "OverallQual":get_scalar_value("OverallQual",OverallQuality,base),
                            "Neighborhood": get_scalar_value("Neighborhood", Neighborhood,base),
                            "GarageCars": get_scalar_value("GarageCars", GarageCars,base),
                            "KitchenQual": get_scalar_value("KitchenQual", KitchenQual,base),
                            "1stFlrSF": s1stFlrSF
                            }])


    return data


model = pickle.load(open('house_model.pkl', 'rb'))

GrLivArea = np.log(st.slider('GrLivArea: Above grade (ground) living area square feet', 1, 5000, 1))/8.517193191416238
OverallQuality = st.slider('OverallQual : Overall material and finish quality', 1, 10, 1)

Neighborhood_lis=["Blmngtn","Blueste","BrDale","BrkSide","ClearCr" , "CollgCr","Crawfor",
                  "Edwards","Gilbert","IDOTRR","MeadowV","Mitchel","NAmes",	"NoRidge",
                  "NPkVill",	"NridgHt",	"NWAmes",	"OldTown",	"Sawyer",	"SawyerW",
                  "Somerst",	"StoneBr",	"SWISU",	"Timber",	"Veenker",]

Neighborhood = st.selectbox('Neighborhood: Physical locations within Ames city limits',Neighborhood_lis)
GarageCars = st.slider('GarageCars: Size of garage in car capacity', 0, 4, 1)

KitchenQual_lis=["Ex","Fa","Gd","TA"]

KitchenQual = st.selectbox('KitchenQual: Kitchen quality', KitchenQual_lis)
s1stFlrSF = np.log(st.slider('1stFlrSF: First Floor square feet', 1, 5000, 1))/8.517193191416238



data=df_creator(GrLivArea,OverallQuality,Neighborhood,GarageCars,KitchenQual,s1stFlrSF,base)


print(data.to_clipboard())

if (st.button("Predit Value of The house") ):
    value=round(np.exp(model.predict(data[:1]))[0])
    st.write("""
    <div style="margin-align:center;">
    <H4>Your House Will be valued at $ {:,}</H4>
    
    </div>""".format(value)

    ,
    unsafe_allow_html = True
    )

