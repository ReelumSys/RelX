import streamlit as st
import streamlit as st
import pandas as pd
import plost
import base64
import numpy as np
import Main
import Main as StValv
from PIL import Image
import altair as alt

im = 'favicon2.png'
st.set_page_config(
    page_title="RelX v0.9",
    page_icon=im,
    layout="wide",
)

#st.image("../RelX/favicon2.png")
#global StValv
#StValv =20
#print(StValv)
#StartingValue = StValv

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
st.sidebar.header('')

plot_height = st.sidebar.slider('Specify plot height', 200, 1000, 250)
#st.button("Next Page")
st.sidebar.markdown('''
---

''')
#st.markdown('### XRD Charts')
user_input = st.number_input("Please enter a starting number for  \u00b0 2Theta. Minimum should be at least the first higher number \u00b0 2Theta data point of the two diffraction patterns, so they can adjust.")


df1 = pd.read_csv('ksev1.csv', names=['Theta','Int'])#, skiprows = 80
df2 = pd.read_csv('ksev1rand.csv', names=['Theta2','Int2'])


#df1zone = df1['Int']
#df2zone = df2['Int2']
#df1Theta = df1['Theta']
global failure_count
global failure_count2

StartingValue =user_input 

values = df1['Theta']  # list of values

failure_count = ["fail" for x in values if x < StartingValue].count("fail")  # list comprehension

print(failure_count)

values2= df2['Theta2']  # list of values

failure_count2 = ["fail" for x in values2 if x < StartingValue].count("fail")  # list comprehension

print(failure_count2)


df1 = pd.read_csv('ksev1.csv', names=['Theta','Int'], skiprows = failure_count)

global dfSize
#df1 = dfSize

df2 = pd.read_csv('ksev1rand.csv', names=['Theta2','Int2'], skiprows = failure_count2)
#print(df1)

weatherTheta2 = df2['Theta2']

#np.savetxt('testksev1.txt', df1, fmt='%f', delimiter=',')
#np.savetxt('testksev2.txt', df2, fmt='%f', delimiter=',')
df1zone = df1['Int']
df2zone = df2['Int2']
np.savetxt('test1zone.txt', df1zone, fmt='%f', delimiter=',')
np.savetxt('test2zone.txt', df2zone, fmt='%f', delimiter=',')


df1Theta = df1['Theta']
np.savetxt('testTheta.txt', df1Theta, fmt='%f', delimiter=',')
#np.savetxt('testInt.txt', df1zone, fmt='%f', delimiter=',')
#np.savetxt('testInt2.txt', df2zone, fmt='%f', delimiter=',')
df1 = pd.read_csv('test1zone.txt', names=['Int'])
df2 = pd.read_csv('test2zone.txt', names=['Int'])

dfmess = df1 - df2
dfmess.dropna(how='any', inplace=True)
np.savetxt('testInt10.txt', dfmess, fmt='%f', delimiter=',')

dfTheta = pd.read_csv('testTheta.txt', header=None)
dfTheta.columns = ['Theta']
df = pd.read_csv('testInt10.txt', header=None)
df.columns = ['Int']

#dfmerge = df1Theta.join(df)
#print(df1Theta)
#df_merged = pd.concat([df, dfmess], ignore_index=True) #WORKK

df_merged = dfTheta.join(df)
np.savetxt('testTheta10.txt', df_merged, fmt='%f', delimiter=',')
#print(df_merged)
df_merged.dropna(how='any', inplace=True)
np.savetxt('testTheta2.txt', df_merged, fmt='%f', delimiter=',')


global weather1
weather1 = pd.read_csv('ksev1.csv', names=['\u00b0 2Theta','Int'], skiprows=failure_count)
weather2 = pd.read_csv('ksev1rand.csv', names=['\u00b0 2Theta','Int'], skiprows=failure_count2)
weather3 = pd.read_csv('testTheta2.txt', names=['\u00b0 2Theta','Int'])



weatherTheta = dfTheta['Theta']
#weatherTheta = dfTheta['Theta']
#print(weatherTheta)
weahter5 = weather1['Int']
weahter6 = weather2['Int']


weatherlog1 = np.log(weahter5)
weatherlog4 = np.log(weahter6)

np.savetxt('testLog.txt', weatherlog1, fmt='%f', delimiter=',')
np.savetxt('testLogComp.txt', weatherlog4, fmt='%f', delimiter=',')
np.savetxt('testThetaLog.txt', weatherTheta, fmt='%f', delimiter=',')
np.savetxt('testThetaLog2.txt', weatherTheta2, fmt='%f', delimiter=',')

weatherLogXX = pd.read_csv('testLog.txt', names=['Int'])
weatherLogYY = pd.read_csv('testLogComp.txt', names=['Int'])
weatherThetaXX = pd.read_csv('testThetaLog.txt', names=['\u00b0 2Theta'])
weatherThetaYY = pd.read_csv('testThetaLog2.txt', names=['\u00b0 2Theta'])


weatherMerge = weatherThetaXX.join(weatherLogXX)
weatherMerge2 = weatherThetaYY.join(weatherLogYY)
#print(weatherMerge)
print(weatherThetaXX)
df1 = pd.DataFrame(weatherLogXX)
df2 = pd.DataFrame(weatherLogYY)
print(weatherLogXX)
print(weatherLogYY)
weatherLogDiff10 = weatherLogXX['Int'] - weatherLogYY['Int']

weatherLogDiff10.columns = ['Int']
print(weatherLogDiff10)
weatherLogDiff12 = pd.DataFrame(weatherLogDiff10)
weatherLogDiff12.columns = ['Int']
weatherThetaXX['Int'] = weatherLogDiff12
print(weatherThetaXX)



#weather1 = pd.read_csv('../RelX/ksev1.csv', names=['2Theta','Int'])
#weather2 = pd.read_csv('../RelX/ksev1rand.csv', names=['2Theta','Int2'])
#weather3 = pd.read_csv('../RelX/testTheta2.txt', names=['2Theta','Diff'])


st.markdown('##### Main')
st.line_chart(weather1, x = '\u00b0 2Theta', y = 'Int', height = plot_height)
st.markdown('##### Comparing')
st.line_chart(weather2, x = '\u00b0 2Theta', y = 'Int', height = plot_height)
st.markdown('##### Main - Comparing')
st.line_chart(weather3, x = '\u00b0 2Theta', y = 'Int', height = plot_height)

st.markdown('##### Main - Log Scale')
st.line_chart(weatherMerge, x = '\u00b0 2Theta', y = 'Int', height = plot_height)
st.markdown('##### Comp - Log Scale')
st.line_chart(weatherMerge2, x = '\u00b0 2Theta', y = 'Int', height = plot_height)
st.markdown('##### Main - Comp - Log Scale')
st.line_chart(weatherThetaXX, x = '\u00b0 2Theta', y = 'Int', height = plot_height)

chart_data = pd.DataFrame(
    weatherThetaXX,
    columns=['a', 'b'])

c = alt.Chart(chart_data).mark_circle().encode(
    x='a', y='b', size='b', color='b', tooltip=['a', 'b'])

st.altair_chart(c, use_container_width=True)
