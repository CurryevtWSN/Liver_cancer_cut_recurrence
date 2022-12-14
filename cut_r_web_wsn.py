import streamlit as st
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
#应用标题
st.set_page_config(page_title='Prediction model for recurrence of hepatocellular carcinoma after hepatectomy')
st.title('Prediction model for recurrence of hepatocellular carcinoma after hepatectomy：Machine Learning-Based development and interpretation study')
st.sidebar.markdown('## Variables')
Gender = st.sidebar.selectbox('Gender',('male','female'),index=1)
Tumor_size_3 = st.sidebar.selectbox('Tumor size',('≤3cm','>3cm'),index=1)
Tumor_single = st.sidebar.selectbox('Tumor number',('Single tumor','Multiple tumor'),index=1)
BCLC = st.sidebar.selectbox('BCLC stage',('Stage 0','Stage A','Stage B','Stage C'),index=1)
HBV_DNA_10000 = st.sidebar.selectbox('HBV DNA',('≤10000','>10000'),index=1)


N = st.sidebar.slider("Neutral granulocyte (10^9/L)", 0.00, 10.00, value=4.50, step=0.01)
M = st.sidebar.slider("Macrophages (10^9/L)", 0.00, 1.00, value=0.50, step=0.01)
NLR = st.sidebar.slider("Ratio of neutrophils to lymphocytes", 0.00, 5.00, value=2.50, step=0.01)
Fibrinogen = st.sidebar.slider("Fibrinogen (mg/dL)", 0.00, 10.0, value=5.00, step=0.01)

#分割符号
st.sidebar.markdown('#  ')
st.sidebar.markdown('#  ')
st.sidebar.markdown('##### All rights reserved') 
st.sidebar.markdown('##### For communication and cooperation, please contact wshinana99@163.com, Wu Shi-Nan, Nanchang university')
#传入数据
map = {'≤3cm':0,'>3cm':1,'Single tumor':1, 'Multiple tumor':2,'female':2, 'male':1,'Stage 0':0,"Stage A":1,'Stage B':2,'Stage C':3,'≤10000':0,'>10000':1}
Gender =map[Gender]
Tumor_size_3 = map[Tumor_size_3]
Tumor_single = map[Tumor_single]
BCLC =map[BCLC]
HBV_DNA_10000 =map[HBV_DNA_10000]

hp_train = pd.read_csv('liver_cut_data_recurrence.csv')
hp_train['Recurrence'] = hp_train['Recurrence'].apply(lambda x : +1 if x==1 else 0)
features =["Gender","Tumor_size_3","Tumor_single","BCLC","HBV_DNA_10000",'N',"M",'NLR','Fibrinogen']
target = 'Recurrence'
random_state_new = 50
data = hp_train[features]
for name in ['Fibrinogen']:
    X = data.drop(columns=f"{name}")
    Y = data.loc[:, f"{name}"]
    X_0 = SimpleImputer(missing_values=np.nan, strategy="constant").fit_transform(X)
    y_train = Y[Y.notnull()]
    y_test = Y[Y.isnull()]
    x_train = X_0[y_train.index, :]
    x_test = X_0[y_test.index, :]

    rfc = RandomForestRegressor(n_estimators=100, random_state=random_state_new)
    rfc = rfc.fit(x_train, y_train)
    y_predict = rfc.predict(x_test)

    data.loc[Y.isnull(), f"{name}"] = y_predict
    
X_data = data

#转换自变量
X_ros = np.array(X_data)
y_ros = np.array(hp_train[target])
mlp = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='lbfgs',
                    alpha=0.0001,
                    batch_size='auto',
                    learning_rate='constant',
                    learning_rate_init=0.01,
                    power_t=0.5,
                    max_iter=200,
                    shuffle=True, random_state=random_state_new)

mlp.fit(X_ros, y_ros)
sp = 0.5
#figure
is_t = (mlp.predict_proba(np.array([[Gender,Tumor_size_3,Tumor_single,BCLC,HBV_DNA_10000,N,M,NLR,Fibrinogen]]))[0][1])> sp
prob = (mlp.predict_proba(np.array([[Gender,Tumor_size_3,Tumor_single,BCLC,HBV_DNA_10000,N,M,NLR,Fibrinogen]]))[0][1])*1000//1/10


if is_t:
    result = 'High Risk Recurrence'
else:
    result = 'Low Risk Recurrence'
if st.button('Predict'):
    st.markdown('## Result:  '+str(result))
    if result == '  Low Risk Recurrence':
        st.balloons()
    st.markdown('## Probability of High risk Recurrence group:  '+str(prob)+'%')

