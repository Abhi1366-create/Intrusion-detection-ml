# ai intrusion detection system

a machine learning system to detect and classify network attacks using the nsl-kdd dataset

## preview

<table>
<tr>
<td><img src="Screenshoot/screenshot-logs.png" width="100%"></td>
<td><img src="Screenshoot/screenshot-charts.png" width="100%"></td>
</tr>
<tr>
<td><img src="Screenshoot/screenshot-sim1.png" width="100%"></td>
<td><img src="Screenshoot/screenshot-sim2.png" width="100%"></td>
</tr>
</table>

dashboard showing logs, attack distribution and simulation results

---

## overview

this project builds a simple intrusion detection system using a two-stage approach

- stage 1 detects whether traffic is normal or attack  
- stage 2 classifies the attack into dos, probe, r2l, u2r  

it also includes a dashboard to visualize results and simulate attacks

---

## features

- binary and multi-class classification  
- dashboard using streamlit  
- attack simulation  
- logs and charts  
- separate test script  

---

## project structure

ml-workspace/

- train.py  
- ui.py  
- test.py  

data/  
- kddtrain+.txt  
- kddtest+.txt  

models/  
- binary_model.pkl  
- multi_model.pkl  
- preprocessor.pkl  
- label_encoder.pkl  

readme.md  
requirements.txt  
.gitignore  

---

## setup

open your project folder

cd ml-workspace

create required folders

mkdir data  
mkdir models  

install dependencies

pip install pandas numpy scikit-learn xgboost streamlit joblib  

---

## dataset

download the nsl-kdd dataset from

https://web.archive.org/web/20150205070216/http://nsl.cs.unb.ca/nsl-kdd/

you only need these files

- kddtrain+.txt  
- kddtest+.txt  

place them like this

ml-workspace/data/

- kddtrain+.txt  
- kddtest+.txt  

---

## run the project

train the model

python train.py  

run the dashboard

streamlit run ui.py  

run test script (optional)

python test.py  

---

## how to use

manual mode

- enter values in sidebar  
- click run detection  
- check logs  

simulation mode

- select attack type  
  normal / dos / probe / r2l / u2r  
- click run simulation  
- view logs and charts  

---

## output

- prediction (normal or attack)  
- attack type  
- confidence score  
- risk level  
- logs  
- charts  

---

## important

- dataset is not included  
- run train.py before ui.py  
- dataset must be inside data folder  

---

## common errors

dataset not found  
place kddtrain+.txt and kddtest+.txt inside data  

models not found  
run python train.py  

---

## author

Abhishek