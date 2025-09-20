# Hour 4  - Modelling and Evaluation
## วัตถุประสงค์ 
- แยกแยะโจทย์ Regression (ทำนายตัวเลข) กับ Classification (ทำนายกลุ่ม)
- เข้าใจแนวคิด Baseline (ตัวตั้งต้น):
- Regression baseline: ทายค่ากลาง (เช่น ค่าเฉลี่ย)
- Classification baseline: ทาย คลาสที่มากสุด (majority class)
= สร้างโมเดลง่าย ๆ: Linear Regression และ Logistic Regression
= หลักปฏิบัติพื้นฐาน: Train/Test Split เพื่อกันการจำ (overfitting) 
 
## Background


### 1. Regression Models

Regression คือกระบวนการสร้างแบบจำลองความสัมพันธ์ระหว่างตัวแปรอิสระ (features) และตัวแปรตามเชิงตัวเลข (target variable) เพื่อใช้ในการ **ทำนาย (prediction)** หรือ **อธิบายความสัมพันธ์ (explanation)**

โดยทั่วไป **Linear Regression** ถือเป็นจุดเริ่มต้น (assumption-based) ส่วน **Tree-based และ Ensemble** ใช้เพื่อความแม่นยำสูงขึ้น แต่ยากต่อการตีความ

**การประเมิน (Evaluation Metrics)**
* **MAE (Mean Absolute Error):** ความคลาดเคลื่อนเฉลี่ยเชิงสัมบูรณ์ → คลาดเคลื่อนโดยเฉลี่ยเท่าไหร่
* **RMSE (Root Mean Squared Error):** วัดความคลาดเคลื่อนที่ให้โทษกับ error ขนาดใหญ่ → ดีสำหรับงานที่ outlier สำคัญ
* **\$R^2\$ (Coefficient of Determination):** วัดว่าโมเดลอธิบายความแปรปรวนของข้อมูลได้กี่เปอร์เซ็นต์
* **Baseline comparison:** เปรียบเทียบกับการทำนายง่าย ๆ เช่น ค่าเฉลี่ยหรือค่ามัธยฐาน เพื่อดูว่าโมเดลเพิ่มประสิทธิภาพจริงหรือไม่

**การแปลผลเชิงความหมาย**

* ถ้า **MAE = 5** → โดยเฉลี่ยโมเดลทำนายเพี้ยน 5 หน่วย (เช่น ราคาเพี้ยน 5 บาท)
* ถ้า **RMSE สูงกว่า MAE มาก** → มี error ขนาดใหญ่บางจุด (outlier)
* ถ้า **\$R^2 = 0.9\$** → โมเดลอธิบายความแปรปรวนของข้อมูลได้ 90% ถือว่าดี
* แต่ค่า metric ต้องแปลในบริบทของปัญหา เช่น การทำนายราคาบ้านเพี้ยน 10,000 บาทอาจเล็กน้อย หากราคาบ้านหลักล้าน แต่ถือว่าสำคัญมากในบริบทสินค้าราคาถูก


**ตารางสรุป Regression Algorithms**

| **ชื่อ**                                            | **ทฤษฎี (หลักการสำคัญ)**                                                                   | **การแปลผล (Interpretation)**                                                                                                          | **การประเมินที่เหมาะสม**                                           |
| --------------------------------------------------- | ------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| **Linear Regression**                               | หาความสัมพันธ์เชิงเส้นระหว่าง \$x\$ และ \$y\$ โดยใช้ OLS เพื่อลดผลรวมกำลังสองของ residuals | ค่าสัมประสิทธิ์ \$\beta\_j\$ = การเปลี่ยนแปลงของ \$y\$ ต่อ 1 หน่วยของ \$x\_j\$ (เมื่อคงตัวแปรอื่นคงที่), \$R^2\$ ใช้วัดการอธิบายข้อมูล | MAE, RMSE, \$R^2\$ (ควรดู \$R^2\$ เป็นหลักเพื่อแสดงความสัมพันธ์)   |
| **Polynomial Regression**                           | ขยาย Linear Regression โดยเพิ่ม \$x^2, x^3,...\$ เพื่อจับความโค้ง                          | ค่าสัมประสิทธิ์ไม่ตีความง่าย → ใช้กราฟแสดงความสัมพันธ์แทน                                                                              | MAE, RMSE, \$R^2\$ (เน้น RMSE เพราะอ่อนไหวต่อ overfitting)         |
| **Support Vector Regression (SVR)**                 | ใช้ margin และ kernel trick เพื่อสร้างฟังก์ชันที่เพี้ยนไม่เกิน \$\varepsilon\$             | ไม่ตีความเชิงค่าสัมประสิทธิ์ → มองเป็นโมเดลทำนาย, ใช้พารามิเตอร์ \$C, \varepsilon\$, kernel                                            | MAE, RMSE (เหมาะสำหรับข้อมูลที่มี outlier), \$R^2\$                |
| **Decision Tree Regression**                        | แบ่งข้อมูลเป็นกลุ่มย่อย (splits) โดยเลือกจุดที่ลด variance ได้มากที่สุด                    | แปลผลได้ง่ายเป็นกฎเชิงเงื่อนไข เช่น ถ้า \$x\_1>5\$ → ค่าเฉลี่ย = 120                                                                   | MAE, RMSE (แต่ควรใช้ cross-validation ป้องกัน overfitting)         |
| **Random Forest Regression**                        | สร้างหลาย ๆ tree แบบสุ่มแล้วเฉลี่ยผล → ลด variance                                         | ไม่ตีความตรง ๆ แต่ใช้ Feature Importance เพื่อดูความสำคัญของตัวแปร                                                                     | MAE, RMSE, \$R^2\$ (นิยมใช้ RMSE เพราะค่าเฉลี่ยหลาย tree ลด noise) |
| **Gradient Boosting (XGBoost, LightGBM, CatBoost)** | สร้าง tree แบบลำดับ โดย tree ใหม่แก้ error ของ tree เก่า → ลด bias                         | ตีความด้วย Feature Importance หรือ SHAP values                                                                                         | MAE, RMSE (นิยมใช้ RMSE เพราะอ่อนไหวต่อ error ขนาดใหญ่), \$R^2\$   |
| **K-Nearest Neighbors (KNN) Regression**            | ใช้ค่าเฉลี่ยของ \$k\$ เพื่อนบ้านใกล้เคียงทำนายค่าเป้าหมาย                                  | เข้าใจง่าย → ค่าที่คล้ายกันให้ผลใกล้เคียงกัน                                                                                           | MAE (ทนต่อ outlier ได้ดีกว่า RMSE), RMSE, \$R^2\$                  |
| **Neural Networks (MLP, Deep Learning)**            | ใช้โครงข่าย neuron หลายชั้นเพื่อ approximate ฟังก์ชันซับซ้อน                               | ตีความตรง ๆ ยาก (black-box) → ใช้ SHAP, LIME หรือ attention analysis                                                                   | RMSE (เหมาะกับงานที่ต้อง penalize error ใหญ่), MAE, \$R^2\$        |


**สรุปเชิงภาพรวม**

* **ถ้าต้องการตีความง่าย** → Linear Regression, Decision Tree
* **ถ้าต้องการความแม่นยำสูง** → Ensemble (Random Forest, Gradient Boosting) หรือ Neural Networks
* **การประเมิน (Metrics):**

  * ใช้ **MAE** ถ้าอยากวัดความคลาดเคลื่อนเฉลี่ยแบบตรงไปตรงมา
  * ใช้ **RMSE** ถ้าอยากให้ความสำคัญกับ error ขนาดใหญ่
  * ใช้ **\$R^2\$** ถ้าอยากดูว่าสัดส่วนความแปรปรวนที่อธิบายได้มากน้อยแค่ไหน
 

---

### 2. Classification

Classification คือปัญหาที่โมเดลต้อง **ทำนายคลาส (class/label)** ของข้อมูล เช่น email = spam / not spam, ผู้ป่วย = ป่วย / ไม่ป่วย โดยโมเดลเรียนรู้ **boundary** หรือ **probability distribution** ของคลาสจากข้อมูลฝึก (training set)

อัลกอริทึมต่าง ๆ มีพื้นฐานต่างกัน เช่น
* **Linear models (Logistic Regression, Linear SVM):** หาขอบเขตเส้นตรง
* **Tree-based (Decision Tree, Random Forest, Boosting):** หากฎเชิงเงื่อนไข
* **Instance-based (KNN):** ใช้ความใกล้เคียงของข้อมูล
* **Probabilistic (Naïve Bayes):** อิงกฎ Bayes
* **Neural Networks:** สร้าง decision boundary ซับซ้อน

**การประเมิน (Evaluation Metrics)**

ขึ้นกับความสมดุลของข้อมูล (class balance):

* **Accuracy**: สัดส่วนทำนายถูก → เหมาะเมื่อ class balance
* **Precision**: สัดส่วนทำนาย Positive ที่ถูกต้อง → เหมาะในกรณีต้องลด False Positive เช่น Spam Detection
* **Recall**: สัดส่วน Positive ที่ถูกจับได้ → เหมาะในกรณีต้องลด False Negative เช่น การแพทย์ (ห้ามพลาดผู้ป่วยจริง)
* **F1-score**: ค่าเฉลี่ยเชิงฮาร์โมนิกของ Precision และ Recall → ใช้เมื่อ class imbalance
* **Confusion Matrix**: ตารางสรุปผลทำนาย TP, FP, TN, FN
* **ROC-AUC**: วัดคุณภาพการแยกคลาสโดยไม่ขึ้นกับ threshold

**การแปลผลเชิงความหมาย**

* ถ้า **Accuracy = 95%** → ดีถ้า class balance, แต่ถ้า imbalance เช่น 95% = Negative, ทำนาย “Negative เสมอ” ก็ได้ Accuracy 95% เช่นกัน
* ถ้า **Precision = 90%** → แปลว่าในผลที่โมเดลทำนายว่า “Positive” มี 90% ที่ถูกต้องจริง
* ถ้า **Recall = 90%** → แปลว่าโมเดลสามารถตรวจจับ Positive จริงได้ 90%
* ถ้า **F1 = 0.88** → โมเดลมีสมดุลที่ดีระหว่าง Precision และ Recall
* ถ้า **AUC = 0.95** → โมเดลแยก positive กับ negative ได้ดีมาก (ใกล้ 1 = ยอดเยี่ยม, 0.5 = เดาสุ่ม)

**ตารางสรุป Classification Algorithms**

| **ชื่อ**                                            | **ทฤษฎี (หลักการสำคัญ)**                                       | **การแปลผล (Interpretation)**                                       | **การประเมินที่เหมาะสม**                                             |                                          |
| --------------------------------------------------- | -------------------------------------------------------------- | ------------------------------------------------------------------- | -------------------------------------------------------------------- | ---------------------------------------- |
| **Logistic Regression**                             | สร้างสมการเชิงเส้นแล้วผ่าน sigmoid เพื่อให้ได้ \$P(Y=1\| x)\$                                                                | ค่าสัมประสิทธิ์ = log-odds ของการเป็นคลาสเป้าหมาย → แปลได้ตรงไปตรงมา | Accuracy, Precision, Recall, F1, ROC-AUC |
| **Naïve Bayes**                                     | ใช้กฎ Bayes โดยสมมติว่า features เป็นอิสระต่อกัน               | ค่า posterior probability แสดงความเชื่อมั่นว่าข้อมูลอยู่ในคลาสใด    | Accuracy, Precision, Recall, F1 (ดีในงาน text classification)        |                                          |
| **K-Nearest Neighbors (KNN)**                       | ทำนายคลาสจากคลาสของเพื่อนบ้าน \$k\$ จุดที่ใกล้ที่สุด           | เข้าใจง่าย → “สิ่งที่คล้ายกันจะอยู่คลาสเดียวกัน”                    | Accuracy, F1 (sensitive ต่อ imbalance), ROC-AUC                      |                                          |
| **Support Vector Machine (SVM)**                    | หาขอบเขต (hyperplane) ที่ maximize margin ระหว่างคลาส          | ตีความยากใน high dimension แต่ boundary ชัดเจน                      | Accuracy, Precision, Recall, F1, ROC-AUC                             |                                          |
| **Decision Tree**                                   | แบ่งข้อมูลตาม feature ที่ลด impurity มากที่สุด (Gini/Entropy)  | แปลเป็นกฎ IF-THEN ได้ง่าย เช่น “ถ้าอายุ < 30 และรายได้สูง → คลาส A” | Accuracy, Precision, Recall, F1 (ดู Confusion Matrix)                |                                          |
| **Random Forest**                                   | สร้างหลาย tree แบบสุ่มแล้วโหวต → ลด overfitting                | ใช้ Feature Importance อธิบายได้ว่าตัวแปรใดมีผลมากที่สุด            | Accuracy, F1, ROC-AUC                                                |                                          |
| **Gradient Boosting (XGBoost, LightGBM, CatBoost)** | สร้าง tree ต่อเนื่อง แก้ error ของ tree ก่อนหน้า               | ใช้ SHAP values อธิบายผลลัพธ์และความสำคัญของ features               | Accuracy, F1, ROC-AUC (นิยมใน Kaggle)                                |                                          |
| **Neural Networks (MLP, CNN, RNN, Transformers)**   | ใช้โครงข่าย neuron ซับซ้อน สร้าง decision boundary ที่ยืดหยุ่น | ตีความยาก (black-box) → ใช้ LIME/SHAP ช่วยตีความ                    | Accuracy, F1, ROC-AUC (เหมาะกับงานภาพ เสียง ข้อความ)                 |                                          |



**สรุปเชิงภาพรวม**

* ถ้า **ต้องการตีความง่าย** → Logistic Regression, Decision Tree
* ถ้า **ต้องการความแม่นยำสูง** → Ensemble (Random Forest, Gradient Boosting), Neural Networks
* **การเลือก metric:**

  * ใช้ **Accuracy** ถ้า class balance
  * ใช้ **Precision/Recall/F1** ถ้า class imbalance
  * ใช้ **ROC-AUC** ถ้าต้องการดูความสามารถในการแยกคลาสโดยไม่ขึ้นกับ threshold


### 3. Clustering (การจัดกลุ่ม)

Clustering คือการจัดกลุ่มข้อมูลโดยไม่รู้ label ล่วงหน้า (unsupervised learning) เป้าหมายคือให้ข้อมูลที่คล้ายกันอยู่กลุ่มเดียวกัน และข้อมูลที่ต่างกันอยู่คนละกลุ่ม

**การประเมิน**

* **Silhouette Score**: วัดความ “กระชับ” ของกลุ่ม (ใกล้ 1 ดี, ใกล้ -1 แย่)
* **Davies-Bouldin Index (DBI)**: ค่ายิ่งน้อยยิ่งดี
* **Calinski-Harabasz Index (CHI)**: ค่ายิ่งมากยิ่งดี
* **External metrics** (ถ้ามี label จริง): Adjusted Rand Index (ARI), Normalized Mutual Information (NMI)

**การแปลผลเชิงความหมาย**

* กลุ่มที่ได้ต้อง “ตีความเชิงธุรกิจ” เช่น กลุ่มลูกค้า VIP, กลุ่มราคาประหยัด
* ค่า **Silhouette Score = 0.7** → กลุ่มชัดเจนและไม่ overlap กันมาก
* ถ้าได้ cluster ที่ไม่แยกชัดเจน → อาจต้องปรับจำนวน cluster (k) หรือเลือกอัลกอริทึมใหม่

**ตารางสรุป**

| **ชื่อ**                         | **ทฤษฎี**                                                          | **การแปลผล**                                   | **การประเมินที่เหมาะสม**   |
| -------------------------------- | ------------------------------------------------------------------ | ---------------------------------------------- | -------------------------- |
| **K-Means**                      | หาจุด centroid \$k\$ จุดให้ค่า SSE (sum of squared error) ต่ำสุด   | แต่ละ cluster = กลุ่มที่ใกล้ centroid ที่สุด   | Silhouette Score, CHI, DBI |
| **Hierarchical Clustering**      | รวม (agglomerative) หรือแยก (divisive) จุดข้อมูลเป็นลำดับขั้น      | ใช้ dendrogram เพื่อดูความสัมพันธ์ระหว่างกลุ่ม | Silhouette, ARI, NMI       |
| **DBSCAN**                       | สร้าง cluster จากจุดที่หนาแน่น โดยจุดเบาบางเป็น noise              | ดีสำหรับกลุ่มไม่ทรงกลมและมี noise              | Silhouette, DBI            |
| **Gaussian Mixture Model (GMM)** | สมมติว่าข้อมูลมาจากการผสมของหลาย distribution (Gaussian)           | ให้ probability ของการเป็นสมาชิกแต่ละกลุ่ม     | BIC, AIC, Silhouette       |
| **Spectral Clustering**          | ใช้ eigenvectors ของ similarity graph เพื่อลดมิติแล้วทำ clustering | จับโครงสร้างที่ไม่เป็นเส้นตรงได้ดี             | Silhouette, NMI, ARI       |

---

### 4. Dimensionality Reduction / Feature Extraction

กลุ่มนี้ใช้ลดจำนวนมิติของข้อมูล โดยคง “สาระสำคัญ” ไว้ → ช่วยแก้ปัญหา curse of dimensionality, ทำ visualization, และ preprocessing สำหรับโมเดลอื่น ๆ

**การประเมิน**

* **Explained Variance Ratio (สำหรับ PCA)**
* **Reconstruction Error (สำหรับ Autoencoder)**
* **Visualization/Separability**: ใช้กราฟ 2D/3D ตรวจสอบว่าข้อมูลแยกกลุ่มได้ดีหรือไม่

**การแปลผลเชิงความหมาย**

* PCA component 1–2 อธิบาย variance ได้ >70% → เพียงพอสำหรับ visualization
* ถ้า reduction แล้วโมเดล downstream (เช่น classification) ดีขึ้น → แสดงว่าลดมิติได้เหมาะสม

**ตารางสรุป**

| **ชื่อ**                               | **ทฤษฎี**                                                                | **การแปลผล**                                                     | **การประเมินที่เหมาะสม**             |
| -------------------------------------- | ------------------------------------------------------------------------ | ---------------------------------------------------------------- | ------------------------------------ |
| **PCA (Principal Component Analysis)** | หาทิศทาง principal components ที่อธิบาย variance ได้สูงสุด               | Component 1, 2 ใช้อธิบายข้อมูลได้มาก → เหมาะสำหรับ visualization | Explained Variance Ratio             |
| **t-SNE**                              | แปลงข้อมูล high-dimension → low-dimension โดยรักษาความคล้ายกันเชิง local | ดีสำหรับการแสดงผลใน 2D/3D แต่ไม่ดีสำหรับ scaling                 | Visualization, Cluster separability  |
| **UMAP**                               | ใช้ manifold learning ลดมิติ โดยรักษา structure ทั้ง global และ local    | visualization ที่ดีกว่า t-SNE ในหลายกรณี                         | Trustworthiness score, Visualization |
| **LDA (Linear Discriminant Analysis)** | หา projection ที่ maximize class separation                              | ถ้า class แยกออกได้ → โมเดล classification downstream จะดีขึ้น   | Classification accuracy หลังลดมิติ   |
| **Autoencoder**                        | Neural Network ที่บีบอัด (encode) และขยาย (decode) ข้อมูล                | Reconstruction error ต่ำ = representation ดี                     | Reconstruction Error                 |


---

### 5. Anomaly Detection

Anomaly Detection มุ่งหาข้อมูลที่แตกต่างจาก pattern ส่วนใหญ่ เช่น การทุจริตทางการเงิน, การบุกรุกระบบเครือข่าย, sensor error

**การประเมิน**

* **Precision, Recall, F1 (ถ้ามี label)**
* **ROC-AUC, PR-AUC (สำหรับ imbalance)**
* **Unsupervised metrics**: ใช้ reconstruction error, anomaly score distribution

**การแปลผลเชิงความหมาย**

* **High Recall** → ตรวจจับ anomaly ได้ครบ แต่เสี่ยง false alarm
* **High Precision** → แจ้ง anomaly น้อยแต่แม่นยำ
* การเลือก metric ขึ้นกับบริบท เช่น การเงินเน้น Precision, ระบบความปลอดภัยเน้น Recall

**Anomaly Detection**

| **ชื่อ**                         | **ทฤษฎี**                                                                 | **การแปลผล**                                   | **การประเมินที่เหมาะสม**                |
| -------------------------------- | ------------------------------------------------------------------------- | ---------------------------------------------- | --------------------------------------- |
| **Isolation Forest**             | สุ่มสร้าง tree เพื่อตัดข้อมูล → anomaly ใช้ split น้อย                    | ค่า anomaly score สูง → ข้อมูลผิดปกติ          | Precision, Recall, F1, ROC-AUC          |
| **One-Class SVM**                | หาขอบเขต (boundary) รอบข้อมูลปกติ → จุดนอก boundary = anomaly             | anomaly = ข้อมูลที่ไม่เหมือน distribution ปกติ | Precision, Recall, ROC-AUC              |
| **Local Outlier Factor (LOF)**   | เปรียบเทียบ density ของเพื่อนบ้าน → density ต่ำผิดปกติ = anomaly          | ใช้สำหรับ dataset ที่มี local density แตกต่าง  | Precision, Recall, ROC-AUC              |
| **Autoencoder (Anomaly)**        | โมเดลเรียนรู้ reconstruct ข้อมูลปกติ → anomaly = ข้อมูล reconstruct ไม่ดี | Reconstruction error สูง → anomaly             | Reconstruction Error, Precision, Recall |
| **Gaussian Mixture for Outlier** | สร้าง probabilistic model → จุดที่มี likelihood ต่ำ = anomaly             | ความน่าจะเป็นต่ำผิดปกติ                        | Log-likelihood, Precision, Recall       |



## Hands-on A (Regression): ทำนายคะแนนคณิตจากชั่วโมงอ่าน

```python
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

np.random.seed(42)

# สร้างข้อมูลสังเคราะห์
n = 200
hours = np.random.uniform(0, 16, size=n)
noise = np.random.normal(0, 5, size=n)
math = 30 + 4*hours + noise            # คะแนนโดยประมาณ
math = np.clip(math, 0, 100)           # จำกัดช่วงคะแนน 0-100

df_reg = pd.DataFrame({"StudyHours": hours, "Math": math})

# แบ่ง train/test
X = df_reg[["StudyHours"]]
y = df_reg["Math"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

# สร้าง baseline แบบทายค่าเฉลี่ย
y_pred_baseline = np.repeat(ytr.mean(), len(yte))

# โมเดลเชิงเส้น
lr = LinearRegression().fit(Xtr, ytr)
y_pred = lr.predict(Xte)

print("Coefficient (β1):", lr.coef_[0])
print("Intercept (β0):", lr.intercept_)
```
## Hands-on B (Classification): ทำนาย “ผ่าน/ไม่ผ่าน” จากชั่วโมงอ่าน+การเข้าเรียน
```python
# ===== Classification: Pass/Fail from StudyHours & Attendance =====
from sklearn.linear_model import LogisticRegression

np.random.seed(7)
n = 400
hours = np.random.uniform(0, 16, size=n)
attend = np.random.uniform(0.5, 1.0, size=n)   # อัตราเข้าเรียน 50%–100%
# สร้างฉลาก: ผ่าน (1) ถ้า 0.35*hours + 0.8*attend > เกณฑ์ (มี noise)
score = 0.35*hours + 0.8*attend + np.random.normal(0, 0.1, size=n)
y = (score > 0.9).astype(int)

df_clf = pd.DataFrame({"StudyHours": hours, "Attendance": attend, "Pass": y})

X = df_clf[["StudyHours", "Attendance"]]
y = df_clf["Pass"]

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=7, stratify=y)

# baseline: ทายคลาสที่มากสุด
majority = int(ytr.value_counts().idxmax())
y_pred_base = np.repeat(majority, len(yte))

# Logistic Regression
logreg = LogisticRegression().fit(Xtr, ytr)
y_pred = logreg.predict(Xte)
y_proba = logreg.predict_proba(Xte)[:,1]

print("Coefficients (w):", dict(zip(X.columns, logreg.coef_[0])))
print("Intercept (b):", logreg.intercept_[0])
```
## Model Evaluation
### Hands-on A (Regression Metrics + กราฟ Residual)

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ประเมิน baseline vs linear regression
mae_base = mean_absolute_error(yte, y_pred_baseline)
rmse_base = mean_squared_error(yte, y_pred_baseline, squared=False)

mae_lr = mean_absolute_error(yte, y_pred)
rmse_lr = mean_squared_error(yte, y_pred, squared=False)
r2_lr = r2_score(yte, y_pred)

print(f"Baseline  MAE={mae_base:.2f}, RMSE={rmse_base:.2f}")
print(f"LinearReg MAE={mae_lr:.2f}, RMSE={rmse_lr:.2f}, R^2={r2_lr:.3f}")

# Residual plot
resid = yte - y_pred
plt.scatter(y_pred, resid)
plt.axhline(0, color="black", linewidth=1)
plt.xlabel("Predicted Math")
plt.ylabel("Residual (y - ŷ)")
plt.title("Residual Plot (ควรกระจายรอบศูนย์แบบไร้รูปแบบ)")
plt.show()
```
การตีความ: โมเดลดีควร MAE/RMSE ต่ำกว่า baseline และกราฟ residual ควรไม่มีรูปแบบชัดเจน (สัญญาณของความเป็นเส้นตรงและความแปรปรวนคงที่)

### Hands-on B (Classification Metrics + Confusion Matrix + ROC-AUC)
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, RocCurveDisplay

# Baseline vs Logistic Regression
acc_base = accuracy_score(yte, y_pred_base)
acc = accuracy_score(yte, y_pred)
prec = precision_score(yte, y_pred, zero_division=0)
rec = recall_score(yte, y_pred, zero_division=0)
f1 = f1_score(yte, y_pred, zero_division=0)
auc = roc_auc_score(yte, y_proba)

print(f"Baseline Accuracy: {acc_base:.3f}")
print(f"LogReg   Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}, ROC-AUC: {auc:.3f}")

# Confusion Matrix
cm = confusion_matrix(yte, y_pred)
print("Confusion Matrix:\n", cm)

# ROC Curve
RocCurveDisplay.from_predictions(yte, y_proba)
plt.title("ROC Curve (พื้นที่ใต้โค้งมาก → แยกคลาสได้ดี)")
plt.show()
```

การตีความ:
- nหาก Accuracy/Precision/Recall/F1 สูงกว่า baseline แสดงว่าโมเดลมีประโยชน์กว่าการเดาสุ่ม/เดาคลาสที่มากสุด
- ROC-AUC เข้าใกล้ 1 แปลว่าโมเดลแยกคลาสได้ดีในหลาย threshold



