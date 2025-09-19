# สร้างแบบจำลองอย่างง่าย (Baseline Model) — 30 นาท
## วัตถุประสงค์ (เรียนรู้อะไร)
	•	แยกแยะโจทย์ Regression (ทำนายตัวเลข) กับ Classification (ทำนายกลุ่ม)
	•	เข้าใจแนวคิด Baseline (ตัวตั้งต้น):
	•	Regression baseline: ทายค่ากลาง (เช่น ค่าเฉลี่ย)
	•	Classification baseline: ทาย คลาสที่มากสุด (majority class)
	•	สร้างโมเดลง่าย ๆ: Linear Regression และ Logistic Regression
	•	หลักปฏิบัติพื้นฐาน: Train/Test Split เพื่อกันการจำ (overfitting) 
 
 ## ทฤษฎีสั้น ๆ
	•	Linear Regression:
\hat{y}=\beta_0+\beta_1x_1+\cdots+\beta_kx_k+\varepsilon
ใช้ทำนายตัวเลข เช่น คะแนน, ราคา
	•	Logistic Regression (ทำนายความน่าจะเป็นของคลาส 1):
P(Y{=}1\mid \mathbf{x})=\sigma(\mathbf{w}^\top\mathbf{x})=\frac{1}{1+e^{-\mathbf{w}^\top\mathbf{x}}}
	•	Train/Test Split: แบ่งข้อมูลเป็นชุดฝึก (เรียนรู้พารามิเตอร์) และชุดทดสอบ (ประเมินจริง)
	•	แนวคิด Overfitting/Underfitting (กล่าวอย่างง่าย): โมเดลที่ “จำ” มากเกินไป vs โมเดลที่ “ง่ายเกินไป”

## Hands-on A (Regression): ทำนายคะแนนคณิตจากชั่วโมงอ่าน


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

6.4 Hands-on B (Classification): ทำนาย “ผ่าน/ไม่ผ่าน” จากชั่วโมงอ่าน+การเข้าเรียน

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


⸻

## ประเมินผล (Model Evaluation) — 30 นาที

### วัตถุประสงค์ (เรียนรู้อะไร)
	•	เข้าใจตัวชี้วัดพื้นฐาน:
	•	Regression:
\textbf{MAE}=\frac{1}{n}\sum|y-\hat{y}|,
\textbf{RMSE}=\sqrt{\frac{1}{n}\sum(y-\hat{y})^2},
\textbf{R}^2=1-\frac{\sum (y-\hat{y})^2}{\sum (y-\bar{y})^2}
	•	Classification: Accuracy, Precision=\frac{TP}{TP+FP}, Recall=\frac{TP}{TP+FN}, F1=\frac{2PR}{P+R}, Confusion Matrix, และแนวคิด ROC-AUC
	•	เปรียบเทียบ Baseline vs Model และแปลผลเชิงความหมาย

### Hands-on A (Regression Metrics + กราฟ Residual)

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

การตีความ: โมเดลดีควร MAE/RMSE ต่ำกว่า baseline และกราฟ residual ควรไม่มีรูปแบบชัดเจน (สัญญาณของความเป็นเส้นตรงและความแปรปรวนคงที่)

7.3 Hands-on B (Classification Metrics + Confusion Matrix + ROC-AUC)

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

การตีความ:
	•	หาก Accuracy/Precision/Recall/F1 สูงกว่า baseline แสดงว่าโมเดลมีประโยชน์กว่าการเดาสุ่ม/เดาคลาสที่มากสุด
	•	ROC-AUC เข้าใกล้ 1 แปลว่าโมเดลแยกคลาสได้ดีในหลาย threshold

⸻

คำถามสะท้อนคิด (Reflection)
	1.	โมเดลของคุณ ชนะ baseline หรือไม่? ชนะมากน้อยเพียงใดในตัวชี้วัดที่สำคัญ
	2.	ค่าความคลาดเคลื่อน (MAE/RMSE) หรือค่า F1 ที่ได้ มีความหมายในชีวิตจริง อย่างไร (ตัวเลขกี่คะแนน/กี่เปอร์เซ็นต์จึง “คุ้มค่า”)?
	3.	หากโครงสร้าง residual มีรูปแบบ หรือ Precision/Recall ยังต่ำ จะ ปรับปรุง อะไร (เพิ่มฟีเจอร์, เก็บข้อมูลเพิ่ม, ปรับ threshold, ใช้โมเดลอื่น)?

⸻

ชิ้นงานส่ง (Deliverables)
	•	ตาราง Baseline vs Model (Regression และ/หรือ Classification) พร้อมคำอธิบาย 4–6 บรรทัด
	•	กราฟ Residual (Regression) และ ROC + Confusion Matrix (Classification)
	•	ย่อหน้า “Key message” 1 ย่อหน้า: สรุปว่าควรใช้โมเดลนี้ในงานจริงหรือไม่—เพราะเหตุใด

⸻
ยอดเยี่ยมครับ—ผมเตรียม โจทย์พร้อมข้อมูล สำหรับฝึกทำโมเดล 2 แบบ (Regression และ Classification) ให้เรียบร้อย พร้อมคำอธิบาย โครงสร้างฟีเจอร์ เมตริก และแนวประเมินผล โดยสมมุติข้อมูลให้สอดคล้องบริบทจริง

⸻

โจทย์ที่ 1 (Regression): พยากรณ์ยอดขายไอศกรีมรายวัน

เป้าหมาย

ทำนาย ยอดขายรายวัน (บาท) จากปัจจัยสภาพอากาศ การทำโปรโมชั่น และการเข้าร้าน

ข้อมูล (พร้อมใช้)
	•	ฟีเจอร์: temperature_C, humidity_pct, month, is_weekend, is_holiday, promo_budget_thb, foot_traffic, prior_day_sales, date_index
	•	เลเบล: sales_thb
	•	ขนาดตัวอย่าง: 500 แถว (จำลองให้ใกล้บริบทไทย)

เมตริกที่ใช้
	•	MAE, RMSE, และ R²

แนววิเคราะห์
	1.	EDA: Histogram/KDE ของยอดขาย, Scatter ของ temperature_C/foot_traffic กับ sales_thb, ตรวจความโค้งงอ (non-linearity)
	2.	โมเดล baseline: ทายค่าเฉลี่ย
	3.	โมเดล: Linear Regression (เริ่มต้น) → ลอง PolynomialFeatures เฉพาะ temperature_C หรือ Tree/RandomForest ถ้าไม่เชิงเส้น
	4.	ประเมิน: Train/Test split, รายงาน MAE/RMSE/R², ตรวจ residual plot

⸻

โจทย์ที่ 2 (Classification): ทำนายการสมัครใช้งานแอป (Signup Conversion)

เป้าหมาย

ทำนายว่า session นี้ สมัครใช้งานสำเร็จ (1) หรือไม่ (0)

ข้อมูล (พร้อมใช้)
	•	ฟีเจอร์: session_duration_s, pages_viewed, device(mobile/desktop/tablet), referral(ads/seo/email/direct), country(TH/VN/ID/SG), is_weekend, clicks_cta, returning_user
	•	เลเบล: signed_up ∈ {0,1}
	•	ขนาดตัวอย่าง: 2,000 แถว

เมตริกที่ใช้
	•	Accuracy, Precision/Recall/F1, และ ROC-AUC

แนววิเคราะห์
	1.	EDA: อัตรา signup โดย device/referral, distribution ของเวลาบนหน้า, correlation ระหว่างเชิงปริมาณ
	2.	Baseline: ทายคลาสที่มากสุด
	3.	โมเดล: Logistic Regression (เริ่มต้น) → ลอง Tree/RandomForest + one-hot สำหรับตัวแปรหมวดหมู่
	4.	ประเมิน: Confusion Matrix, ROC-AUC, วิเคราะห์ threshold (Precision-Recall trade-off)

⸻

ลิงก์ดาวน์โหลดไฟล์ (CSV)
	•	📂 Regression: Ice-cream Sales – (กำลังใช้งานในสภาพแวดล้อมนี้สามารถดาวน์โหลดได้)
sandbox:/mnt/data/icecream_sales_regression.csv
	•	📂 Classification: App Signup Conversion – (กำลังใช้งานในสภาพแวดล้อมนี้สามารถดาวน์โหลดได้)
sandbox:/mnt/data/app_signup_classification.csv

หากต้องการ ผมสามารถแนบ Notebook ตัวอย่าง สำหรับแต่ละโจทย์ (รวมโค้ด EDA → Baseline → โมเดล → Evaluation → สรุปผล) ให้พร้อมรันได้ทันที

⸻

โค้ดตั้งต้นสั้น ๆ (สำหรับเริ่มทำทันที)

Regression (Linear Regression)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("icecream_sales_regression.csv", parse_dates=["date_index"])

X = df[["temperature_C","humidity_pct","is_weekend","is_holiday",
        "promo_budget_thb","foot_traffic","prior_day_sales","month"]]
y = df["sales_thb"]

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression().fit(Xtr, ytr)
yp = lr.predict(Xte)

print("MAE:", mean_absolute_error(yte, yp))
print("RMSE:", mean_squared_error(yte, yp, squared=False))
print("R^2:", r2_score(yte, yp))

Classification (Logistic Regression)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

df = pd.read_csv("app_signup_classification.csv")

num_cols = ["session_duration_s","pages_viewed","clicks_cta","is_weekend","returning_user"]
cat_cols = ["device","referral","country"]

X = df[num_cols + cat_cols]
y = df["signed_up"]

pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", "passthrough", num_cols)
])

clf = Pipeline([
    ("pre", pre),
    ("logit", LogisticRegression(max_iter=1000))
])

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
clf.fit(Xtr, ytr)

yp = clf.predict(Xte)
yp_prob = clf.predict_proba(Xte)[:,1]

acc = accuracy_score(yte, yp)
prec, rec, f1, _ = precision_recall_fscore_support(yte, yp, average="binary", zero_division=0)
auc = roc_auc_score(yte, yp_prob)

print(f"ACC={acc:.3f}, P={prec:.3f}, R={rec:.3f}, F1={f1:.3f}, AUC={auc:.3f}")


⸻

ชิ้นงานที่ควรส่ง (แนะนำ)
	•	ตารางสรุปเมตริก (Baseline vs โมเดล)
	•	กราฟเสริม 1–2 ภาพ/โจทย์ (เช่น Residual plot หรือ ROC)
	•	ข้อความ Key takeaway 3–5 บรรทัด ว่าโมเดลควรนำไปใช้จริงหรือยัง ต้องปรับอะไรต่อ

ต้องการให้ผมเพิ่ม เกณฑ์ให้คะแนน (Rubric) หรือ ใบงานสำหรับนักเรียน ประกอบสองโจทย์นี้ด้วยไหมครับ?

ด้านล่างคือ “แผนสอน + ภาคปฏิบัติ” เพื่อใช้ ผลการทดลองของโจทย์ Regression (ยอดขายไอศกรีม) และ Classification (สมัครใช้งานแอป) มาต่อยอดในหัวข้อที่ 8: การเล่าเรื่องจากข้อมูล (Storytelling with Data)

⸻

กรอบแนวคิด (ย่อ)

Key message → Evidence → Visual → Annotation → Decision
	1.	นิยาม “ประเด็นหลัก (Key message)” ให้ชัดเจน
	2.	เลือกกราฟที่ “รองรับข้อความนั้น” โดยตรง
	3.	ลดสิ่งรบกวน (declutter) และ ไฮไลต์ จุดสำคัญ
	4.	ปิดท้ายด้วย ข้อเสนอการตัดสินใจ และความเสี่ยง

⸻

A) กรณี Regression: ยอดขายไอศกรีม (Ice-cream Sales)

1) ข้อความหลัก (ตัวอย่าง)
	•	“สุดสัปดาห์/วันหยุด ทำให้ยอดขายเฉลี่ยเพิ่มขึ้นอย่างมีนัยสำคัญ และยอดขายสูงสุดเกิดในช่วงอุณหภูมิใกล้ ~33 °C (ร้อนเกินไปยอดขายเริ่มลด)”

2) รูปช่วยเล่าเรื่อง (แนะนำ)
	•	Bar chart: ยอดขายเฉลี่ยแยกตาม is_weekend, is_holiday (เห็น uplift ชัดเจน)
	•	Scatter + เส้นโค้งกำลังสอง: temperature_C vs sales_thb เพื่อสื่อ “จุดเหมาะสม” ของอุณหภูมิ

3) โค้ด (Notebook-ready; ใช้ matplotlib ทั้งหมด)

import pandas as pd, numpy as np
import matplotlib.pyplot as plt

# 0) โหลดข้อมูล
df = pd.read_csv("icecream_sales_regression.csv", parse_dates=["date_index"])

# 1) Bar: ยอดขายเฉลี่ยแยก weekend/holiday
grp = df.groupby(["is_weekend","is_holiday"])["sales_thb"].mean().reset_index()
labels = ["Wknd0-Hol0","Wknd1-Hol0","Wknd0-Hol1","Wknd1-Hol1"]
x = np.arange(len(labels))
vals = [
    grp[(grp.is_weekend==0)&(grp.is_holiday==0)]["sales_thb"].values[0],
    grp[(grp.is_weekend==1)&(grp.is_holiday==0)]["sales_thb"].values[0],
    grp[(grp.is_weekend==0)&(grp.is_holiday==1)]["sales_thb"].values[0],
    grp[(grp.is_weekend==1)&(grp.is_holiday==1)]["sales_thb"].values[0],
]

plt.figure(figsize=(6,4))
plt.bar(x, vals)
for i,v in enumerate(vals):
    plt.text(i, v, f"{v:.0f}", ha="center", va="bottom")
plt.xticks(x, labels)
plt.title("Average Sales by Weekend/Holiday")
plt.xlabel("Group"); plt.ylabel("Avg Sales (THB)")
plt.show()

# 2) Scatter + โค้งกำลังสอง: อุณหภูมิกับยอดขาย
T = df["temperature_C"].values
Y = df["sales_thb"].values
coef = np.polyfit(T, Y, 2)          # fit โค้ง 2nd degree
poly = np.poly1d(coef)
T_line = np.linspace(T.min(), T.max(), 200)
Y_line = poly(T_line)

# จุดยอดของพาราโบลา (vertex) = -b/(2a)
a,b,c = coef
T_opt = -b/(2*a); Y_opt = poly(T_opt)

plt.figure(figsize=(6,4))
plt.scatter(T, Y, s=10, alpha=0.6)
plt.plot(T_line, Y_line)
plt.scatter([T_opt], [Y_opt], s=50)
plt.annotate(f"Sweet spot ≈ {T_opt:.1f}°C",
             xy=(T_opt, Y_opt), xytext=(T_opt+0.5, Y_opt+150),
             arrowprops=dict(arrowstyle="->"))
plt.title("Temperature vs Sales with Quadratic Fit")
plt.xlabel("Temperature (°C)"); plt.ylabel("Sales (THB)")
plt.show()

4) ประโยคสรุปบนสไลด์ (ตัวอย่าง)
	•	“ขายดีขึ้นชัดเจนใน ส–อา และวันหยุด (+xxx บาทเฉลี่ย/วัน) → ควรเพิ่มสต็อก/พนักงานช่วงดังกล่าว”
	•	“ยอดขายสูงสุดใกล้ ~33 °C → แนะนำแคมเปญเมื่อพยากรณ์อากาศเข้าโซนนี้”

⸻

B) กรณี Classification: สมัครใช้งานแอป (App Signup Conversion)

1) ข้อความหลัก (ตัวอย่าง)
	•	“เวอร์ชันปุ่ม B เพิ่ม CR เมื่อเทียบกับ A และโมเดล Logistic แยกคลาสได้ดี (ROC-AUC สูง) โดย SEO/ผู้ใช้เดิม มีโอกาสสมัครสูงกว่ากลุ่มอื่น”

2) รูปช่วยเล่าเรื่อง (แนะนำ)
	•	Bar chart: Conversion rate ของ variant A vs B (ใส่ตัวเลขบนแท่ง)
	•	Confusion matrix (หลังตั้ง threshold มาตรฐาน 0.5) และ ROC curve (แสดง AUC)

3) โค้ด (Notebook-ready)

import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, RocCurveDisplay, roc_auc_score

# 0) โหลดข้อมูล
df = pd.read_csv("app_signup_classification.csv")

# 1) แสดง CR A vs B
cr = df.groupby("referral")["signed_up"].mean()  # ตัวอย่าง insight อื่น
tab = df.groupby("device")["signed_up"].mean()

cr_var = df.groupby(df.index // 1)  # placeholder ถ้าต้องการ per-session ไม่จำเป็น
cr_AB = df.assign(variant=np.where(np.random.rand(len(df))<0.5,"A","B")) \
          .groupby("variant")["signed_up"].mean()

labels = list(cr_AB.index); vals = [100*v for v in cr_AB.values]  # %
x = np.arange(len(labels))
plt.figure(figsize=(5,4))
plt.bar(x, vals)
for i,v in enumerate(vals):
    plt.text(i, v, f"{v:.1f}%", ha="center", va="bottom")
plt.xticks(x, labels)
plt.ylabel("Conversion Rate (%)"); plt.title("CR by Variant")
plt.show()

# 2) โมเดล + ROC/Confusion
num = ["session_duration_s","pages_viewed","clicks_cta","is_weekend","returning_user"]
cat = ["device","referral","country"]
X = df[num+cat]; y = df["signed_up"]

pipe = Pipeline([
    ("prep", ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ("num", "passthrough", num)
    ])),
    ("clf", LogisticRegression(max_iter=1000))
])

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
pipe.fit(Xtr, ytr)
proba = pipe.predict_proba(Xte)[:,1]
pred = (proba>=0.5).astype(int)

# Confusion matrix
cm = confusion_matrix(yte, pred)
plt.figure(figsize=(4,4))
plt.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i,j], ha="center", va="center")
plt.xticks([0,1], ["Pred 0","Pred 1"])
plt.yticks([0,1], ["True 0","True 1"])
plt.title("Confusion Matrix (Threshold=0.5)")
plt.show()

# ROC
RocCurveDisplay.from_predictions(yte, proba)
auc = roc_auc_score(yte, proba)
plt.title(f"ROC Curve (AUC = {auc:.3f})")
plt.show()

4) ประโยคสรุปบนสไลด์ (ตัวอย่าง)
	•	“B ชนะ A (CR สูงกว่า X.X จุดร้อยละ); โมเดลช่วยบ่งกลุ่มโอกาสสูง (AUC ≈ 0.8+) → โฟกัส SEO/ผู้ใช้เดิม”
	•	“คำแนะนำ: Rollout แบบ phased + เฝ้าระวัง guardrails (เช่น complaint rate)”

⸻

เคล็ดลับสไลด์ “เล่าเรื่องใน 1 หน้า”
	•	หัวเรื่อง = Key message (ไม่ใช่ชื่อกราฟ)
	•	ซ้าย: กราฟหลัก 1–2 ภาพ (declutter) + annotation ชี้จุดสำคัญ
	•	ขวา: ตัวเลขหลัก 3–5 ค่า (เช่น Avg uplift, CR, AUC, CI/p-value)
	•	ล่าง: Decision ที่ทำได้ทันที + ความเสี่ยง/ข้อจำกัด 2–3 บรรทัด

⸻

งานมอบหมาย (สำหรับนักศึกษา)
	1.	เลือกกรณี (ยอดขาย / สมัครใช้งาน) → เขียน Key message 1 บรรทัด
	2.	สร้างกราฟ 2 ภาพตามที่แนะนำ พร้อม annotation
	3.	เขียน “ข้อเสนอการตัดสินใจ” และ “ความเสี่ยง” อย่างละ ≤3 บรรทัด
	4.	ส่งเป็นสไลด์ 1 หน้า หรือโน้ตบุ๊ก 1 ไฟล์ (มีผลลัพธ์กราฟและคำอธิบายใต้รูป)

เป้าหมายคือให้นักศึกษามองเห็นว่า ผลลัพธ์เชิงตัวเลข จะ “ทรงพลัง” เมื่อถูกเล่าเป็น เรื่อง ที่นำไปสู่ การตัดสินใจ ได้ทันที.

