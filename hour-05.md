# Hour 5 : Usecase
## Case 1: พยากรณ์ยอดขายไอศกรีมรายวัน


1. เป้าหมาย (Objective)

ทำนาย **ยอดขายรายวัน (sales\_thb)** โดยใช้ข้อมูลจากสภาพอากาศ, ปัจจัยการตลาด และพฤติกรรมลูกค้า

* **Target (y):** sales\_thb (บาท)
* **Features (X):** temperature\_C, humidity\_pct, month, is\_weekend, is\_holiday, promo\_budget\_thb, foot\_traffic, prior\_day\_sales, date\_index


2. การวิเคราะห์เบื้องต้น (EDA)

    1. **Distribution ของยอดขาย:**

    * ใช้ Histogram/KDE → ตรวจว่ามี skew หรือ outlier หรือไม่
    * ถ้า skew มาก อาจพิจารณา log-transform

    2. **ความสัมพันธ์ระหว่าง feature กับยอดขาย:**

    * **temperature\_C vs sales\_thb:** คาดว่าเป็นโค้ง (อากาศเย็นขายน้อย, ร้อนจัดขายมาก → non-linear)
    * **foot\_traffic vs sales\_thb:** น่าจะสัมพันธ์เชิงเส้น (คนเข้าร้านมาก → ยอดขายสูง)
    * **promo\_budget\_thb vs sales\_thb:** ตรวจความสัมพันธ์, อาจมี diminishing return

    3. **Seasonality/Trend:**

    * month, is\_holiday, is\_weekend → ตรวจว่าแต่ละกลุ่มมีค่าเฉลี่ยยอดขายต่างกันหรือไม่
    * prior\_day\_sales → ใช้จับ autocorrelation


3. โมเดลที่เลือก (Modeling Strategy)

    1. **Baseline:**

    * โมเดลง่ายที่สุด = ทำนายค่าเฉลี่ยยอดขายทั้งหมด
    * ใช้เป็นเกณฑ์เปรียบเทียบ

    2. **Linear Regression:**

    * เริ่มต้นด้วยสมการเชิงเส้น $\hat{y} = \beta\_0 + \beta\_1 x\_1 + \cdots + \beta\_k x\_k$
    * เหมาะสำหรับฟีเจอร์ที่มีความสัมพันธ์เชิงเส้นกับยอดขาย

    3. **Polynomial Features (เฉพาะ temperature\_C):**

    * เพิ่ม $temperature^2$ เพื่อตรวจ non-linearity

    4. **Tree-based Models (Decision Tree, Random Forest):**

    * ใช้ถ้า EDA พบ non-linear และ interaction ระหว่าง features ชัดเจน


4. การประเมินผล (Evaluation)

ใช้ **Train/Test Split (เช่น 80/20)** แล้ววัดผลด้วย

* **MAE (Mean Absolute Error):** ค่า error โดยเฉลี่ยในหน่วยบาท → แสดงความเพี้ยนที่เข้าใจง่าย
* **RMSE (Root Mean Squared Error):** ลงโทษ error ขนาดใหญ่ → ใช้ดูว่ามี outlier หรือจุดทำนายพลาดหนักหรือไม่
* **$R^2$ (Coefficient of Determination):** วัดว่ายอดขายที่โมเดลอธิบายได้กี่ %

เพิ่มเติม:

* **Residual Plot:** ตรวจสอบว่า error กระจายตัวแบบ random หรือมี pattern (ถ้ามี → โมเดลยังไม่เหมาะสม)


5. การแปลผลเชิงความหมาย (Interpretation)

* ถ้า **MAE = 500 บาท** → ทำนายยอดขายคลาดเคลื่อนเฉลี่ย \~500 บาทต่อวัน
* ถ้า **RMSE = 700 บาท** และสูงกว่า MAE มาก → แสดงว่ามีบางวัน error ใหญ่มาก (เช่น วันหยุดยาว/พิเศษ)
* ถ้า **$R^2 = 0.85\$** → โมเดลอธิบายความแปรปรวนของยอดขายได้ 85% ถือว่าดีมาก
* **การตีความ feature:**

  * $\beta\_{temperature} > 0$ → อุณหภูมิสูงขึ้น → ยอดขายสูงขึ้น
  * $\beta\_{promo\_budget} > 0$ → งบโปรโมชั่นมาก → ยอดขายสูงขึ้น
  * $\beta\_{is\_weekend}\$ บวก → วันหยุดขายดีกว่าวันธรรมดา

**สรุป Workflow**

1. **EDA:** ตรวจ distribution, correlation, seasonality
2. **Baseline:** ค่าเฉลี่ยยอดขาย → ใช้เป็นจุดเปรียบเทียบ
3. **Model:** Linear Regression → Polynomial (temp²) → Random Forest (ถ้า non-linear ชัดเจน)
4. **Evaluation:** MAE, RMSE, R² + Residual Plot
5. **Interpretation:** ตีความเชิงธุรกิจ (เช่น อากาศร้อน + งบโปรสูง = ยอดขายพุ่ง)


## Case 2: ทำนายการสมัครใช้งานแอป (Signup Conversion)

**เป้าหมาย**

ทำนายว่า session นี้ สมัครใช้งานสำเร็จ (1) หรือไม่ (0)

**ข้อมูล (พร้อมใช้)**
- ฟีเจอร์: session_duration_s, pages_viewed, device(mobile/desktop/tablet), referral(ads/seo/email/direct), country(TH/VN/ID/SG), is_weekend, clicks_cta, returning_user
- เลเบล: signed_up ∈ {0,1}
- ขนาดตัวอย่าง: 2,000 แถว

**เมตริกที่ใช้**
- Accuracy, Precision/Recall/F1, และ ROC-AUC

**แนววิเคราะห์**
1. EDA: อัตรา signup โดย device/referral, distribution ของเวลาบนหน้า, correlation ระหว่างเชิงปริมาณ
2. Baseline: ทายคลาสที่มากสุด
3. โมเดล: Logistic Regression (เริ่มต้น) → ลอง Tree/RandomForest + one-hot สำหรับตัวแปรหมวดหมู่
4. ประเมิน: Confusion Matrix, ROC-AUC, วิเคราะห์ threshold (Precision-Recall trade-off)


### Regression (Linear Regression)
```python 
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
```

### Classification (Logistic Regression)
```python
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
```



### กรอบแนวคิด

Key message → Evidence → Visual → Annotation → Decision
1. นิยาม “ประเด็นหลัก (Key message)” ให้ชัดเจน
2. เลือกกราฟที่ “รองรับข้อความนั้น” โดยตรง
3. ลดสิ่งรบกวน (declutter) และ ไฮไลต์ จุดสำคัญ
4. ปิดท้ายด้วย ข้อเสนอการตัดสินใจ และความเสี่ยง


### A) กรณี Regression: ยอดขายไอศกรีม (Ice-cream Sales)

1) ข้อความหลัก (ตัวอย่าง)
	- “สุดสัปดาห์/วันหยุด ทำให้ยอดขายเฉลี่ยเพิ่มขึ้นอย่างมีนัยสำคัญ และยอดขายสูงสุดเกิดในช่วงอุณหภูมิใกล้ ~33 °C (ร้อนเกินไปยอดขายเริ่มลด)”

2) รูปช่วยเล่าเรื่อง (แนะนำ)
	- Bar chart: ยอดขายเฉลี่ยแยกตาม is_weekend, is_holiday (เห็น uplift ชัดเจน)
	- Scatter + เส้นโค้งกำลังสอง: temperature_C vs sales_thb เพื่อสื่อ “จุดเหมาะสม” ของอุณหภูมิ

3) โค้ด (Notebook-ready; ใช้ matplotlib ทั้งหมด)
```python
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
```

4) ประโยคสรุปบนสไลด์ (ตัวอย่าง)
	- “ขายดีขึ้นชัดเจนใน ส–อา และวันหยุด (+xxx บาทเฉลี่ย/วัน) → ควรเพิ่มสต็อก/พนักงานช่วงดังกล่าว”
	- “ยอดขายสูงสุดใกล้ ~33 °C → แนะนำแคมเปญเมื่อพยากรณ์อากาศเข้าโซนนี้”

## B) กรณี Classification: สมัครใช้งานแอป (App Signup Conversion)

1) ข้อความหลัก (ตัวอย่าง)
	- “เวอร์ชันปุ่ม B เพิ่ม CR เมื่อเทียบกับ A และโมเดล Logistic แยกคลาสได้ดี (ROC-AUC สูง) โดย SEO/ผู้ใช้เดิม มีโอกาสสมัครสูงกว่ากลุ่มอื่น”

2) รูปช่วยเล่าเรื่อง (แนะนำ)
	- Bar chart: Conversion rate ของ variant A vs B (ใส่ตัวเลขบนแท่ง)
	- Confusion matrix (หลังตั้ง threshold มาตรฐาน 0.5) และ ROC curve (แสดง AUC)

3) โค้ด (Notebook-ready)
```python 
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
```

4) ประโยคสรุปบนสไลด์ (ตัวอย่าง)
	- “B ชนะ A (CR สูงกว่า X.X จุดร้อยละ); โมเดลช่วยบ่งกลุ่มโอกาสสูง (AUC ≈ 0.8+) → โฟกัส SEO/ผู้ใช้เดิม”
	- “คำแนะนำ: Rollout แบบ phased + เฝ้าระวัง guardrails (เช่น complaint rate)”



