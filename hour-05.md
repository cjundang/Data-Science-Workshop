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

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------
# 1. สร้างข้อมูลสมมุติ (200 records)
# -----------------------------
np.random.seed(42)
n = 200
date_index = pd.date_range("2023-01-01", periods=n, freq="D")

temperature = np.random.normal(30, 5, n)          # อุณหภูมิ (°C)
humidity = np.random.uniform(40, 90, n)           # ความชื้น (%)
month = date_index.month
is_weekend = (date_index.weekday >= 5).astype(int)
is_holiday = np.random.binomial(1, 0.1, n)        # 10% เป็นวันหยุดพิเศษ
promo_budget = np.random.choice([0, 500, 1000, 2000], n, p=[0.4, 0.3, 0.2, 0.1])
foot_traffic = np.random.poisson(lam=200, size=n) # จำนวนลูกค้าเข้าร้าน
prior_day_sales = np.random.normal(5000, 1000, n)

# ฟังก์ชันสร้างยอดขายจริง (target)
sales = (
    2000
    + temperature * 120
    - humidity * 15
    + promo_budget * 1.5
    + foot_traffic * 10
    + prior_day_sales * 0.4
    + is_weekend * 500
    + is_holiday * 1000
    + np.random.normal(0, 800, n)  # noise
)

df = pd.DataFrame({
    "date_index": date_index,
    "temperature_C": temperature,
    "humidity_pct": humidity,
    "month": month,
    "is_weekend": is_weekend,
    "is_holiday": is_holiday,
    "promo_budget_thb": promo_budget,
    "foot_traffic": foot_traffic,
    "prior_day_sales": prior_day_sales,
    "sales_thb": sales
})

print(df.head())

# -----------------------------
# 2. EDA เบื้องต้น
# -----------------------------
plt.figure(figsize=(10,4))
df["sales_thb"].hist(bins=20)
plt.title("Distribution of Daily Ice Cream Sales")
plt.xlabel("Sales (THB)")
plt.ylabel("Frequency")
plt.show()

plt.scatter(df["temperature_C"], df["sales_thb"], alpha=0.6)
plt.title("Temperature vs Sales")
plt.xlabel("Temperature (°C)")
plt.ylabel("Sales (THB)")
plt.show()

plt.scatter(df["foot_traffic"], df["sales_thb"], alpha=0.6, color="orange")
plt.title("Foot Traffic vs Sales")
plt.xlabel("Foot Traffic")
plt.ylabel("Sales (THB)")
plt.show()

# -----------------------------
# 3. สร้างโมเดล (Baseline vs Linear Regression)
# -----------------------------
X = df[["temperature_C","humidity_pct","month","is_weekend","is_holiday",
        "promo_budget_thb","foot_traffic","prior_day_sales"]]
y = df["sales_thb"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# baseline = ค่าเฉลี่ย
baseline_pred = np.repeat(y_train.mean(), len(y_test))

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

# -----------------------------
# 4. การประเมิน
# -----------------------------
def evaluate(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} -> MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.3f}")

evaluate(y_test, baseline_pred, "Baseline")
evaluate(y_test, y_pred, "Linear Regression")

# -----------------------------
# 5. Residual Plot
# -----------------------------
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.title("Residual Plot")
plt.xlabel("Predicted Sales")
plt.ylabel("Residuals")
plt.show()
```


## Case 2: Signup Conversion Prediction

### 1. เป้าหมาย (Objective)

* ทำนายว่าผู้ใช้ใน session จะ **สมัครใช้งาน (signed\_up=1)** หรือไม่ (signed\_up=0)
* ใช้ฟีเจอร์ด้านพฤติกรรมการใช้งาน, แหล่ง referral, device และสถานะผู้ใช้



### 2. ข้อมูล (Dataset)

* **Features (X):**

  * *session\_duration\_s:* เวลาที่อยู่ใน session
  * *pages\_viewed:* จำนวนหน้าที่เปิด
  * *device:* mobile/desktop/tablet
  * *referral:* ads/seo/email/direct
  * *country:* TH/VN/ID/SG
  * *is\_weekend:* วันหยุดหรือไม่
  * *clicks\_cta:* จำนวนครั้งที่กดปุ่ม call-to-action
  * *returning\_user:* เคยมาใช้งานมาก่อนหรือไม่

* **Label (y):** `signed_up ∈ {0,1}`

* **Sample size:** 2,000 records

### 3. การวิเคราะห์เบื้องต้น (EDA)

1. **Rate ของการสมัครใช้งาน (conversion rate):**

   * แบ่งตาม `device` → mobile อาจสมัครน้อยกว่า desktop
   * แบ่งตาม `referral` → traffic จาก email/ads มัก conversion สูงกว่า direct
2. **Distribution:**

   * *session\_duration\_s:* อาจ skew (ส่วนใหญ่ใช้เวลาสั้น, บางรายอยู่นานมาก)
   * *pages\_viewed:* ตรวจว่าหน้าเยอะสัมพันธ์กับ signup หรือไม่
3. **Correlation:**

   * `clicks_cta` และ `session_duration_s` มักมี positive correlation กับการ signup


### 4. Baseline Model

* **Predict majority class:** เช่น ถ้า 70% ของ session ไม่สมัคร → baseline accuracy = 70%
* ใช้เปรียบเทียบกับโมเดลจริงว่าดีกว่าการทายแบบ “เดาสุ่ม” หรือไม่



### 5. โมเดล (Modeling Strategy)

1. **Logistic Regression (เริ่มต้น):**

   * One-hot encode categorical features (device, referral, country)
   * ตีความ coefficient เป็น log-odds → เข้าใจได้ว่า feature ไหนเพิ่มโอกาส signup

2. **Tree-based (Decision Tree / Random Forest):**

   * รองรับ non-linearity และ interaction ระหว่างฟีเจอร์
   * ตีความง่ายด้วย feature importance

### 6. การประเมิน (Evaluation)

* **Confusion Matrix:** แสดง TP, TN, FP, FN
* **Accuracy:** วัดการทำนายถูกโดยรวม (ดีเมื่อ class balance)
* **Precision:** สัดส่วน signup ที่ทำนายว่า signup แล้วถูกต้องจริง
* **Recall:** สัดส่วน signup จริงที่ถูกจับได้
* **F1-score:** สมดุลระหว่าง Precision และ Recall
* **ROC-AUC:** ความสามารถแยก positive/negative โดยไม่ขึ้นกับ threshold

เพิ่มเติม:

* **Threshold analysis:** เลือก threshold ที่เหมาะสมขึ้นกับ business goal เช่น

  * ถ้าอยาก “ลด false positive” (ไม่อยากตามลูกค้าที่ไม่สมัครจริง) → เน้น **Precision**
  * ถ้าอยาก “จับลูกค้าสมัครให้ครบ” → เน้น **Recall**



### 7. การแปลผลเชิงความหมาย (Interpretation)

* ถ้า **Accuracy = 85%** แต่ dataset imbalance (90% ไม่สมัคร, 10% สมัคร) → อาจไม่ดีนัก ต้องดู Precision/Recall ด้วย
* ถ้า **Precision = 0.80, Recall = 0.60** → 80% ของที่โมเดลบอกว่าสมัครจริงสมัครจริง แต่จับลูกค้าสมัครมาได้แค่ 60%
* ถ้า **F1 = 0.69** → สมดุลกลาง ๆ ระหว่าง Precision และ Recall
* ถ้า **AUC = 0.90** → โมเดลแยก “signup vs not signup” ได้ดีมาก


### 🔹 สรุป Workflow

1. **EDA:** Conversion rate ตาม device/referral/country, distribution ของ session duration, correlation ของเชิงปริมาณ
2. **Baseline:** ทาย majority class
3. **Model:** Logistic Regression (เริ่มต้น) → Tree/Random Forest พร้อม one-hot encoding
4. **Evaluation:** Confusion Matrix, Accuracy, Precision, Recall, F1, ROC-AUC
5. **Threshold tuning:** วิเคราะห์ Precision-Recall trade-off
6. **Interpretation:**

   * Logistic: coefficient/log-odds แปลตรงไปตรงมา (เช่น `clicks_cta` +1 → โอกาสสมัครเพิ่ม)
   * Tree/Forest: ใช้ feature importance ดูปัจจัยที่สำคัญที่สุดต่อ signup

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

# -----------------------------
# 1. สร้างข้อมูลสมมุติ
# -----------------------------
np.random.seed(42)
n = 2000

session_duration = np.random.exponential(scale=120, size=n)   # วินาที
pages_viewed = np.random.poisson(lam=4, size=n)
device = np.random.choice(["mobile", "desktop", "tablet"], size=n, p=[0.6, 0.3, 0.1])
referral = np.random.choice(["ads", "seo", "email", "direct"], size=n, p=[0.3, 0.3, 0.2, 0.2])
country = np.random.choice(["TH", "VN", "ID", "SG"], size=n, p=[0.4, 0.2, 0.3, 0.1])
is_weekend = np.random.binomial(1, 0.3, size=n)
clicks_cta = np.random.poisson(lam=1, size=n)
returning_user = np.random.binomial(1, 0.4, size=n)

# target: signed_up
# โมเดลกำหนดโอกาสสมัครจากบาง feature
logits = (
    0.001*session_duration
    + 0.2*pages_viewed
    + 0.5*clicks_cta
    + 0.8*returning_user
    + np.where(device=="desktop", 0.5, 0)
    + np.where(referral=="email", 0.7, 0)
    + np.where(is_weekend==1, -0.2, 0)
)
prob = 1 / (1 + np.exp(-logits))
signed_up = np.random.binomial(1, prob)

df = pd.DataFrame({
    "session_duration_s": session_duration,
    "pages_viewed": pages_viewed,
    "device": device,
    "referral": referral,
    "country": country,
    "is_weekend": is_weekend,
    "clicks_cta": clicks_cta,
    "returning_user": returning_user,
    "signed_up": signed_up
})

print(df.head())

# -----------------------------
# 2. Train/Test Split
# -----------------------------
X = df.drop("signed_up", axis=1)
y = df["signed_up"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# 3. Preprocessing + Models
# -----------------------------
categorical = ["device", "referral", "country"]
numeric = ["session_duration_s", "pages_viewed", "clicks_cta", "is_weekend", "returning_user"]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(drop="first"), categorical),
    ("num", "passthrough", numeric)
])

# Logistic Regression pipeline
log_reg = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Random Forest pipeline
rf = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

# -----------------------------
# 4. Baseline
# -----------------------------
baseline_pred = np.repeat(y_train.mode()[0], len(y_test))

# -----------------------------
# 5. Training
# -----------------------------
log_reg.fit(X_train, y_train)
rf.fit(X_train, y_train)

y_pred_log = log_reg.predict(X_test)
y_prob_log = log_reg.predict_proba(X_test)[:,1]

y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:,1]

# -----------------------------
# 6. Evaluation Function
# -----------------------------
def evaluate(y_true, y_pred, y_prob, model_name):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    print(f"{model_name}:")
    print(f" Accuracy={acc:.3f}, Precision={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f}, ROC-AUC={auc:.3f}")
    print("-"*50)

evaluate(y_test, baseline_pred, np.zeros(len(y_test)), "Baseline")
evaluate(y_test, y_pred_log, y_prob_log, "Logistic Regression")
evaluate(y_test, y_pred_rf, y_prob_rf, "Random Forest")

# -----------------------------
# 7. Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Signup","Signup"], yticklabels=["Not Signup","Signup"])
plt.title("Confusion Matrix (Random Forest)")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()

# -----------------------------
# 8. ROC Curve
# -----------------------------
fpr, tpr, _ = roc_curve(y_test, y_prob_rf)
plt.plot(fpr, tpr, label="Random Forest")
fpr2, tpr2, _ = roc_curve(y_test, y_prob_log)
plt.plot(fpr2, tpr2, label="Logistic Regression")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
```