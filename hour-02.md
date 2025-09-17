
# 📘 Hour 2 – Data Cleaning & Preprocessing


## 1. ความรู้ขั้นต้น

### A. Missing Data (ข้อมูลที่หายไป)

#### 🔹 วิธีจัดการ Missing Values

1. **ลบแถว/คอลัมน์ (Drop)**

* ใช้เมื่อ missing มีจำนวนน้อยมาก (<5%)

```python
df = df.dropna()              # ลบทุกแถวที่มี NaN
df = df.dropna(axis=1)        # ลบคอลัมน์ที่มี NaN
```

2. **เติมด้วยสถิติ (Mean / Median / Mode Imputation)**

* **Mean Imputation:**
  $x_i = \frac{1}{n} \sum_{j=1}^n x_j$

* **Median Imputation:** ใช้ค่ากลางแทน missing

* **Mode Imputation:** ใช้ค่าที่พบบ่อยที่สุด

```python
df["Age"].fillna(df["Age"].mean(), inplace=True)     # mean
df["Age"].fillna(df["Age"].median(), inplace=True)   # median
df["City"].fillna(df["City"].mode()[0], inplace=True) # mode
```

3. **Interpolation (ประมาณค่า)**

* ใช้ค่าที่อยู่รอบ ๆ เพื่อคำนวณแทนค่า missing
* **สมการ Linear Interpolation:**
  $x_{t} = x_{t-1} + \frac{(x_{t+1} - x_{t-1})}{(t+1 - (t-1))} \times (t - (t-1))$

```python
df["Temperature"] = df["Temperature"].interpolate(method="linear")
```

4. **Regression Imputation**

* ใช้ **สมการถดถอยเชิงเส้น (Linear Regression)** เพื่อทำนาย missing
* **สมการ:**
  $\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n$

```python
from sklearn.linear_model import LinearRegression

X = df[["Age","Experience"]]    # ตัวแปรอิสระ
y = df["Salary"]                # ตัวแปรเป้าหมาย
model = LinearRegression().fit(X.dropna(), y.dropna())
df.loc[df["Salary"].isnull(), "Salary"] = model.predict(X[df["Salary"].isnull()])
```


 5. **KNN Imputation**

* ใช้ค่าเฉลี่ยของ **K เพื่อนบ้านที่ใกล้ที่สุด**
* **สมการ:**
  $x_i = \frac{1}{k} \sum_{j \in N_k(i)} x_j$

```python
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=3)
df_filled = imputer.fit_transform(df)
```
 

### B. Feature Transformation (การแปลงคุณลักษณะ)


#### 🔹 Scaling

1. **Min-Max Scaling (Normalization)**

* **แนวคิด**: ย่อข้อมูลให้อยู่ในช่วง $[0,1]$
* **สมการ:**

  $$
  x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}
  $$

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df["Salary_norm"] = scaler.fit_transform(df[["Salary"]])
```

2. **Standardization (Z-score)**

* **แนวคิด**: ทำให้ข้อมูลมีค่าเฉลี่ย (mean) = 0 และส่วนเบี่ยงเบนมาตรฐาน (std) = 1
* **สมการ:**

  $$
  z = \frac{x - \mu}{\sigma}
  $$

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df["Salary_std"] = scaler.fit_transform(df[["Salary"]])
```
**ตัวอย่าง**


| Name    | Salary | Salary\_MinMax | Salary\_Zscore |
| ------- | ------ | -------------- | -------------- |
| Alice   | 50,000 | 0.33           | -0.26          |
| Bob     | 60,000 | 1.00           | 1.30           |
| Charlie | 45,000 | 0.00           | -1.04          |

**คำอธิบาย**

* **Salary (ดิบ):** ข้อมูลเงินเดือนจริง → scale มีความต่างกันมาก
* **Salary\_MinMax (0–1):** ย่อข้อมูลให้อยู่ในช่วง 0 ถึง 1 → ดีสำหรับ neural network หรือ distance-based methods (KNN, clustering)
* **Salary\_Zscore:** ทำให้ข้อมูลมี mean = 0, std = 1 → ดีสำหรับ linear models, PCA


#### 🔹 Encoding

1. **Label Encoding**

* **แนวคิด**: แปลงค่าข้อความ → ตัวเลขเรียงลำดับ
* เช่น `["Bangkok","Phuket","Chiang Mai"]` → `[0,1,2]`

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["City_encoded"] = le.fit_transform(df["City"])
```

2. **One-Hot Encoding**

* **แนวคิด**: แปลง category → binary columns
* เช่น `City = ["Bangkok","Phuket"]` → `Bangkok=1, Phuket=0`

```python
df = pd.get_dummies(df, columns=["City"])
```
**ตัวอย่าง**

| Name    | City       | City\_Label | City\_Bangkok | City\_ChiangMai | City\_Phuket |
| ------- | ---------- | ----------- | ------------- | --------------- | ------------ |
| Alice   | Bangkok    | 0           | 1             | 0               | 0            |
| Bob     | Phuket     | 2           | 0             | 0               | 1            |
| Charlie | Chiang Mai | 1           | 0             | 1               | 0            |


**คำอธิบาย**

* **City\_Label** → ใช้ **Label Encoding** (Bangkok=0, Chiang Mai=1, Phuket=2)
* **City\_Bangkok / City\_ChiangMai / City\_Phuket** → ใช้ **One-Hot Encoding** (binary columns)

#### 🔹 Date/Time Features

1. **Extract Components**

* แยกปี เดือน วัน วันในสัปดาห์ ชั่วโมง

```python
df["JoinDate"] = pd.to_datetime(df["JoinDate"])
df["Year"] = df["JoinDate"].dt.year
df["Month"] = df["JoinDate"].dt.month
df["Day"] = df["JoinDate"].dt.day
df["Weekday"] = df["JoinDate"].dt.weekday   # 0=Monday
df["Hour"] = df["JoinDate"].dt.hour
```

2. **สร้าง Features ใหม่**

* **is\_weekend:**

  $$
  is\_weekend = 
  \begin{cases} 
  1 & \text{if weekday ∈ \{5,6\}} \\ 
  0 & \text{otherwise} 
  \end{cases}
  $$

```python
df["IsWeekend"] = df["JoinDate"].dt.weekday >= 5
```

* **Season (ฤดู)** – ตัวอย่าง (ไทย: Summer=3–5, Rainy=6–10, Winter=11–2)

```python
def season(month):
    if month in [3,4,5]:
        return "Summer"
    elif month in [6,7,8,9,10]:
        return "Rainy"
    else:
        return "Winter"

df["Season"] = df["JoinDate"].dt.month.apply(season)
```
 
#### สรุป

* **Scaling** → ทำให้ numerical features มี scale ที่เหมาะสม (ช่วยให้โมเดล converge เร็วขึ้น)
* **Encoding** → แปลง categorical features ให้โมเดลเข้าใจได้
* **Date/Time Features** → สกัดข้อมูลเชิงเวลาเพื่อเพิ่มพลังการพยากรณ์

**ตัวอย่าง Date/Time Feature Extraction**

| Name    | JoinDate   | Year | Month | Day | Weekday | IsWeekend | Season |
| ------- | ---------- | ---- | ----- | --- | ------- | --------- | ------ |
| Alice   | 2020-01-15 | 2020 | 1     | 15  | 2 (Wed) | 0         | Winter |
| Bob     | 2020-07-20 | 2020 | 7     | 20  | 0 (Mon) | 0         | Rainy  |
| Charlie | 2020-12-05 | 2020 | 12    | 5   | 5 (Sat) | 1         | Winter |

**คำอธิบาย**

* **Year / Month / Day** → ดึงจาก `JoinDate` โดยตรง
* **Weekday** → ค่า 0–6 (0=Monday, …, 6=Sunday)
* **IsWeekend** → 1 ถ้าเป็นวันเสาร์–อาทิตย์, 0 ถ้าเป็นวันทำงาน
* **Season** → กำหนดเอง (ไทย: Summer=3–5, Rainy=6–10, Winter=11–2)
 
 

### C. Unstructured Data

### 🔹 ตัวอย่างข้อมูลข้อความ (Raw Text)

```
"Data Science is FUN!!! Data science helps in decision making."
```

---

### 1. การทำความสะอาดข้อความ (Text Cleaning)

* **Lowercasing** → เปลี่ยนเป็นตัวพิมพ์เล็กทั้งหมด
* **Stopword Removal** → ลบคำที่ไม่ช่วยเชิงความหมาย เช่น "is", "in", "the"
* **Punctuation Removal** → ลบสัญลักษณ์ เช่น `!`, `.`

```python
import re
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

text = "Data Science is FUN!!! Data science helps in decision making."

# Lowercasing
text = text.lower()

# Remove punctuation
text = re.sub(r"[^a-zA-Z\s]", "", text)

# Remove stopwords
tokens = [word for word in text.split() if word not in stop_words]
print(tokens)
```

✅ Output:

```
['data', 'science', 'fun', 'data', 'science', 'helps', 'decision', 'making']
```

---

### 2. Tokenization

* การแยกข้อความเป็นคำ (tokens)

```python
from nltk.tokenize import word_tokenize
nltk.download("punkt")

tokens = word_tokenize(text)
print(tokens)
```

✅ Output:

```
['data', 'science', 'fun', 'data', 'science', 'helps', 'decision', 'making']
```

---

### 3. Bag-of-Words (BoW)

* นับจำนวนคำในเอกสาร (word frequency)

```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    "Data science is fun",
    "Science helps in decision making"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

print(vectorizer.get_feature_names_out())
print(X.toarray())
```

✅ Output:

| Word | data | decision | fun | helps | in | is | making | science |
| ---- | ---- | -------- | --- | ----- | -- | -- | ------ | ------- |
| Doc1 | 1    | 0        | 1   | 0     | 0  | 1  | 0      | 1       |
| Doc2 | 0    | 1        | 0   | 1     | 1  | 0  | 1      | 1       |

---

### 4. TF-IDF (Term Frequency – Inverse Document Frequency)

* ใช้น้ำหนักแทนการนับคำ เพื่อเน้นคำที่สำคัญในเอกสาร
* **สมการ:**

$$
TFIDF(t,d) = TF(t,d) \times \log \frac{N}{DF(t)}
$$

* $TF(t,d)$ = ความถี่ของคำ t ในเอกสาร d
* $DF(t)$ = จำนวนเอกสารที่มีคำ t
* $N$ = จำนวนเอกสารทั้งหมด

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

print(vectorizer.get_feature_names_out())
print(X.toarray())
```

✅ Output (ค่าระหว่าง 0–1 แทนความสำคัญของแต่ละคำ)

| Word | data | decision | fun  | helps | in   | is   | making | science |
| ---- | ---- | -------- | ---- | ----- | ---- | ---- | ------ | ------- |
| Doc1 | 0.58 | 0.00     | 0.58 | 0.00  | 0.00 | 0.58 | 0.00   | 0.45    |
| Doc2 | 0.00 | 0.50     | 0.00 | 0.50  | 0.50 | 0.00 | 0.50   | 0.38    |

---

## ✅ สรุปสิ่งที่สอน (สำหรับชั่วโมงนี้)

1. ความแตกต่างระหว่าง **Structured vs Unstructured Data**
2. การ Clean ข้อความ (lowercasing, stopwords, punctuation removal)
3. การแปลงข้อความเป็น features ด้วย **Tokenization → Bag-of-Words → TF-IDF**

 

## 2. Hands-on Exercises 

### Structure Data

```python
# ========================================
# STEP 0: Import Libraries
# ========================================
import pandas as pd
import numpy as np

# ========================================
# STEP 1: Load Dataset
# ========================================
file_path = "messy_data.csv"   # แก้เป็น path ของไฟล์ในเครื่อง/Colab
df = pd.read_csv(file_path)

print("🔹 ข้อมูล 5 แถวแรก:")
print(df.head())
print("\n🔹 ข้อมูลเบื้องต้น:")
print(df.info())
```

---

```python
# ========================================
# STEP 2: ตรวจสอบ Missing Values
# ========================================
print("จำนวน Missing values ต่อคอลัมน์:")
print(df.isnull().sum())

# แสดง % missing
print("\nเปอร์เซ็นต์ Missing:")
print((df.isnull().mean() * 100).round(2))
```

---

```python
# ========================================
# STEP 3: Handling Missing Data
# ========================================

# Age → เติมด้วยค่าเฉลี่ย
df["Age"].fillna(df["Age"].mean(), inplace=True)

# City → เติมด้วย mode (ค่าที่พบบ่อยที่สุด)
df["City"].fillna(df["City"].mode()[0], inplace=True)

# Salary → เติมด้วย median
df["Salary"].fillna(df["Salary"].median(), inplace=True)

# Comments → เติมด้วย "Unknown"
df["Comments"].fillna("Unknown", inplace=True)
```

---

```python
# ========================================
# STEP 4: Clean Text (Comments)
# ========================================
import re

def clean_text(text):
    text = str(text).lower()                       # lowercase
    text = re.sub(r"[^a-z\s]", "", text)           # ลบ punctuation/ตัวเลข
    return text.strip()

df["Comments_clean"] = df["Comments"].apply(clean_text)

print(df[["Comments", "Comments_clean"]].head(10))
```

---

```python
# ========================================
# STEP 5: Remove Duplicates
# ========================================
before = df.shape[0]
df.drop_duplicates(inplace=True)
after = df.shape[0]

print(f"ลบ duplicates แล้ว: {before - after} records ถูกลบ")
```

---

```python
# ========================================
# STEP 6: Feature Transformation
# ========================================

from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

# ---- Scaling ----
scaler = MinMaxScaler()
df["Salary_MinMax"] = scaler.fit_transform(df[["Salary"]])

scaler = StandardScaler()
df["Salary_Zscore"] = scaler.fit_transform(df[["Salary"]])

# ---- Encoding ----
le = LabelEncoder()
df["Department_Label"] = le.fit_transform(df["Department"])

df = pd.get_dummies(df, columns=["City"], prefix="City")

# ---- Date/Time ----
df["JoinDate"] = pd.to_datetime(df["JoinDate"])
df["Year"] = df["JoinDate"].dt.year
df["Month"] = df["JoinDate"].dt.month
df["Day"] = df["JoinDate"].dt.day
df["Weekday"] = df["JoinDate"].dt.weekday
df["IsWeekend"] = (df["JoinDate"].dt.weekday >= 5).astype(int)
```

---

```python
# ========================================
# STEP 7: Feature Engineering - Season
# ========================================
def season(month):
    if month in [3,4,5]:
        return "Summer"
    elif month in [6,7,8,9,10]:
        return "Rainy"
    else:
        return "Winter"

df["Season"] = df["Month"].apply(season)
```

---

```python
# ========================================
# STEP 8: Export Ready-to-Use Dataset
# ========================================
output_file = "cleaned_ready_data.csv"
df.to_csv(output_file, index=False)

print(f"✅ Preprocessed dataset saved as {output_file}")
```

---

### ✅ สิ่งที่นักศึกษาจะได้เรียนรู้

1. การตรวจสอบและจัดการ **Missing Values** (drop, fillna ด้วย mean/median/mode)
2. การจัดการข้อความเบื้องต้น (text cleaning)
3. การลบ **Duplicates**
4. **Scaling & Encoding** เพื่อทำให้ numerical/categorical features ใช้กับโมเดลได้
5. **Date/Time Feature Engineering** (Year, Month, Day, Weekday, IsWeekend, Season)
6. การ export dataset ที่พร้อมใช้งาน (**Ready-to-Use File**)


## Unstructure data (Text)

```python
# ========================================
# STEP 0: Import Libraries
# ========================================
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import re
```

---

```python
# ========================================
# STEP 1: Load Dataset
# ========================================
file_path = "sms_spam_ham.csv"   # เปลี่ยนเป็น path ที่เก็บไฟล์
df = pd.read_csv(file_path)

print("🔹 ข้อมูล 5 แถวแรก:")
print(df.head())
print("\nจำนวน spam:", (df["Label"]=="spam").sum())
print("จำนวน ham:", (df["Label"]=="ham").sum())
```

---

```python
# ========================================
# STEP 2: Text Preprocessing
# ========================================
def clean_text(text):
    text = text.lower()                         # lowercase
    text = re.sub(r"[^a-z\s]", "", text)        # ลบตัวเลข/สัญลักษณ์
    return text.strip()

df["Clean_Message"] = df["Message"].apply(clean_text)

print(df[["Message", "Clean_Message"]].head(10))
```

---

```python
# ========================================
# STEP 3: Split Train/Test
# ========================================
X = df["Clean_Message"]
y = df["Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("ขนาด Training set:", len(X_train))
print("ขนาด Test set:", len(X_test))
```

---

```python
# ========================================
# STEP 4: TF-IDF Vectorization
# ========================================
vectorizer = TfidfVectorizer(stop_words="english")

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("รูปแบบ TF-IDF:", X_train_tfidf.shape)
```

---

```python
# ========================================
# STEP 5: Train Naive Bayes Classifier
# ========================================
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)
```

---

```python
# ========================================
# STEP 6: Evaluation
# ========================================
print("🔹 Classification Report:")
print(classification_report(y_test, y_pred))

print("🔹 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
```

---

### ✅ สิ่งที่นักศึกษาจะได้เรียนรู้

1. การทำ **Text Cleaning** (lowercasing, remove punctuation)
2. การแปลงข้อความเป็น **TF-IDF features**
3. การแบ่ง dataset เป็น **Train/Test**
4. การสร้างและ train **Naive Bayes Classifier** สำหรับงาน text classification
5. การประเมินผลด้วย **Confusion Matrix และ Classification Report**

