
# Hour 3  - Exploratory Data Analysis (EDA) & Visualization

## Background

### 1. หลักการของ Exploratory Data Analysis (EDA)

EDA เป็นกระบวนการสำรวจข้อมูลเพื่อค้นหารูปแบบ ความสัมพันธ์ และความผิดปกติ ก่อนนำไปสู่การสร้างโมเดล โดยมีเป้าหมายหลักคือ

 

**📊 ข้อมูลสมมุติ: คะแนนสอบนักศึกษา**

| Student | Math | English | StudyHours | Attendance (%) |
| ------- | ---- | ------- | ---------- | -------------- |
| S1      | 80   | 75      | 10         | 90             |
| S2      | 60   | 65      | 6          | 70             |
| S3      | 95   | 88      | 15         | 95             |
| S4      | 40   | 50      | 3          | 60             |
| S5      | 85   | 82      | 12         | 92             |
| S6      | 30   | 35      | 2          | 55             |
| S7      | 78   | 70      | 9          | 85             |
| S8      | 55   | 58      | 5          | 65             |
| S9      | 98   | 90      | 16         | 96             |
| S10     | 45   | 55      | 4          | 62             |

---

 1. **สรุปข้อมูลเบื้องต้น (Descriptive Statistics)**

```python
import pandas as pd

data = {
    "Math": [80, 60, 95, 40, 85, 30, 78, 55, 98, 45],
    "English": [75, 65, 88, 50, 82, 35, 70, 58, 90, 55],
    "StudyHours": [10, 6, 15, 3, 12, 2, 9, 5, 16, 4],
    "Attendance": [90, 70, 95, 60, 92, 55, 85, 65, 96, 62]
}
df = pd.DataFrame(data)
print(df.describe())
```

📌 **การแปลผล**

* ค่าเฉลี่ย (mean) ของคณิตศาสตร์ = \~66.6 แปลว่าคะแนนนักศึกษาส่วนใหญ่ปานกลาง
* ค่ามัธยฐาน (median) ของการเข้าเรียน = 77.5 แปลว่าครึ่งหนึ่งมาเรียนเกิน 77.5%
* ค่า SD (ส่วนเบี่ยงเบนมาตรฐาน) ของ StudyHours สูง → บางคนอ่านหนังสือมากผิดปกติเมื่อเทียบกับเพื่อน
 

 2. **ตรวจสอบโครงสร้างของข้อมูล**

  Distribution

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(df["Math"], bins=5, kde=True)
plt.show()
```

📌 **การแปลผล**

* Histogram แสดงว่าคะแนนคณิตศาสตร์มีการกระจายกว้าง บางคนได้คะแนนต่ำมาก (30–40) แต่บางคนสูงมาก (95–98) → ข้อมูลกระจายสองฝั่ง (high vs low performers)

  Correlation

```python
print(df.corr())
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.show()
```

📌 **การแปลผล**

* พบว่า **StudyHours มีค่าสหสัมพันธ์สูงกับ Math (r ≈ 0.95)** → ยิ่งอ่านหนังสือมาก คะแนนคณิตยิ่งสูง
* English ก็สัมพันธ์กับ Math (r ≈ 0.9) → นักเรียนที่เก่งมักจะทำได้ดีทั้งสองวิชา

  3. **ค้นหา Outlier**

**Boxplot**

```python
sns.boxplot(x=df["Math"])
plt.show()
```

**การแปลผล**

* เห็นว่า **คะแนน Math = 30** (นักเรียน S6) เป็น Outlier เพราะต่ำกว่ากลุ่มมาก
* อาจสะท้อนว่ามีปัญหาเฉพาะบุคคล (ไม่เข้าเรียน, ไม่อ่านหนังสือ) ที่ต่างจากเพื่อน ๆ

 4. **สร้างสมมติฐานเบื้องต้น**

จากการสังเกตเบื้องต้น สามารถตั้งสมมติฐานได้ว่า:

1. นักศึกษาที่มี **ชั่วโมงการอ่านหนังสือมากกว่า 10 ชั่วโมง/สัปดาห์** จะได้คะแนนคณิตศาสตร์สูงกว่า 80 คะแนน
2. อัตราการเข้าเรียนที่สูงกว่า 85% จะสัมพันธ์กับคะแนนภาษาอังกฤษสูงกว่า 75 คะแนน
3. Outlier เช่น S6 ควรศึกษาเพิ่มเติมว่าเกิดจากปัจจัยอื่น เช่น ขาดแรงจูงใจหรือมีปัญหาด้านการเรียน
 

### 2. หลักการเลือกกราฟที่เหมาะสม

 
**📊 ข้อมูลสมมุติ: ข้อมูลนักศึกษา**

| Student | Math | English | StudyHours | Department |
| ------- | ---- | ------- | ---------- | ---------- |
| S1      | 80   | 75      | 10         | IT         |
| S2      | 60   | 65      | 6          | HR         |
| S3      | 95   | 88      | 15         | Finance    |
| S4      | 40   | 50      | 3          | IT         |
| S5      | 85   | 82      | 12         | Finance    |
| S6      | 30   | 35      | 2          | HR         |
| S7      | 78   | 70      | 9          | IT         |
| S8      | 55   | 58      | 5          | Finance    |
| S9      | 98   | 90      | 16         | IT         |
| S10     | 45   | 55      | 4          | HR         |


```python
# !pip install seaborn plotly --quiet   # (หากยังไม่มี ให้เปิดคอมเมนต์บรรทัดนี้)
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# สร้างข้อมูลนักศึกษา (สมมุติ)
df = pd.DataFrame({
    "Student": [f"S{i}" for i in range(1, 11)],
    "Math": [80,60,95,40,85,30,78,55,98,45],
    "English": [75,65,88,50,82,35,70,58,90,55],
    "StudyHours": [10,6,15,3,12,2,9,5,16,4],
    "Department": ["IT","HR","Finance","IT","Finance","HR","IT","Finance","IT","HR"]
})

df.head()
```

1. ข้อมูลเชิงปริมาณ (Numeric Data)

**Histogram (การกระจายของคะแนน Math)**

* ใช้ดูว่าคะแนนนักศึกษากระจายอยู่ช่วงใดบ้าง
```python
display(df.describe())  # mean, std, quartiles สำหรับตัวเลข
print("ค่าเฉลี่ยรายแผนก (Math):")
print(df.groupby("Department")["Math"].mean())

# แนวทางแปลผล:
# - mean/std บอกระดับและการกระจายโดยรวม
# - groupby ช่วยดูความต่างเชิงหมวดหมู่ (เช่น IT vs HR vs Finance)

# Histogram: การกระจายของคะแนน Math
sns.histplot(df["Math"], bins=5, kde=True)
plt.title("Histogram of Math Scores")
plt.xlabel("Math")
plt.ylabel("Count")
plt.show()
# ตีความ: ดูรูปทรงการกระจาย (มีหาง/หลายยอด/คับแน่นช่วงใด)

```
* **แปลผล:** เห็นว่าคะแนนมีทั้งกลุ่มต่ำ (30–40) และกลุ่มสูง (80–100) → distribution อาจเป็น **bimodal**



**Boxplot (การกระจายของ English)**

* ใช้หาค่า outlier และสรุปค่ากลางของข้อมูล
```python

# Boxplot: Outlier + การกระจายของ English
sns.boxplot(x=df["English"])
plt.title("Boxplot of English Scores")
plt.xlabel("English")
plt.show()
# ตีความ: ตำแหน่ง median, IQR, จุด outliers (ค่าผิดปกติ)

```

* **แปลผล:** ค่า median ประมาณ 70 แต่มี outlier ต่ำ (35 คะแนน) → บ่งชี้ว่านักศึกษา HR บางคนอาจมีปัญหาด้านภาษา

**Scatter plot (Math vs StudyHours)**

* ใช้ตรวจสอบความสัมพันธ์ระหว่างชั่วโมงอ่านหนังสือกับคะแนน
```python

# Scatter: ความสัมพันธ์ Math vs StudyHours
sns.scatterplot(x="StudyHours", y="Math", data=df)
plt.title("StudyHours vs Math")
plt.xlabel("StudyHours")
plt.ylabel("Math")
plt.show()
# ตีความ: ความชันโดยรวมเป็นบวก? จุดกระจายแน่น/หลวม? มีจุดผิดปกติหรือไม่?
```
* **แปลผล:** เห็นว่า **ยิ่งอ่านมาก คะแนนยิ่งสูง** → ความสัมพันธ์เชิงบวก (positive correlation)

2. ข้อมูลเชิงหมวดหมู่ (Categorical Data)

**Bar chart (คะแนนเฉลี่ยของแต่ละแผนก)**

* แสดงค่าเฉลี่ยของ Department: IT, HR, Finance
```python
# Bar chart: คะแนนเฉลี่ย Math ต่อ Department
dept_mean = df.groupby("Department")["Math"].mean().reset_index()
sns.barplot(x="Department", y="Math", data=dept_mean)
plt.title("Average Math by Department")
plt.show()
# ตีความ: แผนกใดมีค่าเฉลี่ยสูง/ต่ำกว่ากัน? ความต่างอาจสะท้อนปัจจัยด้านการเรียนรู้

```
* **แปลผล:** IT และ Finance มีคะแนนเฉลี่ยสูงกว่า HR → แสดงความแตกต่างเชิงกลุ่ม

**Pie chart (สัดส่วนนักศึกษาแต่ละแผนก)**

* ใช้เมื่อหมวดหมู่ไม่เกิน 4–5 กลุ่ม
```python
# Pie chart: สัดส่วนนักศึกษาแต่ละแผนก
counts = df["Department"].value_counts()
plt.pie(counts, labels=counts.index, autopct="%1.1f%%")
plt.title("Department Proportion")
plt.show()
# ตีความ: ใช้เมื่อหมวดหมู่ไม่เกิน ~4–5 กลุ่ม เพื่อสื่อสัดส่วนอย่างง่าย
```
* **แปลผล:** HR 30%, IT 40%, Finance 30% → ทำให้เห็นสัดส่วนกลุ่มที่ชัดเจน

3. ข้อมูลหลายมิติ

**Heatmap (Correlation Matrix)**

* แสดงความสัมพันธ์ของ Math, English, StudyHours
```python
# Heatmap: สหสัมพันธ์ระหว่างตัวเลข
corr = df[["Math","English","StudyHours"]].corr()
sns.heatmap(corr, annot=True, fmt=".2f")
plt.title("Correlation Matrix")
plt.show()
# ตีความ: ค่าบวกสูงชี้ความสัมพันธ์ทางบวก (เช่น อ่านมาก → คะแนนสูง)
```
* **แปลผล:** Math ↔ StudyHours มีค่า correlation สูงสุด (r ≈ 0.9)

**Pair plot**

* ใช้ดูการกระจายและความสัมพันธ์ของตัวแปรหลายมิติพร้อมกัน
```python
# Pair plot: ความสัมพันธ์หลายตัวแปรพร้อมกัน
sns.pairplot(df[["Math","English","StudyHours"]], diag_kind="hist")
plt.show()
# ตีความ: เห็นการกระจายของแต่ละตัวแปรบนเส้นทแยงมุม และ scatter คู่ตัวแปรอื่น ๆ
```
* **แปลผล:** เห็น distribution ของแต่ละวิชา และ scatter plot คู่ ๆ กัน → ใช้เพื่อสร้าง insight เพิ่มเติม

**✅ หลักการเลือกกราฟ**

* Histogram/Boxplot → ดีสำหรับ numeric distribution
* Scatter plot → ดีสำหรับความสัมพันธ์ระหว่างตัวแปรสองตัว
* Bar chart/Pie chart → ดีสำหรับ categorical comparison
* Heatmap/Pair plot → ดีสำหรับความสัมพันธ์หลายตัวแปร
* ควรใช้สีที่เหมาะสม, หลีกเลี่ยง scale บิดเบือน และเลือก visualization ที่สื่อสารชัดเจนที่สุด

### 3. Storytelling with Data

Visualization ไม่ใช่แค่การแสดงกราฟ แต่คือการ **เล่าเรื่องจากข้อมูล (Data Storytelling)** ซึ่งมีหลักการคือ:

1. **กำหนดประเด็นหลัก (Key Message):** ต้องการให้ผู้ชมเข้าใจอะไร? เช่น “ยอดขายเพิ่มขึ้นใน Q3”
2. **เลือก visualization ที่สนับสนุนข้อความนั้น:** เช่น line chart เพื่อเล่าแนวโน้มตามเวลา
3. **ลดสิ่งรบกวน (Decluttering):** ตัด labels, gridlines หรือสีที่ไม่จำเป็นออก
4. **เน้นข้อมูลสำคัญ (Highlighting):** ใช้สีหรือ annotation เพื่อดึงความสนใจไปที่ insight
5. **เล่าเป็นเรื่องราว (Narrative):** จาก introduction → insight → conclusion

 

## ชุดข้อมูลสมมุติ (ยอดขายรายสัปดาห์ของปีหนึ่ง ๆ)

สมมุติว่าเราอยากสื่อสารประเด็นหลัก: **“ยอดขายเพิ่มขึ้นชัดเจนในไตรมาส 3 (Q3)”**

```python
import pandas as pd
import numpy as np

# สร้างข้อมูลรายสัปดาห์ 52 สัปดาห์
rng = pd.date_range("2024-01-01", periods=52, freq="W")
np.random.seed(7)
base = np.linspace(100, 140, 52)                 # แนวโน้มโดยรวมเพิ่มขึ้นเล็กน้อย
season = 10*np.sin(np.linspace(0, 6*np.pi, 52))  # ฤดูกาล
noise = np.random.normal(0, 5, 52)
sales = base + season + noise

# กระตุ้น Q3 (สัปดาห์ ~27–39) ให้โดดเด่น
q3_idx = (rng.quarter == 3)
sales[q3_idx] += 15

df = pd.DataFrame({"week": rng, "sales": sales})
df["quarter"] = df["week"].dt.quarter
```


## 1) กำหนดประเด็นหลัก (Key Message)

> “ยอดขายเพิ่มขึ้นเด่นชัดใน Q3 เมื่อเทียบกับ Q1–Q2 และยังคงทรงตัวสูงใน Q4”

**วิธีคิด:** ประเด็นต้องสั้น ชัดเจน ตอบคำถาม “So what?”—สิ่งที่ผู้ชมควรรับไปตัดสินใจได้


 2) เลือก visualization ให้สอดคล้องกับประเด็น

เพราะเป็น **แนวโน้มตามเวลา** → เลือก **line chart** เป็นกราฟหลัก และเสริมด้วย **bar chart แบบสรุปเฉลี่ยรายไตรมาส** เพื่อยืนยันภาพรวม

**Line Chart: แนวโน้มรายสัปดาห์**

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9,4))
ax.plot(df["week"], df["sales"])
ax.set_title("Weekly Sales (2024)")
ax.set_xlabel("Week")
ax.set_ylabel("Sales")
plt.show()
```

**Quarterly Bar: ค่าเฉลี่ยยอดขายรายไตรมาส**

```python
q_summary = df.groupby("quarter")["sales"].mean().reset_index()

fig, ax = plt.subplots(figsize=(6,4))
ax.bar(q_summary["quarter"].astype(str), q_summary["sales"])
ax.set_title("Average Sales by Quarter")
ax.set_xlabel("Quarter")
ax.set_ylabel("Average Sales")
plt.show()
```

**ตีความ:** Q3 มีค่าเฉลี่ยสูงกว่าไตรมาสอื่นอย่างเห็นได้ชัด สนับสนุน key message

 

 3) ลดสิ่งรบกวน (Decluttering)

ลบองค์ประกอบที่ไม่จำเป็น เพื่อลด cognitive load ของผู้ชม

```python
fig, ax = plt.subplots(figsize=(9,4))
ax.plot(df["week"], df["sales"])

# Declutter: ตัดกริดย่อย/เส้นกรอบที่ไม่จำเป็น, ลดจำนวน ticks
ax.grid(False)
for spine in ["top","right"]:
    ax.spines[spine].set_visible(False)
ax.set_xticks(df["week"][::6])  # ทุก 6 สัปดาห์
ax.set_title("Weekly Sales (Decluttered)")
ax.set_xlabel("Week")
ax.set_ylabel("Sales")
plt.show()
```

**แนวคิด:** เก็บเฉพาะสิ่งที่ช่วยให้เข้าใจประเด็นหลักเร็วขึ้น

 

 4) เน้นข้อมูลสำคัญ (Highlighting)

ทำให้ “Q3” โดดเด่นผ่านการแรเงาพื้นหลังและใส่ annotation สั้น ๆ

```python
import matplotlib.dates as mdates

fig, ax = plt.subplots(figsize=(9,4))
ax.plot(df["week"], df["sales"])

# หา boundary ของ Q3
q3_start = df.loc[df["quarter"].eq(3), "week"].min()
q3_end   = df.loc[df["quarter"].eq(3), "week"].max()

# แรเงาพื้นหลังช่วง Q3
ax.axvspan(q3_start, q3_end, alpha=0.15)

# ใส่คำอธิบายสั้น ๆ
peak_wk = df.loc[df["quarter"].eq(3)].sort_values("sales").iloc[-1]
ax.annotate("Q3 surge",
            xy=(peak_wk["week"], peak_wk["sales"]),
            xytext=(peak_wk["week"], peak_wk["sales"]+10),
            arrowprops=dict(arrowstyle="->"))

# ปรับแกนเวลาให้อ่านง่าย
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax.set_title("Weekly Sales with Q3 Highlighted")
ax.set_xlabel("Month")
ax.set_ylabel("Sales")
plt.show()
```

**ตีความ:** ผู้ชมเห็น “ช่วงเวลาที่สำคัญ” ได้ภายในเสี้ยววินาที—นี่คือหัวใจของการเล่าเรื่องด้วยการเน้นจุด


5) เล่าเป็นเรื่องราว (Narrative: Intro → Insight → Conclusion)

* **Introduction:** “ยอดขายรายสัปดาห์ค่อย ๆ เพิ่มขึ้นทั้งปี”
* **Insight:** “แต่ใน Q3 เพิ่มขึ้นแบบโดดเด่นกว่าช่วงอื่น และทำจุดสูงสุดใหม่”
* **Conclusion/Action:** “ควรตรวจสอบแคมเปญ/เหตุการณ์ใน Q3 และพิจารณาขยายงบในกิจกรรมลักษณะเดียวกันสำหรับปีถัดไป”

**ตัวอย่างสไลด์ประกอบ (ข้อความสั้น ๆ):**

* สไลด์ 1: ภาพรวม line chart (decluttered) + Key Message 1 บรรทัด
* สไลด์ 2: Bar เฉลี่ยรายไตรมาส + สรุป “Q3 สูงสุด”
* สไลด์ 3: ปัจจัย/แคมเปญประกอบ + ข้อเสนอการตัดสินใจ


**(เสริม) โชว์ “ก่อน/หลังเล่าเรื่อง” ในกราฟเดียว**

จากกราฟดิบ → กราฟที่ declutter + highlight + annotate

```python
fig, axes = plt.subplots(1,2, figsize=(13,4), sharey=True)

# ก่อน (Raw)
axes[0].plot(df["week"], df["sales"])
axes[0].set_title("Raw Line Chart")
axes[0].set_xlabel("Week"); axes[0].set_ylabel("Sales")

# หลัง (Storytelling)
axes[1].plot(df["week"], df["sales"])
axes[1].axvspan(q3_start, q3_end, alpha=0.15)
axes[1].annotate("Q3 surge",
                 xy=(peak_wk["week"], peak_wk["sales"]),
                 xytext=(peak_wk["week"], peak_wk["sales"]+10),
                 arrowprops=dict(arrowstyle="->"))
axes[1].grid(False)
for spine in ["top","right"]:
    axes[1].spines[spine].set_visible(False)
axes[1].set_title("Storytelling: Focus on Q3")

plt.show()
```

**บทเรียนเชิงหลักการ:**

* **Key Message มาก่อนกราฟ** → กราฟจึงเป็น “หลักฐาน” ที่สนับสนุนข้อความ
* **Decluttering** ลดสิ่งกวนสายตา ให้ insight โดดเด่น
* **Highlight & Annotation** นำสายตาไปยังสารสำคัญ
* **Narrative** ทำให้ข้อมูลกลายเป็น “ข้อเสนอเชิงการตัดสินใจ” ไม่ใช่เพียงภาพสวยงาม



---

### 4. Visualization เป็นเครื่องมือของ EDA

4.1 **ตรวจสอบคุณภาพข้อมูล (Missing values, Outliers)**

**แนวคิด**

* **Missing**: heatmap/บาร์จำนวน NaN ช่วยระบุตำแหน่ง/รูปแบบของการหาย
* **Outliers**: boxplot/violin plot ช่วยเห็นค่าผิดปกติและการกระจาย

**สาธิต**

```python
import pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt
np.random.seed(0)

# สร้างข้อมูลสมมุติ
n = 200
df = pd.DataFrame({
    "sales": np.random.normal(100, 15, n),
    "visits": np.random.poisson(40, n).astype(float),
    "channel": np.random.choice(["Ads","SEO","Email"], n, p=[0.4,0.4,0.2])
})

# สร้าง missing และ outliers
df.loc[np.random.choice(n, 15, replace=False), "visits"] = np.nan
df.loc[np.random.choice(n, 5, replace=False), "sales"] *= 1.8  # outliers high tail

# (ก) Missing matrix แบบง่าย
sns.heatmap(df.isna(), cbar=False)
plt.title("Missingness Map")
plt.show()

# (ข) Boxplot ตรวจ outliers
sns.boxplot(x=df["sales"])
plt.title("Boxplot of Sales (Outliers visible)")
plt.show()
```

**แปลผล**: จุดสว่างใน heatmap บอกคอลัมน์/แถวที่หายบ่อย; หางยาว/จุดนอก whisker ใน boxplot คือ outliers ที่ต้องพิจารณา (ยืนยันด้วยสถิติ/บริบทธุรกิจก่อนตัดสินใจลบหรือตรึงค่า)

4.2 **ค้นหารูปแบบ (seasonality, trend, correlation)**

**แนวคิด**

* **Trend/Seasonality**: line chart ตามเวลา; rolling mean/sd; decomposition
* **Correlation**: heatmap ของค่าสหสัมพันธ์; pairplot ดูรูปทรงความสัมพันธ์

**สาธิต (เวลา & ความสัมพันธ์)**

```python
# สร้าง time series มีแนวโน้ม+ฤดูกาล
rng = pd.date_range("2024-01-01", periods=180, freq="D")
trend = np.linspace(80, 110, len(rng))
season = 8*np.sin(2*np.pi*np.arange(len(rng))/7)  # weekly seasonality
noise = np.random.normal(0, 4, len(rng))
ts = trend + season + noise
tsa = pd.DataFrame({"date": rng, "y": ts})

# Line + rolling mean
tsa["roll7"] = tsa["y"].rolling(7, center=True).mean()
sns.lineplot(data=tsa, x="date", y="y", label="daily")
sns.lineplot(data=tsa, x="date", y="roll7", label="7-day mean")
plt.title("Trend & Weekly Seasonality (smoothed)")
plt.show()

# สัมพันธ์กัน (sales-visits) ด้วย heatmap
df_corr = df[["sales","visits"]].corr()
sns.heatmap(df_corr, annot=True, vmin=-1, vmax=1, cmap="coolwarm")
plt.title("Correlation Heatmap (Sales vs Visits)")
plt.show()

# ดูหลายมิติอย่างรวดเร็ว
sns.pairplot(df[["sales","visits"]])
plt.suptitle("Pairplot: Shape & Relationship", y=1.02)
plt.show()
```

**แปลผล**: เส้น rolling เผยแนวโน้มเพิ่มขึ้นและรูปแบบขึ้นลงประจำสัปดาห์; heatmap ให้ค่าสัมประสิทธิ์ความสัมพันธ์ว่าบวก/ลบ/อ่อนแรงเพียงใด; pairplot ช่วยเห็น nonlinearity หรือกลุ่มย่อย

 
4.3 **สื่อสารกับผู้อื่นได้ง่าย (Communication)**

**แนวคิด**

* ใช้กราฟที่ “ตอบโจทย์คำถามเดียว” ชัดเจน
* ลดสิ่งรบกวน (grid, สีเยอะ, label ซ้ำซ้อน)
* ใส่ **annotation** เฉพาะจุดสำคัญ

 **สาธิต: กราฟเดียวสื่อสารสารหลัก**

```python
ax = sns.lineplot(data=tsa, x="date", y="roll7")
ax.set_title("สัปดาห์นี้ค่าเฉลี่ยพุ่งแตะจุดสูงสุดใหม่")
peak = tsa.loc[tsa["roll7"].idxmax()]
ax.annotate("New high",
            xy=(peak["date"], peak["roll7"]),
            xytext=(peak["date"], peak["roll7"]+3),
            arrowprops=dict(arrowstyle="->"))
sns.despine(); plt.xlabel("Date"); plt.ylabel("7-day mean")
plt.show()
```

**แปลผล**: ผู้ฟังเข้าใจ “สารหลัก” ได้ใน 1 ภาพ โดยตาไปหยุดตรงจุดสำคัญทันที

 

4.4 **เครื่องมือสร้างสมมติฐาน (Hypothesis generation)**

**แนวคิด**

* ใช้กราฟเพื่อ “ตั้งคำถาม” ที่ทดสอบได้ เช่น “เมื่อ visits เกินเกณฑ์หนึ่ง sales จะพุ่งหรือไม่?”, “ฤดูกาลทำให้ผลตอบแทนเปลี่ยนแปลง?”

**สาธิต + ตั้งสมมติฐานจากภาพ**

```python
# แบ่งกลุ่ม visits ต่ำ/สูง แล้วเทียบแจกแจง sales
cut = np.nanmedian(df["visits"])
df["vis_group"] = np.where(df["visits"]>=cut, "HighVisits", "LowVisits")

sns.kdeplot(data=df, x="sales", hue="vis_group", common_norm=False)
plt.title("Sales by Visits Group")
plt.show()
```

**ตีความ & สมมติฐาน**: หากโค้ง HighVisits ขยับไปทางขวาอย่างมีนัยสำคัญ → **H₁**: กลุ่มที่มี visits สูงมีการกระจาย sales สูงกว่า (เฉลี่ย/มัธยฐานสูงกว่า) → นำไปสู่การทดสอบสถิติ (เช่น t-test/Mann–Whitney) หรือการสร้างแบบจำลองในขั้นถัดไป

 

## สรุปแนวทางใช้งาน

* เริ่มที่ **คุณภาพข้อมูล**: หา missing/outliers ด้วย heatmap/boxplot → ตัดสินใจกลยุทธ์ clean
* หา **รูปแบบ**: line/rolling สำหรับเวลา, heatmap/pairplot สำหรับความสัมพันธ์
* **สื่อสาร** ให้ชัด: กราฟเดียวต่อหนึ่งสาร + annotation
* ปิดท้ายด้วย **สมมติฐาน** ที่ทดสอบได้ เพื่อเชื่อม EDA → โมเดล/การทดลองเชิงสถิติ
