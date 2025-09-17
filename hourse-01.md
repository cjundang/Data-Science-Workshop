 

# **ชั่วโมงที่ 1 – การได้มาซึ่งข้อมูลและการจัดการ (Data Acquisition & Management)**

### 1. บทนำสู่การได้มาซึ่งข้อมูล (Introduction to Data Acquisition)

* **ความหมาย**: กระบวนการนำข้อมูลดิบมาจากแหล่งต่าง ๆ เพื่อนำไปวิเคราะห์
* **ความสำคัญ**: คุณภาพและความหลากหลายของแหล่งข้อมูลมีผลโดยตรงต่อความแข็งแรงของข้อสรุปและการวิเคราะห์
 

### 2. แหล่งข้อมูลทั่วไป (Common Data Sources)

* **ชุดข้อมูลสาธารณะ (Open datasets)**: Kaggle, UCI ML Repository, พอร์ทัลข้อมูลภาครัฐ
* **APIs**: REST APIs (ผลลัพธ์ในรูปแบบ JSON/CSV), GraphQL APIs
* **Web scraping**: การดึงข้อมูลที่มีโครงสร้างและไม่มีโครงสร้างจากเว็บไซต์ (ต้องคำนึงถึงข้อกฎหมาย/จริยธรรม)
* **ฐานข้อมูล (Databases)**:

  * **ฐานข้อมูลเชิงสัมพันธ์ (Relational: SQL)** – ข้อมูลเชิงตารางแบบมีโครงสร้าง ใช้ SQL ในการดึงข้อมูล
  * **NoSQL** – ฐานข้อมูลแบบเอกสาร (MongoDB), แบบคีย์-ค่า, หรือฐานข้อมูลกราฟ
 

### 3. ประเด็นสำคัญที่ควรพิจารณา (Key Considerations)

* รูปแบบข้อมูล: CSV, JSON, XML, Parquet, Avro
* วิธีการเข้าถึง: HTTP requests, connectors (เช่น SQLAlchemy, pymongo)
* **ประเด็นด้านจริยธรรม**: เคารพข้อตกลงการให้บริการ สิทธิ์การใช้งาน และความเป็นส่วนตัว

 

## **กิจกรรมปฏิบัติ (Hands-On Exercises) – 40 นาที**

### **A. การดึงข้อมูลจาก Open API**

**วัตถุประสงค์**: สอนนักศึกษาให้สามารถดึงข้อมูลสดจาก API และโหลดเข้าสู่ pandas

#### ตัวอย่าง: OpenWeather API (ต้องใช้ API key ฟรี)

```python
import requests
import pandas as pd

# ตัวอย่าง: ดึงข้อมูลสภาพอากาศของกรุงเทพฯ
API_KEY = "YOUR_API_KEY"  # เตรียม API key ล่วงหน้า
city = "Bangkok"
url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"

response = requests.get(url)
data = response.json()

# ดูข้อมูล JSON ที่ได้มา
print(data)

# แปลงบางฟิลด์มาเก็บใน pandas DataFrame
weather = {
    "city": data["name"],
    "temperature": data["main"]["temp"],
    "humidity": data["main"]["humidity"],
    "weather": data["weather"][0]["description"]
}
df_weather = pd.DataFrame([weather])
print(df_weather)
```

**ประเด็นสอนสำคัญ**:

* โครงสร้างข้อมูล JSON
* วิธีการดึงค่าจากฟิลด์ที่ซ้อนกัน
* การแปลง JSON เป็นตาราง DataFrame ของ pandas

 

### **B. การนำเข้าข้อมูลจาก CSV**

**วัตถุประสงค์**: ทำงานกับข้อมูลเชิงตารางที่พบได้บ่อย

```python
import pandas as pd

# โหลดข้อมูลจากไฟล์ CSV
df_csv = pd.read_csv("sample_data.csv")

# สำรวจข้อมูล
print(df_csv.head())
print(df_csv.info())
print(df_csv.describe())
```

 
### **C. การนำเข้าข้อมูลจาก SQL**

**วัตถุประสงค์**: แสดงการเชื่อม SQL เข้ากับ Python

```python
import sqlite3
import pandas as pd

# เชื่อมต่อกับฐานข้อมูล SQLite
conn = sqlite3.connect("sample_db.sqlite")

# ดึงข้อมูลตัวอย่าง
query = "SELECT * FROM customers LIMIT 10;"
df_sql = pd.read_sql_query(query, conn)

print(df_sql.head())
conn.close()
```
 

### **D. การนำเข้าข้อมูลจาก NoSQL (MongoDB)**

**วัตถุประสงค์**: สาธิตการเชื่อมต่อฐานข้อมูลแบบเอกสาร

```python
from pymongo import MongoClient
import pandas as pd

# เชื่อมต่อ MongoDB (ต้องเปิด MongoDB ไว้ หรือใช้ MongoDB Atlas)
client = MongoClient("mongodb://localhost:27017/")
db = client["sample_db"]
collection = db["customers"]

# ดึงเอกสารตัวอย่าง 10 รายการแรก
docs = list(collection.find().limit(10))

# แปลงเป็น DataFrame
df_nosql = pd.DataFrame(docs)
print(df_nosql.head())
```

 

## **อภิปรายสรุป (Wrap-up Discussion) – 5 นาที**

* ความแตกต่างของการนำเข้าข้อมูลจาก **CSV, SQL, NoSQL และ APIs**
* ข้อดี/ข้อจำกัดของแต่ละแหล่งข้อมูล (เช่น ข้อมูลที่มีโครงสร้าง vs ไม่มีโครงสร้าง, คงที่ vs แบบเรียลไทม์)
* แนวปฏิบัติที่ดี: การบันทึกที่มาของข้อมูล การทำให้ซ้ำได้ (reproducibility) และการกำกับดูแลข้อมูล (data governance)

 