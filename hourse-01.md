 

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

 

### **กิจกรรมปฏิบัติ (Hands-On Exercises) – 40 นาที**

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

 

### **C. การนำเข้าข้อมูลจาก WebAPI** 
 
ตัวอย่างการอ่านข้อมูลฝุ่น (PM2.5 / AQI) ประเทศไทยแบบเรียลไทม์โดยใช้ API จากระบบ WAQI (World Air Quality Index) ผ่านเว็บไซต์ aqicn.org แล้วใช้ Python โค้ดในการดึงข้อมูลมาใช้งานครับ

---

## ข้อมูลเบื้องต้น

* เว็บไซต์ **AQICN** มี API สำหรับดึงข้อมูลคุณภาพอากาศจากสถานีต่าง ๆ เช่น สถานีกรุงเทพฯ (Bangkok) ([aqicn.org][1])
* ต้องมี “token” (API key) ที่ลงทะเบียนกับทาง Data Platform ของ WAQI ก่อนถึงจะเรียก API ได้ ([aqicn.org][1])

---

## โครงสร้าง API ตัวอย่าง

URL แบบพื้นฐาน:

```
https://api.waqi.info/feed/@5773/?token=YOUR_TOKEN
```

* `@5773` คือ identifier ของสถานีวัดคุณภาพอากาศกรุงเทพฯ ([aqicn.org][1])
* `token` ใส่ API key ของเราเอง

ตัว API จะคืนข้อมูล JSON ที่มีฟิลด์ต่าง ๆ เช่น ค่าของ PM2.5, PM10, สถานะคุณภาพอากาศ (AQI), ถ้ามี …

---

## โค้ดตัวอย่างใน Python

```python
import requests
import pandas as pd

def fetch_aqi_bangkok(token):
    # URL ของ API
    station_id = "5773"  # Bangkok station (WAQI)
    url = f"https://api.waqi.info/feed/@{station_id}/?token={token}"
    
    # เรียก API
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Error fetching data: HTTP {response.status_code}")
    
    data = response.json()
    if data.get("status") != "ok":
        raise Exception(f"Error in API response: {data.get('data')}")
    
    # ดึงค่าที่สนใจ
    aqi = data["data"].get("aqi")
    dominent_pollutant = data["data"].get("dominentpol")
    iaqi = data["data"].get("iaqi", {})
    pm25 = iaqi.get("pm25", {}).get("v")  # ถ้ามี
    pm10 = iaqi.get("pm10", {}).get("v")
    time = data["data"].get("time", {}).get("s")
    
    # สร้าง DataFrame
    df = pd.DataFrame([{
        "station": "Bangkok",
        "time": time,
        "AQI": aqi,
        "Dominant Pollutant": dominent_pollutant,
        "PM2.5": pm25,
        "PM10": pm10
    }])
    
    return df

if __name__ == "__main__":
    YOUR_TOKEN = "ใส่โทเค็นของคุณที่นี่"
    df_aqi = fetch_aqi_bangkok(YOUR_TOKEN)
    print(df_aqi)
```

 
 

## **อภิปรายสรุป (Wrap-up Discussion) – 5 นาที**

* ความแตกต่างของการนำเข้าข้อมูลจาก **CSV, SQL, NoSQL และ APIs**
* ข้อดี/ข้อจำกัดของแต่ละแหล่งข้อมูล (เช่น ข้อมูลที่มีโครงสร้าง vs ไม่มีโครงสร้าง, คงที่ vs แบบเรียลไทม์)
* แนวปฏิบัติที่ดี: การบันทึกที่มาของข้อมูล การทำให้ซ้ำได้ (reproducibility) และการกำกับดูแลข้อมูล (data governance)

 