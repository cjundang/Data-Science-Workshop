 # ชั่วโมงที่ 1 – การได้มาซึ่งข้อมูลและการจัดการ
## Learning Outcomes

1. **เข้าใจแนวคิดและแหล่งข้อมูลที่หลากหลาย**

   * ผู้เรียนสามารถอธิบายความหมาย ความสำคัญ และแหล่งข้อมูลทั่วไป (Open Data, APIs, Web Scraping, Databases, IoT/Sensors) ได้
   * ตระหนักถึงประเด็นสำคัญที่ต้องพิจารณา เช่น รูปแบบข้อมูล วิธีการเข้าถึง และประเด็นจริยธรรมในการใช้ข้อมูล

2. **สามารถนำเข้าข้อมูลจากแหล่งต่าง ๆ ได้จริง**

   * ผู้เรียนสามารถเขียนโค้ดเพื่อนำเข้าข้อมูลจาก **CSV, SQL, NoSQL, และ APIs**
   * สามารถแปลงโครงสร้างข้อมูล (เช่น JSON → DataFrame) และทำการสำรวจ (explore) ข้อมูลเบื้องต้น

3. **ประยุกต์ใช้เครื่องมือในการเชื่อมโยงข้อมูลสู่แพลตฟอร์มที่ใช้จริง**

   * ผู้เรียนสามารถเชื่อม API กับ **Google Sheets** เพื่อสร้างแดชบอร์ดข้อมูลสด
   * ทดลองใช้ Google Sheets เป็น **Web API (database-like)** และเรียกใช้งานผ่าน Python ได้


## ความรู้เบื้องต้น

### 1. Introduction to Data Acquisition

**การได้มาซึ่งข้อมูล (Data Acquisition)** หมายถึง กระบวนการรวบรวม นำเข้า 
หรือดึงข้อมูลดิบจากหลากหลายแหล่ง เพื่อใช้ในการวิเคราะห์และประมวลผลต่อไป ข้อมูลเหล่านี้อาจอยู่ในรูปแบบที่แตกต่างกัน เช่น ข้อมูลเชิงตาราง (CSV, Excel, Database), ข้อมูลกึ่งมีโครงสร้าง (JSON, XML, API) หรือข้อมูลไม่มีโครงสร้าง (ข้อความ รูปภาพ วิดีโอ สัญญาณเซนเซอร์)

### 2. ความสำคัญ (Significance)

1. **คุณภาพของข้อมูล (Data Quality)**

   * หากข้อมูลมีความถูกต้อง (accuracy), ครบถ้วน (completeness), สอดคล้อง (consistency) และทันเวลา (timeliness) จะช่วยให้ผลการวิเคราะห์เชื่อถือได้
   * ข้อมูลที่มีคุณภาพต่ำ เช่น ขาดหาย (missing values), มี noise มาก หรือไม่อัปเดต จะนำไปสู่ข้อสรุปที่ผิดพลาด

2. **ความหลากหลายของข้อมูล (Data Variety)**

   * ข้อมูลจากหลายแหล่งช่วยให้ได้มุมมองที่ครอบคลุมมากขึ้น
   * เช่น การวิเคราะห์มลพิษทางอากาศ หากใช้เฉพาะข้อมูลจากเครื่องวัดเพียงเครื่องเดียวอาจไม่สะท้อนสภาพจริง แต่หากรวมข้อมูลจากหลายสถานี พร้อมกับข้อมูลสภาพอากาศ (humidity, wind speed) และข้อมูลกิจกรรมมนุษย์ (การจราจร, อุตสาหกรรม) จะได้ภาพรวมที่แม่นยำกว่า

  3. **ความแข็งแรงของข้อสรุป (Strength of Insights)**

    * ข้อมูลที่ดีและมีหลากหลายจะช่วยให้โมเดลทางสถิติหรือ Machine Learning มีประสิทธิภาพสูงขึ้น
    * สามารถสร้างข้อเสนอแนะหรือการตัดสินใจเชิงนโยบายที่น่าเชื่อถือ

### 3. ตัวอย่างแหล่งข้อมูล (Examples of Data Sources)

* **ข้อมูลสาธารณะ (Open Data)**: ข้อมูลจากหน่วยงานรัฐ, Kaggle, UCI ML Repository
* **API (Application Programming Interface)**: ข้อมูลเรียลไทม์ เช่น สภาพอากาศ ราคาหุ้น ข่าวสาร
* **Web Scraping**: การดึงข้อมูลจากหน้าเว็บไซต์ (ต้องคำนึงถึงข้อกฎหมายและจริยธรรม)
* **ฐานข้อมูลภายในองค์กร**: เช่น ข้อมูลลูกค้า (CRM), ข้อมูลธุรกรรม, Log Files
* **IoT และ Sensor Data**: ข้อมูลจากอุปกรณ์ตรวจวัด เช่น เครื่องวัดฝุ่น PM2.5, เครื่องวัดอุณหภูมิ


การได้มาซึ่งข้อมูลเป็น **ขั้นตอนแรกและสำคัญที่สุด** ของกระบวนการวิเคราะห์ข้อมูลหรือ Data Science Pipeline เพราะหากข้อมูลที่ได้มาไม่ถูกต้อง ไม่ครบถ้วน หรือไม่เกี่ยวข้อง การวิเคราะห์ขั้นต่อไปจะสูญเสียคุณค่า ดังนั้น นักวิจัยและนักวิเคราะห์จำเป็นต้องให้ความสำคัญกับ **การคัดเลือกแหล่งข้อมูลที่เหมาะสม และการตรวจสอบคุณภาพข้อมูลอย่างเคร่งครัด**


## **แหล่งข้อมูลทั่วไป (Common Data Sources)**

 

 
### 1. ชุดข้อมูลสาธารณะ (Open Datasets)

* **ความหมาย**: ชุดข้อมูลที่องค์กร หน่วยงานรัฐ หรือชุมชนวิจัยเปิดให้ใช้งานได้ฟรี เพื่อการศึกษา วิจัย หรือการพัฒนาโมเดล
* **ตัวอย่าง**:

  * **Kaggle** – แพลตฟอร์มประกวด Machine Learning ที่มีชุดข้อมูลหลากหลาย เช่น การแพทย์ เศรษฐกิจ กีฬา
  * **UCI ML Repository** – แหล่งรวมชุดข้อมูลสำหรับ Machine Learning มายาวนาน
  * **Open Government Data Portals** – ข้อมูลสถิติประชากร สิ่งแวดล้อม คมนาคม
* **ข้อดี**: เข้าถึงง่าย ใช้เป็น baseline dataset ในการเรียนรู้และทดลองโมเดล
* **ข้อจำกัด**: อาจไม่อัปเดต ไม่ตรงตามโจทย์วิจัยจริงเสมอ

 
### 2. APIs (Application Programming Interfaces)

* **REST APIs**: ส่งและรับข้อมูลผ่าน HTTP โดยใช้ JSON, CSV หรือ XML

  * ตัวอย่าง: OpenWeather API (ข้อมูลอากาศ), Alpha Vantage (ข้อมูลหุ้น)
* **GraphQL APIs**: ให้ผู้ใช้ query เฉพาะฟิลด์ที่ต้องการ ลดปริมาณข้อมูลที่ส่ง

  * ตัวอย่าง: GitHub GraphQL API (ข้อมูล repository, issues)
* **ข้อดี**: ได้ข้อมูลสด (real-time หรือ near real-time)
* **ข้อจำกัด**: ต้องมี API key, rate limit, และบางครั้งมีค่าใช้จ่าย

 
### 3. Web Scraping

* **ความหมาย**: การดึงข้อมูลโดยตรงจากเว็บไซต์ที่ไม่ได้เปิด API
* **เครื่องมือ**: BeautifulSoup, Scrapy, Selenium
* **ประเภทข้อมูล**:

  * ข้อมูลที่มีโครงสร้าง (เช่น ตาราง HTML)
  * ข้อมูลไม่มีโครงสร้าง (เช่น ข้อความ ข่าว บทความ)
* **ข้อควรระวัง**:

  * ข้อกฎหมาย: ต้องตรวจสอบ Terms of Service
  * ประสิทธิภาพ: เว็บไซต์อาจเปลี่ยนโครงสร้าง DOM ทำให้โค้ดใช้ไม่ได้

 
### 4. ฐานข้อมูล (Databases)

#### 4.1 ฐานข้อมูลเชิงสัมพันธ์ (Relational Databases – SQL)

* ใช้ **SQL (Structured Query Language)** ในการ query ข้อมูล
* เก็บข้อมูลในรูปแบบตาราง (rows & columns)
* ตัวอย่าง: MySQL, PostgreSQL, Oracle, Microsoft SQL Server
* เหมาะกับข้อมูลเชิงธุรกรรม (transactional data) ที่มีโครงสร้างชัดเจน

#### 4.2 ฐานข้อมูลแบบ NoSQL

* เหมาะกับข้อมูลที่มีความหลากหลายและไม่เป็นโครงสร้างตายตัว
* ประเภทหลัก:

  * **Document-based**: เช่น MongoDB (เก็บเป็น JSON document)
  * **Key-Value Stores**: เช่น Redis (เก็บข้อมูลในรูปแบบคู่ key–value)
  * **Graph Databases**: เช่น Neo4j (เก็บความสัมพันธ์ระหว่าง entity)
* ใช้มากในงาน Big Data และระบบที่ต้องการ scale ขนาดใหญ่

 
ดีมากครับ 🙌 ด้านล่างนี้คือการขยายความเชิงทฤษฎีสำหรับหัวข้อ

### **ประเด็นสำคัญที่ควรพิจารณา (Key Considerations)**


#### 1. รูปแบบข้อมูล (Data Formats)

ข้อมูลที่ได้มามีหลายรูปแบบ ซึ่งมีผลต่อวิธีการนำเข้า ประมวลผล และจัดเก็บ

* **CSV (Comma-Separated Values)**

  * เป็นไฟล์ข้อความธรรมดาที่เก็บข้อมูลในรูปแบบตาราง โดยใช้เครื่องหมายจุลภาค (,) หรือเครื่องหมายอื่น ๆ เป็นตัวแบ่งค่า
  * **ข้อดี**: เข้าใจง่าย, รองรับเครื่องมือหลากหลาย (Excel, pandas)
  * **ข้อจำกัด**: ไม่เหมาะกับข้อมูลที่มีโครงสร้างซ้อนกันหรือตารางที่ซับซ้อน

  ลักษณะ: ข้อมูลเชิงตาราง โดยใช้เครื่องหมาย `,` คั่นค่าในแต่ละคอลัมน์

```csv
City,Temperature_C,Humidity_Percent,Weather
Bangkok,32.5,70,Sunny
Chiang Mai,28.3,80,Rainy
Phuket,30.1,85,Cloudy
```


* **JSON (JavaScript Object Notation)**

  * รูปแบบข้อมูลเชิงกุญแจ–ค่า (key-value) ที่สามารถแทนโครงสร้างซ้อนกันได้
  * **ข้อดี**: เหมาะสำหรับ API, ยืดหยุ่น, อ่านง่าย
  * **ข้อจำกัด**: ไฟล์อาจมีขนาดใหญ่ ไม่เหมาะกับการเก็บข้อมูลตารางขนาดมหาศาล

ลักษณะ: ข้อมูลแบบกุญแจ–ค่า (key-value) และรองรับโครงสร้างซ้อนกันได้

```json
{
  "City": "Bangkok",
  "Temperature_C": 32.5,
  "Humidity_Percent": 70,
  "Weather": {
    "Description": "Sunny",
    "Wind_Speed_kmh": 12.5
  }
}
```

* **XML (eXtensible Markup Language)**

  * ใช้แท็ก (tag) ในการห่อหุ้มข้อมูล คล้าย HTML
  * **ข้อดี**: มีโครงสร้างชัดเจน, มาตรฐานใช้ในระบบเดิม (legacy systems)
  * **ข้อจำกัด**: มีความซับซ้อน, ข้อมูลซ้ำซ้อนมากกว่า JSON

ลักษณะ: ใช้แท็กห่อหุ้มข้อมูล คล้าย HTML

```xml
<WeatherRecord>
  <City>Bangkok</City>
  <Temperature_C>32.5</Temperature_C>
  <Humidity_Percent>70</Humidity_Percent>
  <Weather>
    <Description>Sunny</Description>
    <Wind_Speed_kmh>12.5</Wind_Speed_kmh>
  </Weather>
</WeatherRecord>
```

 


#### 2. วิธีการเข้าถึง (Access Methods)

* **HTTP Requests**

  * ใช้สำหรับดึงข้อมูลจาก **Web APIs** เช่น REST/GraphQL
  * ตัวอย่าง: ใช้ `requests` ใน Python เรียก API ที่ส่ง JSON กลับมา

* **Connectors / Libraries**

  * ใช้เชื่อมต่อกับฐานข้อมูลโดยตรง เช่น

    * **SQLAlchemy** → ORM เชื่อมต่อ SQL databases (MySQL, PostgreSQL)
    * **pymongo** → เชื่อมต่อ MongoDB (NoSQL)
    * **ODBC/JDBC** → เชื่อมต่อฐานข้อมูลผ่านมาตรฐานกลาง
  * ทำให้นักวิจัยสามารถเข้าถึงข้อมูลได้โดยไม่ต้องจัดการ low-level driver เอง


#### 3. ประเด็นด้านจริยธรรม (Ethical Considerations)

* **ข้อตกลงการให้บริการ (Terms of Service)**

  * ก่อนใช้ API หรือ web scraping ต้องตรวจสอบเงื่อนไขของผู้ให้บริการ
  * เช่น บางเว็บห้ามดึงข้อมูลอัตโนมัติ

* **สิทธิ์การใช้งาน (Licensing)**

  * ข้อมูลบางชุดอาจอนุญาตให้ใช้เพื่อการศึกษาเท่านั้น ไม่สามารถใช้เชิงพาณิชย์ได้

* **ความเป็นส่วนตัว (Privacy)**

  * ต้องเคารพข้อมูลส่วนบุคคล เช่น ข้อมูลผู้ใช้, พิกัดตำแหน่ง
  * ต้องปฏิบัติตามกฎหมายที่เกี่ยวข้อง เช่น GDPR (EU), PDPA (ไทย)



การได้มาซึ่งข้อมูลไม่ใช่เพียงแค่ “ดึงข้อมูลมาใช้” เท่านั้น แต่ต้องพิจารณา **รูปแบบข้อมูล**, **วิธีการเข้าถึง**, และ **จริยธรรม** เพื่อให้ได้ข้อมูลที่ถูกต้อง มีคุณภาพ และใช้งานได้อย่างถูกกฎหมาย


 
## Hands-On Exercises

### A. การนำเข้าข้อมูลจาก CSV

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

### B. การดึงข้อมูลจาก Open API

**วัตถุประสงค์**: สอนนักศึกษาให้สามารถดึงข้อมูลสดจาก API และโหลดเข้าสู่ pandas

**ตัวอย่าง: OpenWeather API (ต้องใช้ API key ฟรี)**
```python
import requests
import pandas as pd
from google.colab import userdata

# ตัวอย่าง: ดึงข้อมูลสภาพอากาศของกรุงเทพฯ
city = "Bangkok"
OPEN_API_KEY = userdata.get('OPEN_API_KEY')
url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPEN_API_KEY}&units=metric"

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


#### C. การนำเข้าข้อมูลจาก WebAPI
 
* เว็บไซต์ **AQICN** มี API สำหรับดึงข้อมูลคุณภาพอากาศจากสถานีต่าง ๆ เช่น สถานีกรุงเทพฯ (Bangkok) 
* ต้องมี “token” (API key) ที่ลงทะเบียนกับทาง Data Platform ของ WAQI ก่อนถึงจะเรียก API ได้  

**โค้ดตัวอย่างใน Python**

```python
import requests
import pandas as pd

def fetch_aqi_bangkok(station_id, token):
    # URL ของ API
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

# Used to securely store your API key
from google.colab import userdata

# Replace 'WAQI_YOUR_TOKEN' with the name you used for the secret in Colab
WAQI_YOUR_TOKEN = userdata.get('WAQI_YOUR_TOKEN')

# Now you can use WAQI_YOUR_TOKEN in your code
print(WAQI_YOUR_TOKEN)

# WAQI_YOUR_TOKEN = "ใส่โทเค็นของคุณที่นี่"
station_id = "5773"  # Bangkok station (WAQI)
df_aqi = fetch_aqi_bangkok(station_id, WAQI_YOUR_TOKEN)
print(df_aqi)
```
 
#### C. การใช้ GET API ใน Google Sheets

1. **ทำไมต้องใช้ Google Sheets?**

   * เป็นเครื่องมือที่เบา ใช้ง่าย เหมาะสำหรับการทำงานร่วมกันและแชร์ข้อมูล
   * สามารถทำงานเป็นแดชบอร์ดแบบสดเมื่อเชื่อมต่อกับ API

2. **การเชื่อม API ใน Google Sheets**

   * ใช้ **Google Apps Script** (ภาษา JavaScript) เพื่อรันคำสั่ง API request
   * เก็บค่าที่ได้ลงในเซลล์ หรือกำหนดให้รีเฟรชอัตโนมัติได้

3. **ตัวอย่างการใช้งาน (Use Cases)**

   * การอัปเดตข้อมูลสภาพอากาศแบบเรียลไทม์
   * ราคาหุ้น/คริปโต
   * แดชบอร์ดติดตาม COVID-19


**สคริปต์ปฏิบัติ (Hands-On Script)**

1. เปิด Script Editor ของ Google Sheets

* ไปที่ **Extensions > Apps Script**
* นำโค้ดด้านล่างไปแทนที่โค้ดเดิม

2. ตัวอย่าง – ดึงข้อมูลจาก OpenWeather API

```javascript
function getWeatherData() {
  var city = "Bangkok";
  var apiKey = "YOUR_API_KEY";  // ใส่ API key ของคุณ
  var url = "https://api.openweathermap.org/data/2.5/weather?q=" 
            + city + "&appid=" + apiKey + "&units=metric";

  // ดึงข้อมูลจาก API
  var response = UrlFetchApp.fetch(url);
  var data = JSON.parse(response.getContentText());

  // แยกค่าที่ต้องการ
  var temperature = data.main.temp;
  var humidity = data.main.humidity;
  var description = data.weather[0].description;

  // เขียนค่าลงในชีต (แถว 1 = header, แถว 2 = ข้อมูล)
  var sheet = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet();
  sheet.getRange("A1").setValue("City");
  sheet.getRange("B1").setValue("Temperature (C)");
  sheet.getRange("C1").setValue("Humidity (%)");
  sheet.getRange("D1").setValue("Description");

  sheet.getRange("A2").setValue(city);
  sheet.getRange("B2").setValue(temperature);
  sheet.getRange("C2").setValue(humidity);
  sheet.getRange("D2").setValue(description);
}
```

3. รันสคริปต์

* กดบันทึก → คลิก **Run** → อนุญาตการเข้าถึง (grant permissions)
* Google Sheet จะถูกเติมข้อมูลสภาพอากาศสดทันที


4. ตั้งค่ารีเฟรชอัตโนมัติทุกชั่วโมง

เพิ่ม trigger ใน Apps Script:

* ไปที่ **Triggers > Add Trigger**
* เลือกฟังก์ชัน `getWeatherData`
* ตั้งค่า **Time-driven → Every hour**

ทำให้ชีตอัปเดตข้อมูลอัตโนมัติทุกชั่วโมง
 
#### D. Google Sheets as Web API (Database-like)

**แนวคิด**

1. **Google Sheets = Lightweight Database**

   * ใช้เก็บข้อมูลตารางเล็ก–กลาง
   * ใช้ง่าย แชร์ง่าย ไม่ต้องมี server เอง

2. **Apps Script Web App**

   * ใช้ `doGet(e)` และ `doPost(e)` เพื่อเปิด endpoint
   * สามารถรับ parameter และส่ง JSON กลับมาได้

3. **Use Cases**

   * เก็บแบบฟอร์มที่ไม่ได้ผ่าน Google Form
   * Mock database สำหรับ prototype แอป
   * API ให้กับ dashboard หรือ mobile app


**โค้ดตัวอย่าง (Apps Script)**

1. เปิด Script Editor

* Google Sheets → **Extensions > Apps Script**

2. ใส่โค้ด

```javascript
// ฟังก์ชันเมื่อมีการเรียก GET API
function doGet(e) {
  var sheet = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet();
  var rows = sheet.getDataRange().getValues();
  var headers = rows[0];
  var data = [];

  for (var i = 1; i < rows.length; i++) {
    var row = {};
    for (var j = 0; j < headers.length; j++) {
      row[headers[j]] = rows[i][j];
    }
    data.push(row);
  }

  return ContentService
    .createTextOutput(JSON.stringify(data))
    .setMimeType(ContentService.MimeType.JSON);
}

// ฟังก์ชันเมื่อมีการเรียก POST API
function doPost(e) {
  var sheet = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet();
  var data = JSON.parse(e.postData.contents);

  // เพิ่มแถวใหม่ตามข้อมูล JSON ที่ส่งมา
  sheet.appendRow([data.name, data.age, data.city]);

  return ContentService
    .createTextOutput(JSON.stringify({status: "success", data: data}))
    .setMimeType(ContentService.MimeType.JSON);
}
```

**การใช้งาน**

1. Deploy Web App

* ไปที่ **Deploy > New Deployment > Web App**
* เลือก **Anyone with the link** → Copy URL

2. เรียกใช้ API

* **GET ข้อมูลทั้งหมด**

  ```
  https://script.google.com/macros/s/xxxxxx/exec
  ```

* **POST เพิ่มข้อมูลใหม่** (ตัวอย่างใช้ cURL หรือ Postman)

  ```bash
  curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"City": "Bangkok","Temperature (C)": "29.58", "Humidity (%)": "76", "Description": "light rain"}' \
  "https://script.google.com/macros/s/xxxxxxxxx/exec"
  ```


**ตัวอย่างการตอบกลับ JSON (GET)**

```json
[{
  City: "Bangkok",
  Temperature (C): 29.58,
  Humidity (%): 76,
  Description: "light rain"
}]
```
 

**Python Demo: ใช้ Google Sheet เป็น Database ผ่าน Web API**

> ⚠️ ก่อนเริ่ม: ต้อง Deploy Google Apps Script เป็น **Web App (Anyone with the link)** แล้ว copy URL เช่น
> `https://script.google.com/macros/s/AKfycbxxxxx/exec`
 

1. GET: ดึงข้อมูลทั้งหมดจาก Google Sheet

```python
import requests
import pandas as pd

# URL ที่ได้จากการ deploy Apps Script
BASE_URL = "https://script.google.com/macros/s/AKfycbxxxxx/exec"

# เรียก GET
response = requests.get(BASE_URL)

if response.status_code == 200:
    data = response.json()
    # แปลงเป็น DataFrame เพื่อดูเป็นตาราง
    df = pd.DataFrame(data)
    print(df)
else:
    print("Error:", response.status_code, response.text)
```

✅ ผลลัพธ์จะเป็นข้อมูลจากชีตในรูปแบบ pandas DataFrame เช่น:

```
      City  Temperature (C)  Humidity (%) Description
0  Bangkok            29.58            76  light rain
```

2. POST: เพิ่มข้อมูลใหม่ลงในชีต

```python
import requests

# URL ของ Web App
BASE_URL = "https://script.google.com/macros/s/xxxxxxxxxxxxx/exec"

# JSON ที่ต้องการเพิ่ม
new_data = {
    "City": "NST",
    "Temperature (C)": 28,
    "Humidity (%)": "99.0",
    "Description": "Big Rain"
}

# ส่ง POST
response = requests.post(BASE_URL, json=new_data)

if response.status_code == 200:
    print("Response:", response.json())
else:
    print("Error:", response.status_code, response.text)
```

เมื่อรันแล้ว Google Sheet จะมีข้อมูลแถวใหม่เพิ่มทันที เช่น `Charlie | 28 | Phuket`

 