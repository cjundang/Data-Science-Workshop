 # ชั่วโมงที่ 1 – การได้มาซึ่งข้อมูลและการจัดการ

## ความรู้เบื้องต้น

1. **บทนำสู่การได้มาซึ่งข้อมูล (Introduction to Data Acquisition)**

* **ความหมาย**: กระบวนการนำข้อมูลดิบมาจากแหล่งต่าง ๆ เพื่อนำไปวิเคราะห์
* **ความสำคัญ**: คุณภาพและความหลากหลายของแหล่งข้อมูลมีผลโดยตรงต่อความแข็งแรงของข้อสรุปและการวิเคราะห์
 

2. **แหล่งข้อมูลทั่วไป (Common Data Sources)**

* **ชุดข้อมูลสาธารณะ (Open datasets)**: Kaggle, UCI ML Repository, พอร์ทัลข้อมูลภาครัฐ
* **APIs**: REST APIs (ผลลัพธ์ในรูปแบบ JSON/CSV), GraphQL APIs
* **Web scraping**: การดึงข้อมูลที่มีโครงสร้างและไม่มีโครงสร้างจากเว็บไซต์ (ต้องคำนึงถึงข้อกฎหมาย/จริยธรรม)
* **ฐานข้อมูล (Databases)**:

  * **ฐานข้อมูลเชิงสัมพันธ์ (Relational: SQL)** – ข้อมูลเชิงตารางแบบมีโครงสร้าง ใช้ SQL ในการดึงข้อมูล
  * **NoSQL** – ฐานข้อมูลแบบเอกสาร (MongoDB), แบบคีย์-ค่า, หรือฐานข้อมูลกราฟ
 

3. **ประเด็นสำคัญที่ควรพิจารณา (Key Considerations)**

* รูปแบบข้อมูล: CSV, JSON, XML, Parquet, Avro
* วิธีการเข้าถึง: HTTP requests, connectors (เช่น SQLAlchemy, pymongo)
* **ประเด็นด้านจริยธรรม**: เคารพข้อตกลงการให้บริการ สิทธิ์การใช้งาน และความเป็นส่วนตัว

 
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

 