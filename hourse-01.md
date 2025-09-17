
# **Hour 1 – Data Acquisition & Management**

### 1. Introduction to Data Acquisition

* **Definition**: The process of obtaining raw data from different sources for analysis.
* **Why it matters**: The quality and diversity of data sources influence the strength of your insights.

### 2. Common Data Sources

* **Open datasets**: Kaggle, UCI ML Repository, government open data portals.
* **APIs**: REST APIs (JSON/CSV responses), GraphQL APIs.
* **Web scraping**: Extracting structured/unstructured data from websites (legal/ethical caveats).
* **Databases**:

  * **Relational (SQL)**: structured, tabular data, queried with SQL.
  * **NoSQL**: document-based (MongoDB), key-value stores, graph databases.

### 3. Key Considerations

* Data format: CSV, JSON, XML, Parquet, Avro.
* Access method: HTTP requests, connectors (SQLAlchemy, pymongo).
* **Ethical aspects**: respect terms of service, licensing, and privacy.


## **Hands-On Exercises (40 min)**

### **A. Querying an Open API**

**Objective**: Teach students how to fetch live data from an API and load it into pandas.

#### Example: OpenWeather API (requires free API key)

```python
import requests
import pandas as pd

# Example: Weather data for Bangkok
API_KEY = "YOUR_API_KEY"  # teacher prepares ahead
city = "Bangkok"
url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"

response = requests.get(url)
data = response.json()

# Inspect raw JSON
print(data)

# Convert selected fields to pandas DataFrame
weather = {
    "city": data["name"],
    "temperature": data["main"]["temp"],
    "humidity": data["main"]["humidity"],
    "weather": data["weather"][0]["description"]
}
df_weather = pd.DataFrame([weather])
print(df_weather)
```

**Key teaching points**:

* Structure of JSON data
* Extracting nested fields
* Transforming JSON into tabular pandas DataFrame


### **B. Importing CSV Data**

**Objective**: Work with common tabular data.

```python
import pandas as pd

# Load dataset from local CSV
df_csv = pd.read_csv("sample_data.csv")

# Explore dataset
print(df_csv.head())
print(df_csv.info())
print(df_csv.describe())
```


### **C. Importing SQL Data**

**Objective**: Teach SQL integration with Python.

```python
import sqlite3
import pandas as pd

# Create connection to SQLite database
conn = sqlite3.connect("sample_db.sqlite")

# Example query
query = "SELECT * FROM customers LIMIT 10;"
df_sql = pd.read_sql_query(query, conn)

print(df_sql.head())
conn.close()
```


### **D. Importing NoSQL Data (MongoDB)**

**Objective**: Demonstrate connection to a document database.

```python
from pymongo import MongoClient
import pandas as pd

# Connect to MongoDB (ensure MongoDB is running or use MongoDB Atlas)
client = MongoClient("mongodb://localhost:27017/")
db = client["sample_db"]
collection = db["customers"]

# Fetch first 10 documents
docs = list(collection.find().limit(10))

# Convert to DataFrame
df_nosql = pd.DataFrame(docs)
print(df_nosql.head())
```

 
## **Wrap-up Discussion (5 min)**

* Differences in importing from **CSV vs SQL vs NoSQL vs APIs**
* Advantages/disadvantages of each source (e.g., structured vs unstructured, static vs dynamic)
* Best practices: documenting sources, ensuring reproducibility, data governance.

 
