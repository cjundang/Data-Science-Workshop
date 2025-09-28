---
marp: true
theme: default
paginate: true
---

# 📊 รายงานสรุปผลการศึกษา  
**การพยากรณ์ยอดขายไอศกรีมรายวัน**  
ปัจจัย: สภาพอากาศ, พฤติกรรมลูกค้า, การตลาด

---

## 🎯 วัตถุประสงค์
- สร้างแบบจำลองพยากรณ์ยอดขายรายวัน  
- วิเคราะห์ปัจจัยที่ส่งผลต่อยอดขาย  
- นำไปใช้กำหนด **กลยุทธ์การตลาดและการวางแผนธุรกิจ**

---

## 🗂 ข้อมูลที่ใช้
- **จำนวนตัวอย่าง:** 200 วัน  
- **Features:**  
  - temperature_C, humidity_pct  
  - month, is_weekend, is_holiday  
  - promo_budget_thb  
  - foot_traffic, prior_day_sales  
- **Target:** sales_thb (ยอดขายรายวัน)

---

## 🔍 การวิเคราะห์
1. **EDA**:  
   - อุณหภูมิ, foot traffic, prior_day_sales → สัมพันธ์เชิงบวก  
   - ความชื้นสูง → กดยอดขายลง  
   - วันหยุด/โปรโมชัน → ดันยอดขายเพิ่ม  

2. **โมเดลที่ใช้**:  
   - Linear, Polynomial Regression  
   - Random Forest, Gradient Boosting  

---

## 📈 ผลการประเมินโมเดล
- **Linear Regression:** R² ~ 0.75  
- **Polynomial Regression:** R² ~ 0.80  
- **Random Forest:** R² > 0.90  
- **Gradient Boosting:** R² > 0.90  

✅ **Tree-based models** แม่นยำสูงสุด  

---

## 🌟 Feature Importance
**Top factors (Random Forest/GBM):**  
1. temperature_C (~31%)  
2. promo_budget_thb (~30%)  
3. prior_day_sales (~14%)  
4. humidity_pct (~11%)  
5. foot_traffic (~7%)  
👉 Holiday/Weekend/Month → ผลกระทบน้อย

---

## 🔎 SHAP Analysis
- **temperature_C สูง → ดันยอดขายขึ้น**  
- **promo_budget_thb สูง → เพิ่มยอดขายมาก**  
- **prior_day_sales สูง → ทำนายว่ายอดขายวันถัดไปสูง**  
- **humidity สูง → กดยอดขายลง**  
- **holiday → ดันยอดขายเล็กน้อย**

---

## 💡 สรุปเชิงกลยุทธ์
1. **Hot Weather + Promotion = ยอดขายพุ่ง**  
2. **งบโปรโมชันมีผลเกือบเท่าอากาศ**  
3. ใช้ **prior_day_sales** สำหรับ demand planning  
4. ความชื้นสูง → โปรโมทสินค้า cold drink แทนไอศกรีม  

---

## 🚀 แนวทางต่อยอด
- ใช้ **XGBoost / LightGBM** สำหรับ real-time forecast  
- ผสาน **พยากรณ์อากาศล่วงหน้า** เพื่อวางสต็อกและ manpower  
- ระบบ **Dynamic Promotion** ตามอากาศและ traffic  

---

# ✅ สรุป
- **Tree-based models (RF/GBM) → แม่นยำที่สุด**  
- ปัจจัยหลัก: **อุณหภูมิ, โปรโมชัน, ยอดขายวันก่อนหน้า**  
- ธุรกิจสามารถ **เพิ่มยอดขายได้** ด้วย  
  - โปรเชิงรุกในวันที่อากาศร้อน  
  - การเตรียมทรัพยากรช่วงวันหยุด  

