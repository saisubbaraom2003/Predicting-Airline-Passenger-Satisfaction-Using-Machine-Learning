# ✈️ Airline Passenger Satisfaction Prediction

This machine learning project predicts airline passenger satisfaction ("Satisfied" or "Neutral or Dissatisfied") based on various demographic, travel, and service-related features. The model is deployed on **Render**, and was previously tested on **Azure** and **AWS** during free trials.

## 📊 Project Overview

- **Objective**: Classify airline passengers based on their satisfaction levels using supervised learning.
- **Dataset Size**: 129,880 rows × 24 columns
- **ML Techniques**: Data cleaning, feature engineering, model evaluation, cross-validation
- **Model Used**: Random Forest Classifier (Best performing model with 96% accuracy)

---

## 📁 Dataset Details

- **Source**: Provided in `.csv` format
- **Target Variable**: `Satisfaction` (Satisfied vs. Neutral or Dissatisfied)
- **Features**: Includes age, gender, class, type of travel, flight distance, arrival delay, and multiple in-flight service ratings.

---

## 🔧 Steps Performed

1. **Data Cleaning**:
   - Handled missing values and converted `Flight Distance` to numeric
   - Capped outliers using IQR method
   - Dropped `Departure Delay` due to high correlation with `Arrival Delay`

2. **Exploratory Data Analysis**:
   - Visualized satisfaction by age, class, flight distance, etc.
   - Identified top drivers of satisfaction and dissatisfaction

3. **Feature Engineering**:
   - One-hot and label encoding for categorical variables
   - Scaled numerical features using MinMaxScaler

4. **Model Building**:
   - Compared 8 ML classifiers
   - Chose **Random Forest Classifier** for its high performance

5. **Model Evaluation**:
   - Accuracy: **96%**
   - AUC Score: **0.99**
   - Performed 5-fold cross-validation

---

## 💡 Key Insights

- Most influential features:
  - Online Boarding
  - In-flight Wifi Service
  - Type of Travel
  - In-flight Entertainment
- Satisfied passengers often:
  - Experience shorter delays
  - Travel in Business Class
  - Rate service features 4 or 5

---

## 🚀 Deployment

- **Platform**: Render
- **Connected with**: GitHub (Auto Deployment Enabled)
- **App Type**: Flask
- **Website**: [Airline Passenger Satisfaction Prediction](https://predicting-airline-passenger.onrender.com/)
---

## 🛠️ Tech Stack

- Python (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)
- Jupyter Notebook
- Flask / Streamlit (for deployment)
- Git & GitHub
- Render (Live Hosting)

---

## Contact
For any queries, contact me sai.subbu.in@gmail.com

---

## 📌 Future Improvements

- Add a Streamlit frontend for interactive predictions
- Schedule model retraining with updated data
- Add CI/CD pipeline using GitHub Actions

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).





