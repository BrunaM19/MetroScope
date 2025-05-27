# 🚆 MetroScope: Train Delay Prediction System

## 📌 Project Overview

**MetroScope** is a data analysis and prediction project developed for **SNCF** (French National Railway Company) to analyze historical train delay data, uncover patterns, and build a predictive model to forecast train delays. The project also features an interactive dashboard built with **Streamlit** 🧮📊 to help users visualize data insights and get real-time delay predictions.

The main goal is to **improve the efficiency and transparency** of train travel, enabling travelers and operators to better plan their schedules. 🗓️🚉

---

## 🎯 Objectives

* 🧹 **Data Cleaning & Preprocessing:**
  Handle missing values, remove duplicates, convert data types, and engineer new features (like date components).

* 🔍 **Exploratory Data Analysis (EDA):**
  Visualize delay distributions, station-level statistics, and correlations to understand the factors affecting delays.

* 🤖 **Predictive Modeling:**
  Train and evaluate a machine learning model (Random Forest) to predict average arrival delays based on departure station, arrival station, month, and year.

* 💻 **Dashboard Development:**
  Build an interactive **Streamlit** dashboard to display insights and provide real-time delay predictions for user-selected routes.

---

## 🗂️ Project Structure

```
MetroScope/
│
├── dataset.csv               # 📥 Raw data file (input)
├── cleaned_dataset.csv       # 🧽 Cleaned data output after EDA
├── featured_dataset.csv      # 🧠 Dataset enriched with engineered features
├── tardis_eda.ipynb          # 📊 Jupyter notebook for data cleaning and exploratory analysis
├── tardis_model.ipynb        # 🤖 Jupyter notebook for training and evaluating the prediction model
├── tardis_dashboard.py       # 🌐 Streamlit dashboard script for interactive visualization and prediction
├── requirements.txt          # 📦 Python dependencies
└── README.md                 # 📘 This documentation file
```

---

## ⚙️ Installation

1. **📥 Clone the repository:**

```bash
git clone git@github.com:BrunaM19/MetroScope.git
cd MetroScope
```

2. **🐍 Create a virtual environment (recommended):**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **📦 Install required packages:**

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### 🧼 Step 1: Data Cleaning and Exploration

Open and run the notebook to clean the dataset, handle missing data, and explore the data visually:

```bash
jupyter notebook tardis_eda.ipynb
```

📄 This notebook outputs `cleaned_dataset.csv`.

### 🤖 Step 2: Train the Prediction Model

Use the cleaned dataset to train a **Random Forest** regression model for delay prediction:

```bash
jupyter notebook tardis_model.ipynb
```

📁 This notebook saves the trained model as `delay_predictor.pkl`.

### 🌐 Step 3: Launch the Interactive Dashboard

Run the **Streamlit** dashboard to visualize insights and predict delays interactively:

```bash
streamlit run tardis_dashboard.py
```

---

## 📚 Dependencies

* 🐼 pandas
* 🔢 numpy
* 📊 matplotlib
* 🎨 seaborn
* 🤖 scikit-learn
* 🌐 streamlit
* 💾 joblib

All dependencies and their versions are listed in `requirements.txt`.

---

## 📈 Model Details

* **🧠 Algorithm:** Random Forest Regressor

* **🛠️ Features Used:**

  * 🏁 Departure station (categorical)
  * 🎯 Arrival station (categorical)
  * 📅 Month (numerical)
  * 📆 Year (numerical)

* **🔧 Preprocessing:**

  * 🔢 One-hot encoding for categorical features
  * 🧮 Mean imputation for missing numerical values

* **📏 Evaluation Metrics:** RMSE and R² score

---

## 🌟 Future Improvements (Bonus Ideas)

* 🔍 Implement feature selection techniques to improve model performance
* 📡 Incorporate real-time SNCF open data for live updates
* 🗺️ Add geospatial visualizations and animated charts to the dashboard
* 🧬 Experiment with deep learning models for better accuracy
* 🔎 Add explainability features to interpret predictions

---

## 📜 License

This project is licensed under the **MIT License** 🪪

---

## 📬 Contact

Developed by the **SNCF Data Analysis Team**.
For questions or contributions, please contact the project maintainer. 💌
