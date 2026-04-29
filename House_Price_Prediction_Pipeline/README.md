# House Price Prediction: Advanced Regression Pipeline 🏠

**[🇬🇧 English](#english)** | **[🇬🇪 ქართული](#ქართული)**

---

## <a name="english"></a> 🇬🇧 English

### 📌 Project Overview
This project predicts house prices using advanced regression techniques (Lasso & XGBoost). Originally developed as an exploratory Jupyter Notebook, this project has been heavily refactored into a **Production-Ready Machine Learning Pipeline**.

### 🚀 Refactoring & Architecture Upgrades
To make this project portfolio-ready and aligned with software engineering best practices, the following major improvements were implemented:

1. **Monolithic Notebook to Modular OOP:**
   - *Before:* All code (EDA, cleaning, training) was in a single, massive `.ipynb` file.
   - *After:* The code is split into an Object-Oriented structure (`DataLoader`, `DataPreprocessor`) inside a `src/` package.

2. **Eliminated Hardcoded Variables:**
   - *Before:* File paths and outlier IDs were hardcoded deep in the scripts.
   - *After:* Implemented a `config.py` file to control paths, hyperparameters, and thresholds centrally.

3. **Reproducible Environment:**
   - *Before:* Local system dependencies.
   - *After:* Created a strict `requirements.txt` to ensure cross-platform reproducibility.

4. **Data Security & Git Best Practices:**
   - Utilized `.gitignore` to prevent raw data and local environments from being pushed to the repository.

### 🛠️ Tech Stack
- **Python 3.9+**
- **Pandas & NumPy** (Data Manipulation)
- **Scikit-Learn & XGBoost** (Modeling)
- **Optuna** (Hyperparameter Optimization)
- **PyYAML** (Configuration Management)
- **SHAP** (Model Interpretability - *Coming Soon*)

### 📊 Results
- **Lasso RMSE:** ~0.1105
- **XGBoost RMSE:** *To be finalized*
- **Ensemble (Lasso + XGBoost):** *To be finalized*

---

## <a name="ქართული"></a> 🇬🇪 ქართული

### 📌 პროექტის მიმოხილვა
ამ პროექტის მიზანია უძრავი ქონების ფასის პროგნოზირება მაღალრგანზომიანი რეგრესიის (Lasso & XGBoost) გამოყენებით. თავდაპირველად პროექტი შეიქმნა ექსპლორაციულ Jupyter Notebook-ში, თუმცა შემდგომში სრულად გადაკეთდა **პორტფოლიოსთვის მზა მანქანური სწავლების (ML) არქიტექტურად**.

### 🚀 პროექტის დამუშავება და არქიტექტურული ცვლილებები
საუკეთესო საპრაქტიკო პრინციპებთან შესაბამისობისთვის და პორტფოლიოსთვის მომზადებლად, განხორციელდა შემდეგი გაუმჯობესებები:

1. **Notebook-დან მოდულურ OOP არქიტექტურაზე გადასვლა:**
   - *მანამდე:* მთლიანი კოდი (მონაცემთა დამუშავება, გაწმენტება, მოდელის გაწვრთნა) ეწერა ერთ დიდ `.ipynb` ფაილში.
   - *ახლა:* კოდი დაყოფილია ობიექტზე ორიენტირებული პროგრამირების (OOP) პრინციპებით ცალკეულ მოდულებად (`DataLoader`, `DataPreprocessor`) `src/` ფოლდერში.

2. **Hardcoded ცვლადების ამოღება:**
   - *მანამდე:* ფაილების მისამართები (paths) და Outlier-ების ID-ები პირდაპირ კოდში იყო ჩაწერილი.
   - *ახლა:* შეიქმნა `config.py` ფაილი, სადაც ცენტრალიზებულად იმართება მისამართები, ჰიპერპარამეტრები და ზღვრები.

3. **გარემოს სტანდარტიზაცია:**
   - დაემატა `requirements.txt` ფაილი, რათა ნებისმიერ სხვა კომპიუტერზე პროექტის გაშვება მარტივად და უშეცდომოდ მოხდეს.

4. **მონაცემთა უსაფრთხოება (Git):**
   - `.gitignore` ფაილის მეშვეობით დაცულია, რომ ლოკალური გარემო (venv) და დიდი მონაცემთა ფაილები არ აიტვირთოს საჯარო სივრცეში.

### 🛠️ ტექნოლოგიური სტეკი
- **Python 3.9+**
- **Pandas & NumPy** (მონაცემთა მანიპულაცია)
- **Scikit-Learn & XGBoost** (მოდელირება)
- **Optuna** (ჰიპერპარამეტრების ოპტიმიზაცია)
- **PyYAML** (კონფიგურაციის მენეჯმენტი)
- **SHAP** (მოდელის ინტერპრეტაცია - *მალე*)

### 📊 შედეგები
- **Lasso RMSE:** ~0.1105
- **XGBoost RMSE:** *ფინალიზდება*
- **ანსამბლი (Lasso + XGBoost):** *ფინალიზდება*

---

### 🚀 How to Run
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Place `train.csv` and `test.csv` in `data/raw/`
4. Run preprocessing: `python src/data/preprocess.py`
5. Train models: `python src/models/train.py`
