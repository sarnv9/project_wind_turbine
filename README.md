Wind Turbine Predictive Maintenance: End-to-End ML Pipeline

📌 The Business Problem

A wind company in a wind farm looks to optimize their operational costs to ensure the continuity of the service. Currently, the maintenance of the company is inefficient, as they fix the turbines in either a reactive or preventive way, which is more costly. Specifically, the machine learning pipeline is built to predict the turbine's Operational_Status (0 = Normal, 1 = Warning, 2 = Failure) so that critical failures are flagged before they happen. The objective of this project is to shift from reactive to predictive maintenance. By accurately predicting critical failures before they occur, maintenance teams can schedule targeted interventions, thereby optimizing resource allocation and maximizing energy output ROI.



🏗️ Architecture & Tech Stack
This project is structured as a complete end-to-end data pipeline, moving from raw sensor data exploration to a deployable predictive model.
* **Data Processing & EDA:** `pandas`, `seaborn`, `matplotlib`
* **Machine Learning:** `scikit-learn`, `imbalanced-learn` (SMOTE)
* **Model Serialization:** `joblib`
* **Deployment:** `FastAPI`

📊 Key Analytical Insights (EDA)
During the exploratory data analysis phase (eda_exploration.py): what I found and what I did about it:

* **Dataset Overview:** 
  * **Rows:** 3,000 turbines.
  * **Features:** 20 numeric sensor readings (generator, rotor, hydraulics, gearbox, ambient, tower) + 2 categorical (`Installation_Region`, `Turbine_Model`).
  * **Target:** `Operational_Status`, imbalanced at roughly 50% / 30% / 20%.
  * **Missing values:** None.

A duplicated sensor column. Rotor_Shaft_Vibration_Redundant is an exact copy of Rotor_Shaft_Vibration. It was dropped.

When it comes to outliers, the outlier's features represent a low percentage between 0.3% and 1.68%, representing a too low proportion for 3000 instances. Maintaining the outliers was the decision undr the assumption that the models are going to be robust.

The strongest correlation between sensors is Grid_Frequency_Hz with Amb_Ext_Temp (about -0.61).

The bivariant analysis revealed that specific hardware configurations (e.g., Turbine Model C) exhibit a significantly higher proportion of normal operational states compared to others, indicating strong predictive weight. In contrast, the hardward configuration Turbine_Model_D has a more balanced distribution.

The target variable (Operational_Status) is highly imbalanced, with normal operations representing ~50% of the dataset, while critical failures account for only ~20%. The main objective is to maximize the failure detections allowing a moderate rate of false positives by prioritizing the use of Recall as Class 2. This necessitated the use of SMOTE during model training to prevent data lekeage.

🤖 Machine Learning Pipeline
The modeling phase (predictive_maintenance.py) evaluates multiple baseline algorithms (Logistic Regression, Decision Trees, KNN, Naive Bayes) before scaling to robust ensemble methods.

* **Preprocessing:** Pipeline integration of StandardScaler and OneHotEncoder.

* **Imbalance Handling:** Implementation of SMOTE within the cross-validation pipeline to prevent data leakage.

* **Hyperparameter Tuning:** GridSearchCV utilized to optimize the RandomForestClassifier and GradientBoostingClassifier.

* **Primary Metric:** The models are evaluated primarily on Recall for Class 2 (Failure). In an industrial context, the financial penalty of missing an impending failure (False Negative) drastically outweighs the cost of an unnecessary inspection (False Positive).

* **Advanced Ensembles:** Implementation of a StackingClassifier and a BalancedRandomForestClassifier to push the predictive boundaries.

* **Business Optimization (Threshold Moving)**: The final decision threshold was manually adjusted based on the Precision-Recall curve to guarantee an 80% minimum Recall for critical failures, prioritizing operational safety over default statistical accuracy.


🚀 How to Run the Project
1. Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/Project_Wind_Turbine.git
cd Project_Wind_Turbine
pip install -r requirements.txt

```

2. Run the Exploratory Data Analysis:
This script will clean the dataset, generate analytical plots in the images/ directory, and output the processed dataset.


```bash
python eda_exploration.py

```

3. Train the Predictive Models:
This executes the ML pipeline, evaluates ensemble models, and outputs performance metrics and confusion matrices.

```bash
python predictive_maintenance.py
```

4. Launch the Production API:
Start the FastAPI server to serve predictions in real-time.
```bash
uvicorn api:app --reload
```

📈 Results
Metrics on the held-out test set (600 turbines):

Model	                          Recall (Failure - Class 2)	
Logistic Regression (Base)	                  58.0%	
Decision Tree (Base)	                      44.0%	
KNN (Base)	                                  55.0%	
Naive Bayes (Base)	                          60.0%	
Random Forest (GridSearch)	                  58.0%	
Gradient Boosting (Custom Threshold 0.14)	  80.0%	

Best model for detecting failures: Gradient Boosting (optimized via Threshold Moving) with a Failure recall of 80.0% and a macro-F1 of 0.73.

By adjusting the decision threshold from 0.50 to 0.14, the model successfully captures 80% of critical failures, aligning perfectly with the business objective of minimizing costly catastrophic breakdowns at the acceptable expense of minor false positive inspections.

📁 Repository Structure

├── data/                      # Raw and processed datasets
├── images/                    # Analytical plots and confusion matrices
├── models/                    # Serialized .pkl models for production
├── api.py                     # FastAPI application for model deployment
├── eda_exploration.py         # Exploratory analysis and data cleaning
├── predictive_maintenance.py  # Modeling, evaluation, and threshold optimization
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation