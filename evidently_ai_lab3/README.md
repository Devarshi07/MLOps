# 📊 Evidently AI – Data Drift Analysis on UCI Student Performance Dataset

This project demonstrates the use of **Evidently AI** to detect and visualize **data drift** between two subsets of the **UCI Student Performance** dataset.  
Data drift analysis helps monitor the stability of data distributions over time, ensuring the reliability of machine learning models in production.

---

## 🧠 Overview

The experiment evaluates dataset drift between two groups of students based on **gender** (Sex=1 vs. Sex=2).  
The **reference dataset** corresponds to one gender group (Sex=1) and the **production dataset** to the other (Sex=2).  
Evidently AI computes drift metrics for each feature and provides interactive visualizations to identify which attributes have changed significantly.

---

## 📦 Tools and Libraries Used

- 🧮 **Python3**
- 📈 **Evidently AI (v0.7.0)**
- 🐼 **Pandas**, **NumPy**
- 🔢 **scikit-learn**
- 🎨 **Matplotlib**, **Seaborn**
- 📚 **ucimlrepo** (for dataset retrieval)

---

## 📂 Dataset

The **Student Performance** dataset (UCI ID: 856) contains academic and behavioral data for 145 students with 32 features, including:

| Feature | Type | Description |
|----------|------|-------------|
| **Student Age** | Numerical | Encoded age of the student. |
| **Sex** | Categorical | Gender of the student (1 = Male, 2 = Female). |
| **Graduated high-school type** | Categorical | Type of high school attended before university. |
| **Scholarship type** | Categorical | Type of financial aid received, if any. |
| **Additional work** | Categorical | Whether the student is working part-time (1 = No, 2 = Yes). |
| **Regular artistic or sports activity** | Categorical | Participation in extracurricular activities. |
| **Do you have a partner** | Categorical | Relationship status (1 = No, 2 = Yes). |
| **Total salary if available** | Numerical | Reported total monthly salary. |
| **Transportation to university** | Categorical | Mode of commute (e.g., walking, car, public transport). |
| **Preparation to midterm exams 1 & 2** | Categorical | Self-reported preparation effort before exams. |
| **Taking notes in classes** | Numerical | Frequency of note-taking during lectures. |
| **Listening in classes** | Numerical | Level of attentiveness in lectures. |
| **Flip-classroom participation** | Categorical | Whether student engages in flipped classroom sessions. |
| **Cumulative GPA (last semester)** | Numerical | Last semester’s GPA (scaled 1–4). |
| **Expected GPA (graduation)** | Numerical | Anticipated GPA at graduation. |

For this analysis:
- `ref_data` = subset where `Sex == 1`
- `prod_data` = subset where `Sex == 2`


The **Student Performance Dataset** captures demographic, academic, and behavioral data of 145 students studying in Cyprus.  
Each record represents one student, with attributes covering family background, study habits, and academic expectations.


---

## ⚙️ Steps and Methodology

1. **Dataset Loading:**  
   The dataset is fetched using `ucimlrepo.fetch_ucirepo(id=856)` and combined into a single DataFrame.

2. **Schema Definition:**  
   Columns are divided into numerical and categorical types using `DataDefinition()` from Evidently.

3. **Data Splitting:**  
   The data is split by the `Sex` column to form reference and production datasets.

4. **Drift Detection:**  
   Evidently’s `DataDriftPreset()` is used within a `Report()` to detect column-level drift.

5. **Visualization:**  
   - Evidently HTML Dashboard summarizes drift metrics.
   - Matplotlib histograms and box plots visualize the distribution shifts.

---

## 📊 Results

- **Total Columns:** 22  
- **Drifted Columns:** 17  
- **Drift Detected in:** 77.27% of features  
- **Threshold:** 0.5 (dataset drift is detected if share > 0.5)

**Top Drifted Features:**
- Graduated high-school type  
- Father’s education  
- Additional work  
- Preparation to midterm exams  
- Regular artistic or sports activity

**Example Visualizations:**
- Feature distributions for “Student Age”, “Total salary if available”, and “Taking notes in classes”.
- Boxplots showing clear differences between reference and production subsets.

---

## 📈 Example Code

```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=ref_data, current_data=prod_data)
report.save_html("data_drift_report.html")
```

## 💡 Key Insights
---

## 🔍 Detailed Results and Insights

### 🧮 Statistical Summary
- The Evidently AI report analyzed 22 selected columns and identified **data drift in 17 (77.27%)** of them.  
- The **dataset drift threshold** was set to **0.5**, and the computed drift share of **0.773** confirmed substantial distributional change.

### ⚖️ Feature-Level Drift Highlights
- **Graduated high-school type** and **Father’s education** showed strong categorical drift, suggesting possible demographic imbalance.
- **Additional work** and **Preparation to midterm exams 1 & 2** reflected behavioral or academic engagement differences.
- Numerical features such as **Student Age** and **Listening in classes** exhibited moderate drift, highlighting variation in engagement or maturity between genders.

### 📈 Visual Analysis
- **Histograms** illustrated distribution shifts in numerical features (e.g., “Student Age”, “Taking notes in classes”).
- **Boxplots** revealed differences in score spreads, with females tending to report higher consistency in class attentiveness and exam preparation metrics.


#### Evidently’s `DataDriftPreset()` applies statistical tests such as:
- **Chi-square test** for categorical variables.
- **Kolmogorov–Smirnov (KS)** or **Z-test** for numerical variables.  
These tests compare feature distributions to flag significant changes.


## 🚀 Future Scope

- Integrate Evidently into a CI/CD or MLOps pipeline for continuous data quality checks.
- Automate report generation and email alerts for drift threshold breaches.
- Extend to model performance drift and target drift analysis.

## 👤 Author

- Devarshi Mahajan
- M.S. in Data Analytics Engineering, Northeastern University
📧 [mahajan.dev@northeastern.edu](mailto:mahajan.dev@northeastern.edu) | 🌐 [GitHub Profile](https://github.com/devarshi07)

## 🧾 References

- [Evidently AI Documentation](https://docs.evidently.ai/)
- [UCI Machine Learning Repository – Student Performance Dataset](https://archive.ics.uci.edu/dataset/856/student+performance)
