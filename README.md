# User Profiling and Segmentation using Clustering

## Table of Contents
- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Objective](#objective)
- [Team Members](#team-members)
- [Dataset Overview](#dataset-overview)
- [Project Workflow](#project-workflow)
  - [Step 1: Define Objective](#step-1-define-objective)
  - [Step 2: Data Collection](#step-2-data-collection)
  - [Step 3: Data Preprocessing & Feature Engineering](#step-3-data-preprocessing--feature-engineering)
  - [Step 4: Determine Optimal Clusters](#step-4-determine-optimal-clusters)
  - [Step 5: Model Training](#step-5-model-training)
  - [Step 6: Cluster Profiling](#step-6-cluster-profiling)
  - [Step 7: Cluster Visualization](#step-7-cluster-visualization)
  - [Step 8: Business Strategy](#step-8-business-strategy)
- [Technologies Used](#technologies-used)
- [Deployment](#deployment)
- [Conclusion](#conclusion)

---

## Project Overview

This project focuses on segmenting users based on behavioral, demographic, and engagement features using unsupervised learning techniques. By identifying user groups, we can design more personalized and effective advertising strategies.

---

## Problem Statement

Digital platforms often deal with a diverse range of users. Treating all users the same way can lead to inefficient ad spending and reduced engagement. By clustering users with similar characteristics, businesses can target each segment more effectively.

---

## Objective

- Segment users into distinct clusters using machine learning.
- Visualize the distribution and characteristics of each cluster.
- Translate these clusters into actionable marketing strategies.
- Provide an interactive tool to demonstrate the segmentation process using Streamlit.

---

## Team Members

**PVB Adithya**  
- Email: vprata2@gitam.in / adithya.vprata@gmail.com  
- GitHub: [pvba-py](https://github.com/pvba-py)  
- LinkedIn: [pvba](https://www.linkedin.com/in/pvba)

**Jay Varma**  
- Email: jmandapa@gitam.in / jayasimhavarma0401@gmail.com  
- GitHub: [Jayvarma04](https://github.com/Jayvarma04)  
- LinkedIn: [jayvarma04](https://www.linkedin.com/in/jayvarma04)

**Surya Vamsi**  
- Email: suryavamsi.m1@gmail.com  
- GitHub: [surya272004](https://github.com/surya272004)  
- LinkedIn: [surya-vamsi-27o2004](https://www.linkedin.com/in/surya-vamsi-27o2004)

**Ramya K**  
- Email: sriramyakatrapalli@gmail.com  
- GitHub: [Ramyak005](https://github.com/Ramyak005)  
- LinkedIn: [ramya-katrapalli](https://www.linkedin.com/in/ramya-katrapalli-2b0132304/)

---

## Dataset Overview

- **Filename**: user_profiles_for_ads.csv  
- **Records**: 1,000 users  
- **Features**: 16

### Key Feature Categories:
- Demographics: Age, Gender, Income Level, Education Level
- Online Behavior: Device Usage, Time Spent Online (Weekday/Weekend), Top Interests
- Engagement Metrics: CTR, Conversion Rate, Ad Interaction Time
- Meta Data: Location, Language, User ID (excluded from modeling)

---

## Project Workflow

### Step 1: Define Objective
The goal is to identify meaningful user segments to improve ad targeting, boost engagement, and increase conversion rates.

### Step 2: Data Collection
The dataset `user_profiles_for_ads.csv` was loaded and inspected. It contained 16 structured features for 1,000 users.

### Step 3: Data Preprocessing & Feature Engineering
- Converted age groups into numerical midpoints.
- Dropped high-cardinality fields (e.g., User ID, raw text interests).
- One-hot encoded categorical features such as Gender, Income Level, Device, etc.
- Created:
  - `Time Ratio = Weekend Time / Weekday Time`
  - `Engagement Score = CTR × Conversion × Ad Interaction Time`

### Step 4: Determine Optimal Clusters
Used the Elbow Method to analyze inertia and chose **k = 4** as the ideal number of clusters.

### Step 5: Model Training
Applied K-Means clustering on the normalized feature space using `StandardScaler`.

### Step 6: Cluster Profiling
Each cluster was profiled using group-wise means and interpreted as:

- **Cluster 0**: Budget-conscious users, low income, moderate engagement
- **Cluster 1**: High-income, tech-savvy male users, highly engaged
- **Cluster 2**: Young, female, highly social mobile users
- **Cluster 3**: Affluent, urban, moderately engaged professionals

### Step 7: Cluster Visualization
Used PCA, t-SNE, and UMAP to visualize cluster separation:
- PCA: Linearly reduced data to 2D
- t-SNE: Preserved local neighborhood structure
- UMAP: Preserved both global and local structure

### Step 8: Business Strategy
Mapped cluster traits to business strategies such as:
- Mobile-first campaigns
- High-value subscription targeting
- Value deals for budget-conscious users

---

## Technologies Used

- Python (pandas, numpy, seaborn, matplotlib)
- Scikit-learn (KMeans, PCA, t-SNE, preprocessing)
- UMAP-learn
- Streamlit (for dashboard deployment)

---

## Deployment

The project includes a fully functional **Streamlit Web Application** where users can:
- Upload their own dataset
- Run clustering analysis
- Visualize cluster distribution
- Download clustered data

---

## Conclusion

This project applied unsupervised machine learning to segment users based on multi-dimensional behavioral, demographic, and engagement features. A complete pipeline was developed—from raw data preprocessing and feature engineering to model training, visualization, and interpretation. The use of K-Means clustering, supported by dimensionality reduction techniques (PCA, t-SNE, UMAP), revealed four distinct user personas, each tied to tailored business strategies.

The final deployment as an interactive Streamlit web app enables real-time data exploration and strategic decision-making. This solution can serve as a foundation for customer relationship management (CRM), personalized marketing, targeted ad delivery, and product recommendation systems. Overall, the project demonstrates the value of combining statistical rigor with domain-specific business insights for data-driven user understanding.
