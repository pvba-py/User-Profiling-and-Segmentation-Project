import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Set Streamlit page configuration
st.set_page_config(page_title="User Profiling & Segmentation", layout="wide")

# Title and Introduction
st.title("📊 User Profiling and Segmentation")
st.markdown("""
This web application segments users based on behavioral, demographic, and engagement data. 
Upload your dataset, visualize clusters, and generate actionable insights for targeted marketing.
""")

# Upload Section
st.header("Step 1: Upload Your CSV File")
uploaded_file = st.file_uploader("Upload your user dataset (CSV only)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Step 2: Data Preview")
    st.dataframe(df.head())
    st.write("Shape:", df.shape)
    st.write("Columns:", list(df.columns))

    # Preprocessing and Feature Engineering
    st.subheader("Step 3: Preprocessing & Feature Engineering")
    st.markdown("""
    - Converted age groups to numeric midpoints  
    - Created Time Ratio = Weekend / Weekday usage  
    - Calculated Engagement Score = CTR × Conversion × Ad Time  
    - Applied one-hot encoding for categorical features
    """)

    df_clean = df.copy()
    df_clean['Age'] = df_clean['Age'].replace({
        '18-24': 21, '25-34': 29.5, '35-44': 39.5,
        '45-54': 49.5, '55-64': 59.5, '65+': 70
    })

    df_clean['Time Ratio'] = df_clean['Time Spent Online (hrs/weekend)'] / (df_clean['Time Spent Online (hrs/weekday)'] + 0.01)
    df_clean['Engagement Score'] = df_clean['Click-Through Rates (CTR)'] * df_clean['Conversion Rates'] * df_clean['Ad Interaction Time (sec)']

    df_clean.drop(columns=['User ID'], inplace=True, errors='ignore')
    df_encoded = pd.get_dummies(df_clean)

    # Choose number of clusters
    st.subheader("Step 4: Select Number of Clusters")
    k = st.slider("Select number of clusters (k)", min_value=2, max_value=10, value=4)

    # Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_encoded)

    # KMeans Clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    df_encoded['Cluster'] = kmeans.fit_predict(X_scaled)

    # PCA for visualization
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)
    df_encoded['PCA1'] = pca_result[:, 0]
    df_encoded['PCA2'] = pca_result[:, 1]

    st.subheader("Step 5: Cluster Visualization (PCA)")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df_encoded, x='PCA1', y='PCA2', hue='Cluster', palette='Set1', ax=ax)
    st.pyplot(fig)

    # Cluster Summary
    st.subheader("Step 6: Cluster Summary Statistics")
    cluster_summary = df_encoded.groupby("Cluster").mean().round(2)
    st.dataframe(cluster_summary)

    # Business Recommendations with Cluster Descriptions
    st.subheader("Step 7: Cluster Summary and Interpretation (k = 4)")
    st.markdown("""
    **Cluster 0: Budget-Conscious, Moderate Engagement**  
    - Average Age: ~40.2  
    - Gender: ~53% Female  
    - Income Level: 100% in 0–20k bracket  
    - Device Usage: Mixed (Desktop, Mobile, Tablet)  
    - Engagement Score: Low (~0.556)  
    - Language: Higher English usage  
    - Location: Spread across rural, suburban, urban  
    - Education: Mostly High School and Technical  
    **Business Insight**: Target with budget-friendly ads, freemium plans, and discount-based promotions.  

    **Cluster 1: High-Income, Male-Dominated Users**  
    - Average Age: ~40.6  
    - Gender: 100% Male  
    - Income Level: Mostly in 60k–100k+ brackets  
    - Device Usage: Well distributed across all types  
    - Engagement Score: Highest (~0.600)  
    - Language: More Spanish and Mandarin  
    - Education: Strong in Bachelor's and Master's  
    **Business Insight**: Ideal segment for premium products, tech gadgets, high-end subscriptions.  

    **Cluster 2: Young, Female, Highly Active Users**  
    - Average Age: ~40.0  
    - Gender: 100% Female  
    - Likes & Reactions: Highest among all clusters  
    - Income Level: Mostly 40k–80k  
    - Device Usage: High mobile-only usage  
    - Engagement Score: Strong (~0.585)  
    **Business Insight**: Target with fashion, fitness, and lifestyle content. Use mobile-first design and influencer partnerships.  

    **Cluster 3: Affluent Urban Professionals**  
    - Average Age: ~39.9  
    - Gender: ~54% Female  
    - Income Level: 100% in 80k–100k bracket  
    - Device Usage: Slightly higher desktop & tablet  
    - Engagement Score: Moderate (~0.530)  
    - Location: 38% Rural, remainder urban/suburban  
    - Education: Mixed, but balanced  
    **Business Insight**: Target with luxury and curated product bundles. Consider personalized marketing strategies.  
    """)

    # Download Button
    st.subheader("Step 8: Download Clustered Data")
    st.download_button(
        label="Download CSV with Clusters",
        data=df_encoded.to_csv(index=False),
        file_name="user_clusters.csv",
        mime="text/csv"
    )

    # Footer
        # Footer
    st.markdown("""
    ---
    ### Developed by:

    **Team Members:**

    - **PVB Adithya**  
       vprata2@gitam.in / adithya.vprata@gmail.com  
       GitHub: [pvba-py](https://github.com/pvba-py)  
       LinkedIn: [pvba](https://www.linkedin.com/in/pvba/)

    - **Jay Varma**  
       jmandapa@gitam.in / jayasimhavarma0401@gmail.com  
       GitHub: [Jayvarma04](https://github.com/Jayvarma04)  
       LinkedIn: [jayvarma04](https://www.linkedin.com/in/jayvarma04)

    - **Surya Vamsi**  
       suryavamsi.m1@gmail.com  
       GitHub: [surya272004](https://github.com/surya272004)  
       LinkedIn: [surya-vamsi-27o2004](https://www.linkedin.com/in/surya-vamsi-27o2004)

    - **Ramya K**  
       sriramyakatrapalli@gmail.com  
       GitHub: [Ramyak005](https://github.com/Ramyak005)  
       LinkedIn: [ramya-katrapalli](https://www.linkedin.com/in/ramya-katrapalli-2b0132304/)
    """)


else:
    st.info("Please upload a CSV file to begin.")
