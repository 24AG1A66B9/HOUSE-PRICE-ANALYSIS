import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="House Price Analysis & Prediction",
    page_icon="🏠",
    layout="wide"
)


# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------

st.title("🏠 House Price Analysis & Prediction")
st.write(
    "Analyze housing data, understand property features, "
    "and predict house prices using Machine Learning."
)


# ---------------------------------------------------------
# DEMO DATASET
# ---------------------------------------------------------

@st.cache_data
def create_demo_data(n=1000):

    np.random.seed(42)

    area = np.random.randint(500, 4500, n)
    bedrooms = np.random.randint(1, 7, n)
    bathrooms = np.random.randint(1, 5, n)
    stories = np.random.randint(1, 4, n)
    parking = np.random.randint(0, 4, n)

    location = np.random.choice(
        ["Urban", "Suburban", "Rural"],
        n,
        p=[0.45, 0.35, 0.20]
    )

    furnishing = np.random.choice(
        ["Furnished", "Semi-Furnished", "Unfurnished"],
        n
    )

    property_age = np.random.randint(0, 40, n)

    location_effect = np.where(
        location == "Urban", 1500000,
        np.where(location == "Suburban", 800000, 0)
    )

    furnishing_effect = np.where(
        furnishing == "Furnished", 500000,
        np.where(furnishing == "Semi-Furnished", 250000, 0)
    )

    price = (
        area * 4500
        + bedrooms * 350000
        + bathrooms * 250000
        + stories * 150000
        + parking * 200000
        - property_age * 50000
        + location_effect
        + furnishing_effect
        + np.random.normal(0, 500000, n)
    )

    price = np.maximum(price, 500000)

    df = pd.DataFrame({
        "Area": area,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Stories": stories,
        "Parking": parking,
        "Location": location,
        "Furnishing": furnishing,
        "PropertyAge": property_age,
        "Price": price.astype(int)
    })

    return df


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.header("⚙️ Data Options")

uploaded_file = st.sidebar.file_uploader(
    "Upload Housing Dataset",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:

    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.sidebar.success("Dataset uploaded successfully!")

    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

else:

    df = create_demo_data()

    st.sidebar.info(
        "No dataset uploaded. A demo housing dataset is being used."
    )


# ---------------------------------------------------------
# STANDARDIZE COLUMN NAMES
# ---------------------------------------------------------

df.columns = (
    df.columns
    .str.strip()
    .str.replace(" ", "_")
    .str.replace("-", "_")
)


# ---------------------------------------------------------
# REMOVE DUPLICATES
# ---------------------------------------------------------

duplicate_count = df.duplicated().sum()

df = df.drop_duplicates().reset_index(drop=True)


# ---------------------------------------------------------
# FIND TARGET COLUMN
# ---------------------------------------------------------

possible_targets = [
    "Price",
    "price",
    "SalePrice",
    "saleprice",
    "House_Price",
    "house_price",
    "Selling_Price",
    "selling_price"
]

target_candidates = [
    col for col in possible_targets
    if col in df.columns
]

if target_candidates:

    target_col = target_candidates[0]

else:

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    if len(numeric_columns) == 0:

        st.error(
            "No numeric target column found. "
            "Please upload a dataset containing a house price column."
        )

        st.stop()

    target_col = st.sidebar.selectbox(
        "Select House Price Column",
        numeric_columns
    )


# ---------------------------------------------------------
# CONVERT TARGET TO NUMERIC
# ---------------------------------------------------------

df[target_col] = pd.to_numeric(
    df[target_col],
    errors="coerce"
)

df = df.dropna(
    subset=[target_col]
).reset_index(drop=True)


# ---------------------------------------------------------
# SIDEBAR DATA INFORMATION
# ---------------------------------------------------------

st.sidebar.write("### Dataset Information")

st.sidebar.write(
    f"Rows: **{df.shape[0]}**"
)

st.sidebar.write(
    f"Columns: **{df.shape[1]}**"
)

st.sidebar.write(
    f"Target: **{target_col}**"
)


# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview",
    "🧹 Data Cleaning",
    "📈 EDA",
    "🤖 ML Prediction",
    "💡 Insights",
    "⬇️ Downloads"
])


# =========================================================
# TAB 1 - OVERVIEW
# =========================================================

with tab1:

    st.header("📊 Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Houses",
            len(df)
        )

    with col2:
        st.metric(
            "Features",
            df.shape[1] - 1
        )

    with col3:
        st.metric(
            "Missing Values",
            int(df.isnull().sum().sum())
        )

    with col4:
        st.metric(
            "Duplicate Rows",
            duplicate_count
        )

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(20),
        use_container_width=True
    )

    st.subheader("Data Types")

    dtype_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Missing Values": df.isnull().sum().values
    })

    st.dataframe(
        dtype_df,
        use_container_width=True
    )

    st.subheader("Statistical Summary")

    st.dataframe(
        df.describe(include="all").transpose(),
        use_container_width=True
    )


# =========================================================
# TAB 2 - DATA CLEANING
# =========================================================

with tab2:

    st.header("🧹 Data Cleaning & Preparation")

    clean_df = df.copy()

    st.subheader("Missing Values Before Cleaning")

    missing_before = clean_df.isnull().sum()

    missing_table = pd.DataFrame({
        "Column": missing_before.index,
        "Missing Values": missing_before.values
    })

    missing_table = missing_table[
        missing_table["Missing Values"] > 0
    ]

    if len(missing_table) > 0:

        st.dataframe(
            missing_table,
            use_container_width=True
        )

    else:

        st.success("No missing values found.")

    # -----------------------------------------------------
    # MISSING VALUE HANDLING
    # -----------------------------------------------------

    numeric_cols = clean_df.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_cols = clean_df.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    for col in numeric_cols:

        if clean_df[col].isnull().any():

            clean_df[col] = clean_df[col].fillna(
                clean_df[col].median()
            )

    for col in categorical_cols:

        if clean_df[col].isnull().any():

            mode_value = clean_df[col].mode()

            if len(mode_value) > 0:

                clean_df[col] = clean_df[col].fillna(
                    mode_value[0]
                )

            else:

                clean_df[col] = clean_df[col].fillna(
                    "Unknown"
                )

    # -----------------------------------------------------
    # OUTLIER HANDLING
    # -----------------------------------------------------

    st.subheader("Outlier Handling")

    outlier_counts = {}

    for col in numeric_cols:

        if col == target_col:
            continue

        Q1 = clean_df[col].quantile(0.25)
        Q3 = clean_df[col].quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        count = (
            (clean_df[col] < lower)
            | (clean_df[col] > upper)
        ).sum()

        outlier_counts[col] = int(count)

        clean_df[col] = clean_df[col].clip(
            lower,
            upper
        )

    outlier_table = pd.DataFrame({
        "Feature": outlier_counts.keys(),
        "Outliers Handled": outlier_counts.values()
    })

    st.dataframe(
        outlier_table,
        use_container_width=True
    )

    st.success(
        "Missing values were filled and numerical feature "
        "outliers were handled using the IQR method."
    )

    st.subheader("Cleaned Dataset")

    st.dataframe(
        clean_df.head(20),
        use_container_width=True
    )


# =========================================================
# TAB 3 - EDA
# =========================================================

with tab3:

    st.header("📈 Exploratory Data Analysis")

    eda_df = df.copy()

    # -----------------------------------------------------
    # PRICE DISTRIBUTION
    # -----------------------------------------------------

    st.subheader("House Price Distribution")

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    sns.histplot(
        eda_df[target_col].dropna(),
        kde=True,
        ax=ax
    )

    ax.set_xlabel("House Price")
    ax.set_ylabel("Number of Houses")

    st.pyplot(fig)

    # -----------------------------------------------------
    # NUMERIC FEATURES
    # -----------------------------------------------------

    numeric_features = eda_df.select_dtypes(
        include=np.number
    ).columns.tolist()

    numeric_features = [
        col for col in numeric_features
        if col != target_col
    ]

    if len(numeric_features) > 0:

        selected_feature = st.selectbox(
            "Select a feature to compare with house price",
            numeric_features
        )

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        sns.scatterplot(
            data=eda_df,
            x=selected_feature,
            y=target_col,
            ax=ax
        )

        ax.set_title(
            f"{selected_feature} vs {target_col}"
        )

        st.pyplot(fig)

    # -----------------------------------------------------
    # CATEGORICAL ANALYSIS
    # -----------------------------------------------------

    categorical_features = eda_df.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    if len(categorical_features) > 0:

        selected_category = st.selectbox(
            "Select categorical feature",
            categorical_features
        )

        category_summary = (
            eda_df.groupby(selected_category)[target_col]
            .mean()
            .sort_values(ascending=False)
        )

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        category_summary.plot(
            kind="bar",
            ax=ax
        )

        ax.set_ylabel("Average House Price")
        ax.set_xlabel(selected_category)

        plt.xticks(rotation=30)

        st.pyplot(fig)

    # -----------------------------------------------------
    # CORRELATION
    # -----------------------------------------------------

    st.subheader("Correlation Heatmap")

    correlation_df = eda_df.select_dtypes(
        include=np.number
    )

    if correlation_df.shape[1] >= 2:

        fig, ax = plt.subplots(
            figsize=(12, 7)
        )

        sns.heatmap(
            correlation_df.corr(),
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            ax=ax
        )

        st.pyplot(fig)


# =========================================================
# TAB 4 - MACHINE LEARNING
# =========================================================

with tab4:

    st.header("🤖 House Price Prediction")

    ml_df = df.copy()

    # -----------------------------------------------------
    # REMOVE TARGET
    # -----------------------------------------------------

    X = ml_df.drop(
        columns=[target_col]
    )

    y = ml_df[target_col]

    # -----------------------------------------------------
    # REMOVE VERY HIGH MISSING COLUMNS
    # -----------------------------------------------------

    missing_ratio = X.isnull().mean()

    columns_to_keep = missing_ratio[
        missing_ratio < 0.50
    ].index

    X = X[columns_to_keep]

    # -----------------------------------------------------
    # IDENTIFY COLUMN TYPES
    # -----------------------------------------------------

    numeric_features = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    # -----------------------------------------------------
    # PREPROCESSING
    # -----------------------------------------------------

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ])

    preprocessor = ColumnTransformer([
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ])

    # -----------------------------------------------------
    # TRAIN TEST SPLIT
    # -----------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    # -----------------------------------------------------
    # FEATURE SELECTION
    # -----------------------------------------------------

    st.subheader("Feature Selection")

    # First fit preprocessor to find number of features

    temp_preprocessor = preprocessor.fit(
        X_train,
        y_train
    )

    X_train_processed = temp_preprocessor.transform(
        X_train
    )

    total_features = X_train_processed.shape[1]

    k_features = min(
        15,
        total_features
    )

    st.write(
        f"Selecting the top **{k_features}** "
        f"meaningful features from **{total_features}** "
        f"processed features."
    )

    # -----------------------------------------------------
    # MODELS
    # -----------------------------------------------------

    models = {

        "Linear Regression": LinearRegression(),

        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )
    }

    results = []

    trained_models = {}

    # -----------------------------------------------------
    # TRAIN MODELS
    # -----------------------------------------------------

    for model_name, model in models.items():

        pipeline = Pipeline([
            (
                "preprocessor",
                preprocessor
            ),
            (
                "feature_selection",
                SelectKBest(
                    score_func=f_regression,
                    k=k_features
                )
            ),
            (
                "model",
                model
            )
        ])

        pipeline.fit(
            X_train,
            y_train
        )

        predictions = pipeline.predict(
            X_test
        )

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        mse = mean_squared_error(
            y_test,
            predictions
        )

        rmse = np.sqrt(mse)

        r2 = r2_score(
            y_test,
            predictions
        )

        results.append({
            "Model": model_name,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R²": r2
        })

        trained_models[model_name] = pipeline

    results_df = pd.DataFrame(results)

    # -----------------------------------------------------
    # MODEL COMPARISON
    # -----------------------------------------------------

    st.subheader("📊 Model Comparison")

    st.dataframe(
        results_df.style.format({
            "MAE": "{:,.2f}",
            "MSE": "{:,.2f}",
            "RMSE": "{:,.2f}",
            "R²": "{:.4f}"
        }),
        use_container_width=True
    )

    # -----------------------------------------------------
    # BEST MODEL
    # -----------------------------------------------------

    best_model_name = results_df.loc[
        results_df["R²"].idxmax(),
        "Model"
    ]

    best_model = trained_models[
        best_model_name
    ]

    st.success(
        f"🏆 Best Model: **{best_model_name}**"
    )

    # -----------------------------------------------------
    # R2 COMPARISON CHART
    # -----------------------------------------------------

    st.subheader("R² Score Comparison")

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    sns.barplot(
        data=results_df,
        x="Model",
        y="R²",
        ax=ax
    )

    ax.set_ylim(
        min(0, results_df["R²"].min() - 0.1),
        min(1.0, results_df["R²"].max() + 0.1)
    )

    st.pyplot(fig)

    # -----------------------------------------------------
    # ACTUAL VS PREDICTED
    # -----------------------------------------------------

    st.subheader("Actual vs Predicted Prices")

    best_predictions = best_model.predict(
        X_test
    )

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    sns.scatterplot(
        x=y_test,
        y=best_predictions,
        ax=ax
    )

    min_value = min(
        y_test.min(),
        best_predictions.min()
    )

    max_value = max(
        y_test.max(),
        best_predictions.max()
    )

    ax.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--"
    )

    ax.set_xlabel("Actual Price")
    ax.set_ylabel("Predicted Price")

    st.pyplot(fig)

    # -----------------------------------------------------
    # FEATURE IMPORTANCE
    # -----------------------------------------------------

    st.subheader("⭐ Important Features")

    if best_model_name in [
        "Random Forest",
        "Gradient Boosting"
    ]:

        model_object = best_model.named_steps[
            "model"
        ]

        feature_selector = best_model.named_steps[
            "feature_selection"
        ]

        fitted_preprocessor = best_model.named_steps[
            "preprocessor"
        ]

        try:

            feature_names = (
                fitted_preprocessor
                .get_feature_names_out()
            )

            selected_mask = feature_selector.get_support()

            selected_feature_names = (
                feature_names[selected_mask]
            )

            importance_values = (
                model_object.feature_importances_
            )

            importance_df = pd.DataFrame({
                "Feature": selected_feature_names,
                "Importance": importance_values
            }).sort_values(
                "Importance",
                ascending=False
            )

            st.dataframe(
                importance_df,
                use_container_width=True
            )

            fig, ax = plt.subplots(
                figsize=(10, 6)
            )

            top_features = importance_df.head(10)

            sns.barplot(
                data=top_features,
                x="Importance",
                y="Feature",
                ax=ax
            )

            st.pyplot(fig)

        except Exception as e:

            st.warning(
                f"Feature importance could not be displayed: {e}"
            )

    else:

        st.info(
            "Feature importance visualization is shown "
            "for tree-based models."
        )


# =========================================================
# TAB 5 - INSIGHTS
# =========================================================

with tab5:

    st.header("💡 Business & Analytical Insights")

    st.write(
        "The following observations are generated from the "
        "dataset and model results."
    )

    # -----------------------------------------------------
    # TARGET STATISTICS
    # -----------------------------------------------------

    average_price = df[target_col].mean()
    minimum_price = df[target_col].min()
    maximum_price = df[target_col].max()

    st.subheader("🏠 Price Overview")

    st.write(
        f"- Average house price: **{average_price:,.2f}**"
    )

    st.write(
        f"- Minimum house price: **{minimum_price:,.2f}**"
    )

    st.write(
        f"- Maximum house price: **{maximum_price:,.2f}**"
    )

    # -----------------------------------------------------
    # CORRELATION INSIGHT
    # -----------------------------------------------------

    numeric_df = df.select_dtypes(
        include=np.number
    )

    if target_col in numeric_df.columns:

        correlations = (
            numeric_df.corr()[target_col]
            .drop(target_col)
            .sort_values(
                key=abs,
                ascending=False
            )
        )

        if len(correlations) > 0:

            strongest_feature = correlations.index[0]

            strongest_value = correlations.iloc[0]

            st.subheader("📌 Strongest Numeric Relationship")

            st.write(
                f"**{strongest_feature}** has the strongest "
                f"relationship with house price with a "
                f"correlation of **{strongest_value:.2f}**."
            )

    # -----------------------------------------------------
    # MODEL INSIGHT
    # -----------------------------------------------------

    st.subheader("🤖 Machine Learning Observation")

    best_r2 = results_df.loc[
        results_df["R²"].idxmax(),
        "R²"
    ]

    best_rmse = results_df.loc[
        results_df["R²"].idxmax(),
        "RMSE"
    ]

    st.write(
        f"The best performing model is "
        f"**{best_model_name}**, with an R² score of "
        f"**{best_r2:.4f}**."
    )

    st.write(
        f"Its RMSE is approximately "
        f"**{best_rmse:,.2f}**."
    )

    st.write(
        "Higher R² indicates better explanatory power, "
        "while lower MAE and RMSE indicate smaller "
        "prediction errors."
    )

    # -----------------------------------------------------
    # FEATURE IMPORTANCE INSIGHT
    # -----------------------------------------------------

    if (
        best_model_name in
        ["Random Forest", "Gradient Boosting"]
    ):

        try:

            model_object = best_model.named_steps[
                "model"
            ]

            feature_selector = best_model.named_steps[
                "feature_selection"
            ]

            fitted_preprocessor = best_model.named_steps[
                "preprocessor"
            ]

            feature_names = (
                fitted_preprocessor
                .get_feature_names_out()
            )

            selected_mask = (
                feature_selector.get_support()
            )

            selected_names = feature_names[
                selected_mask
            ]

            importance_values = (
                model_object.feature_importances_
            )

            top_index = np.argmax(
                importance_values
            )

            top_feature = selected_names[
                top_index
            ]

            st.subheader(
                "⭐ Most Influential Model Feature"
            )

            st.write(
                f"The model identifies **{top_feature}** "
                "as the most influential selected feature "
                "for predicting house prices."
            )

        except Exception:
            pass


# =========================================================
# TAB 6 - DOWNLOADS
# =========================================================

with tab6:

    st.header("⬇️ Download Results")

    # Clean dataset

    cleaned_csv = df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download Dataset CSV",
        data=cleaned_csv,
        file_name="housing_dataset.csv",
        mime="text/csv"
    )

    # Model results

    if "results_df" in locals():

        results_csv = results_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="📊 Download Model Comparison",
            data=results_csv,
            file_name="house_price_model_comparison.csv",
            mime="text/csv"
        )

    st.success(
        "House Price Analysis & Prediction project completed successfully! 🏠🤖"
    )