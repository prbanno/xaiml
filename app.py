# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression, LogisticRegression
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import r2_score, mean_squared_error
# from sklearn.datasets import fetch_california_housing
# import shap
#
# st.set_page_config(page_title="XAI for Regression", page_icon="🔍", layout="wide")
#
# st.title("🔍 XAI for Regression – Model Comparison Tool")
# st.markdown("""
# Compare regression algorithms and visualize **Explainable AI (XAI)** insights.
# You can use the built-in **California Housing dataset** or upload your own.
# """)
#
#
# # --- Dataset Selection ---
# dataset_choice = st.radio(
#     "Select dataset source:",
#     ["Use sample (California Housing)", "Upload your own CSV"],
#     horizontal=True
# )
#
# if dataset_choice == "Use sample (California Housing)":
#     data = fetch_california_housing(as_frame=True)
#     df = data.frame
#     df["target"] = data.target
#     st.success("Using California Housing dataset.")
#     st.write("### Dataset Preview", df.head())
#
# else:
#     uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])
#     if uploaded_file:
#         df = pd.read_csv(uploaded_file)
#         st.write("### Dataset Preview", df.head())
#     else:
#         st.warning("Please upload a CSV file to proceed.")
#         st.stop()
#
# # --- Target Column Selection ---
# target_col = st.selectbox("Select Target Column (Y)", df.columns, index=len(df.columns) - 1)
#
# X = df.drop(columns=[target_col])
# y = df[target_col]
#
# # --- Model Selection ---
# model_choice = st.selectbox(
#     "Select Regression Model",
#     ["Linear Regression", "Random Forest", "Logistic Regression (for binary target)", "XAI Regression (Explainable Linear Model)", "Compare All Models"]
# )
#
# # --- Train-Test Split ---
# test_size = st.slider("Test Size (fraction for testing)", 0.1, 0.5, 0.2)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
#
# # --- Model Definitions ---
# models = {
#     "Linear Regression": LinearRegression(),
#     "Random Forest": RandomForestRegressor(),
#     "Logistic Regression (for binary target)": LogisticRegression(max_iter=1000),
#     "XAI Regression (Explainable Linear Model)": LinearRegression()
# }
#
# # --- Single Model Mode ---
# if model_choice != "Compare All Models":
#     model = models[model_choice]
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#
#     # --- Evaluation ---
#     st.subheader("Model Evaluation")
#     col1, col2 = st.columns(2)
#     col1.metric("R² Score", f"{r2_score(y_test, y_pred):.3f}")
#     col2.metric("Mean Squared Error", f"{mean_squared_error(y_test, y_pred):.3f}")
#
#     # --- Graph: Actual vs Predicted ---
#     st.subheader("Actual vs Predicted Graph")
#     fig, ax = plt.subplots()
#     ax.scatter(y_test, y_pred, alpha=0.7, color="royalblue")
#     ax.set_xlabel("Actual Values")
#     ax.set_ylabel("Predicted Values")
#     ax.set_title(f"Actual vs Predicted ({model_choice})")
#     st.pyplot(fig)
#
#     # --- SHAP Explainability (for XAI Regression) ---
#     if model_choice == "XAI Regression (Explainable Linear Model)":
#         st.subheader("XAI Regression – Feature Impact Visualization")
#         try:
#             explainer = shap.Explainer(model, X_train)
#             shap_values = explainer(X_test)
#
#             shap.summary_plot(shap_values, X_test, show=False)
#             st.pyplot(bbox_inches='tight', dpi=300, pad_inches=0)
#
#             st.markdown("""
#             **Interpretation:**
#             - Each bar shows how much a feature contributed (positive or negative)
#             - Longer bars → greater influence on the target
#             - Color (red/blue) shows direction of influence
#             """)
#         except Exception as e:
#             st.warning(f"Could not generate SHAP explanation: {e}")
#
#     elif model_choice == "Random Forest":
#         # Feature importance for Random Forest
#         st.subheader("Feature Importance (Random Forest)")
#         try:
#             importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
#             fig_imp, ax_imp = plt.subplots()
#             importance.plot(kind='bar', ax=ax_imp, color="seagreen")
#             ax_imp.set_title("Feature Importance")
#             st.pyplot(fig_imp)
#         except Exception:
#             pass
#
# # --- Comparison Mode ---
# else:
#     st.subheader("Comparing Multiple Models")
#     results = []
#
#     for name, model in models.items():
#         if "Logistic" in name:
#             continue  # skip logistic for continuous regression
#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_test)
#         results.append({
#             "Model": name,
#             "R² Score": r2_score(y_test, y_pred),
#             "MSE": mean_squared_error(y_test, y_pred)
#         })
#
#     results_df = pd.DataFrame(results).set_index("Model")
#     st.dataframe(results_df.style.highlight_max(axis=0, color='lightgreen'))
#
#     # --- Plot R² Score Comparison ---
#     fig_r2, ax_r2 = plt.subplots()
#     results_df["R² Score"].plot(kind="bar", ax=ax_r2, color="skyblue", edgecolor="black")
#     ax_r2.set_title("Model Comparison – R² Score")
#     ax_r2.set_ylabel("R² Score")
#     st.pyplot(fig_r2)
#
#     # --- Plot MSE Comparison ---
#     fig_mse, ax_mse = plt.subplots()
#     results_df["MSE"].plot(kind="bar", ax=ax_mse, color="salmon", edgecolor="black")
#     ax_mse.set_title("Model Comparison – Mean Squared Error")
#     ax_mse.set_ylabel("MSE")
#     st.pyplot(fig_mse)
#
#     st.markdown("""
#     **Interpretation:**
#     - **Higher R²** → better model fit
#     - **Lower MSE** → smaller prediction errors
#     - Use XAI Regression to interpret *why* a model performs the way it does.
#     """)

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# ----------------------------
# App Title
# ----------------------------
st.title("🔍 Explainable AI (XAI) for Regression")
st.markdown("""
Train regression models and explore explainability insights using SHAP and correlation heatmaps.
Select a dataset, pick a model, and see predictions and explanations below.
""")

# ----------------------------
# Dataset Selection
# ----------------------------
st.sidebar.header("1️⃣ Dataset Selection")
dataset_choice = st.sidebar.selectbox(
    "Select Dataset",
    ["California Housing (Default)", "Upload your own CSV"]
)

if dataset_choice == "California Housing (Default)":
    from sklearn.datasets import fetch_california_housing
    housing = fetch_california_housing(as_frame=True)
    X = housing.data
    y = housing.target
    st.sidebar.success("✅ Using California Housing dataset")
else:
    uploaded_file = st.sidebar.file_uploader("📂 Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, encoding="latin1")
        except UnicodeDecodeError:
            df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")
        target_col = st.sidebar.selectbox("🎯 Select Target Column (y)", df.columns)
        y = df[target_col]
        X = df.drop(columns=[target_col])
        st.sidebar.success("✅ Custom dataset loaded")
    else:
        st.warning("Please upload a CSV file to continue.")
        st.stop()

# ----------------------------
# Model Selection
# ----------------------------
st.sidebar.header("2️⃣ Choose Regression Model")
model_choice = st.sidebar.selectbox(
    "Select a Model",
    ["Linear Regression", "Decision Tree Regressor", "Random Forest Regressor"]
)

# ----------------------------
# Train & Explain
# ----------------------------
if st.sidebar.button("🚀 Train & Explain"):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    if model_choice == "Linear Regression":
        model = LinearRegression()
    elif model_choice == "Decision Tree Regressor":
        model = DecisionTreeRegressor(random_state=42)
    else:
        model = RandomForestRegressor(random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # ----------------------------
    # Model Evaluation Metrics
    # ----------------------------
    st.subheader("📈 Model Evaluation")
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.metric("Mean Squared Error (MSE)", f"{mse:.2f}")
    st.metric("R² Score", f"{r2:.3f}")

    # Regression Plot
    st.subheader("📊 Regression Plot: Actual vs Predicted")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.scatterplot(x=y_test, y=y_pred, color="dodgerblue", label="Predicted")
    sns.lineplot(x=y_test, y=y_test, color="orange", label="Ideal Fit")
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title(f"{model_choice} Regression Plot")
    ax.legend()
    st.pyplot(fig)

    # ----------------------------
    # SHAP Explainability
    # ----------------------------
    st.subheader("🤖 SHAP Global Feature Importance & Local Explanation")
    try:
        # Use background sampling for speed
        if model_choice in ["Decision Tree Regressor", "Random Forest Regressor"]:
            background = shap.kmeans(X_train, 50)  # summarize 50 samples
            explainer = shap.TreeExplainer(model, data=background)  # Removed check_additivity
            shap_values = explainer.shap_values(X_test)
        else:  # Linear Regression
            background = shap.sample(X_train, 50)
            explainer = shap.KernelExplainer(model.predict, background)
            shap_values = explainer.shap_values(X_test)

        # SHAP summary plot (global importance)
        st.write("**Global Feature Importance (SHAP summary plot):**")
        fig1 = plt.figure()
        shap.summary_plot(shap_values, X_test, show=False)
        st.pyplot(fig1)

        # Local explanation for first test sample
        st.write("**Local Explanation (First Sample Waterfall Plot):**")
        fig2 = plt.figure()
        shap.waterfall_plot(shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value if isinstance(explainer.expected_value, (int, float)) else
            explainer.expected_value[0],
            data=X_test.iloc[0].values,
            feature_names=X_test.columns.tolist()
        ), show=False)
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"❌ SHAP explainability failed: {e}")
    # ----------------------------
    # Correlation Heatmap
    # ----------------------------
    st.subheader("📌 Feature Correlation Heatmap")
    corr = X.corr()
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar=True, ax=ax3)
    ax3.set_title("Feature Correlation Heatmap")
    st.pyplot(fig3)

    # ----------------------------
    # Feature Importance (optional bar plot for trees)
    # ----------------------------
    st.subheader("🌟 Feature Importance (Model-Based)")
    if model_choice in ["Decision Tree Regressor", "Random Forest Regressor"]:
        importance = model.feature_importances_
    else:  # Linear Regression
        importance = np.abs(model.coef_)
    fi_df = pd.DataFrame({"Feature": X.columns, "Importance": importance}).sort_values(by="Importance", ascending=False)
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    sns.barplot(x="Importance", y="Feature", data=fi_df, palette="viridis", ax=ax4)
    ax4.set_title("Feature Importance")
    st.pyplot(fig4)
