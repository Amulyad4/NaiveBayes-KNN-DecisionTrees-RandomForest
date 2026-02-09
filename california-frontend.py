import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, confusion_matrix, classification_report
)

from sklearn import tree

st.set_page_config(page_title="DT Regression & Classification", layout="wide")
st.title("🏠 California Housing – Decision Tree")

file = st.sidebar.file_uploader("Upload california_housing_test.csv", type="csv")

if file:
    df = pd.read_csv(file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    X = df.drop(columns="median_house_value")
    y_reg = df["median_house_value"]

    y_class = pd.qcut(y_reg, q=3, labels=[0, 1, 2])

    st.sidebar.header("Model Selection")

    task = st.sidebar.selectbox(
        "Select Task",
        ["Regression", "Classification"]
    )

    technique = st.sidebar.selectbox(
        "Select Technique",
        ["Pre-Pruning", "Post-Pruning"]
    )

    if task == "Regression":
        st.subheader("Decision Tree Regression")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_reg, test_size=0.2, random_state=42
        )

        if technique == "Pre-Pruning":
            max_depth = st.sidebar.slider("Max Depth", 1, 20, 5)
            min_samples_split = st.sidebar.slider("Min Samples Split", 2, 20, 2)

            model = DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=42
            )

        elif technique == "Post-Pruning":
            base = DecisionTreeRegressor(random_state=42)
            path = base.cost_complexity_pruning_path(X_train, y_train)

            ccp_alpha = st.sidebar.slider(
                "ccp_alpha",
                float(path.ccp_alphas.min()),
                float(path.ccp_alphas.max()),
                step=0.0005
            )

            model = DecisionTreeRegressor(
                random_state=42,
                ccp_alpha=ccp_alpha
            )

        

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.write("R² Score:", r2_score(y_test, y_pred))
        st.write("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
        st.write("MAE:", mean_absolute_error(y_test, y_pred))
    else:
        st.subheader("Decision Tree Classification")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_class, test_size=0.2, random_state=42
        )

        if technique == "Pre-Pruning":
            max_depth = st.sidebar.slider("Max Depth", 1, 20, 5)
            min_samples_split = st.sidebar.slider("Min Samples Split", 2, 20, 2)

            model = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=42
            )

        elif technique == "Post-Pruning":
            base = DecisionTreeClassifier(random_state=42)
            path = base.cost_complexity_pruning_path(X_train, y_train)

            ccp_alpha = st.sidebar.slider(
                "ccp_alpha",
                float(path.ccp_alphas.min()),
                float(path.ccp_alphas.max()),
                step=0.0005
            )

            model = DecisionTreeClassifier(
                random_state=42,
                ccp_alpha=ccp_alpha
            )

        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.write("Accuracy:", accuracy_score(y_test, y_pred))
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_test, y_pred))
        st.text("Classification Report")
        st.text(classification_report(y_test, y_pred))

    st.subheader("Decision Tree Visualization")
    fig, ax = plt.subplots(figsize=(20, 10))
    tree.plot_tree(
        model,
        feature_names=X.columns,
        filled=True,
        max_depth=max_depth,
        ax=ax
    )
    st.pyplot(fig)
