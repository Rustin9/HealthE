import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


@st.cache_data
def load_csv(path: Path):
    return pd.read_csv(path)


def main():
    st.title("Project Data Visualizer")
    data_dir = Path(__file__).parent / "data"
    datasets = {
        "Diet Dataset": data_dir / "diet_dataset.csv",
        "Health Dataset": data_dir / "new_health_dataset.csv",
    }

    dataset_name = st.sidebar.selectbox("Select dataset", list(datasets.keys()))
    path = datasets[dataset_name]

    if not path.exists():
        st.error(f"Dataset not found: {path}")
        return

    df = load_csv(path)
    st.sidebar.markdown(f"**Rows:** {df.shape[0]}  
**Columns:** {df.shape[1]}")

    if st.sidebar.checkbox("Show raw data", value=False):
        st.dataframe(df.head(200))

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    st.header("Quick Summary")
    with st.expander("Dataframe info"):
        st.write(df.describe(include='all').T)

    st.header("Plots")

    if numeric_cols:
        st.subheader("Numeric plots")
        plot_type = st.selectbox("Plot type", ["Histogram", "Scatter", "Correlation matrix"])

        if plot_type == "Histogram":
            col = st.selectbox("Select numeric column", numeric_cols)
            bins = st.slider("Bins", 5, 200, 30)
            fig, ax = plt.subplots()
            ax.hist(df[col].dropna(), bins=bins, color="#4C72B0", edgecolor="white")
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
            st.pyplot(fig)

        elif plot_type == "Scatter":
            x_col = st.selectbox("X", numeric_cols, index=0)
            y_col = st.selectbox("Y", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
            color_col = st.selectbox("Color by (categorical, optional)", [None] + categorical_cols)
            fig, ax = plt.subplots()
            if color_col:
                groups = df.groupby(color_col)
                for name, g in groups:
                    ax.scatter(g[x_col], g[y_col], label=str(name), alpha=0.7)
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                ax.scatter(df[x_col], df[y_col], alpha=0.6)
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            st.pyplot(fig)

        else:  # correlation
            corr = df[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(min(10, len(numeric_cols)), min(8, len(numeric_cols))))
            im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
            ax.set_xticks(range(len(numeric_cols)))
            ax.set_yticks(range(len(numeric_cols)))
            ax.set_xticklabels(numeric_cols, rotation=45, ha="right")
            ax.set_yticklabels(numeric_cols)
            fig.colorbar(im, ax=ax)
            st.pyplot(fig)
    else:
        st.info("No numeric columns available for plotting in this dataset.")

    if categorical_cols:
        st.subheader("Categorical counts")
        cat_col = st.selectbox("Select categorical column", categorical_cols)
        if cat_col:
            counts = df[cat_col].value_counts().head(40)
            fig, ax = plt.subplots(figsize=(8, 4))
            counts.plot(kind="bar", ax=ax, color="#55A868")
            ax.set_ylabel("Count")
            st.pyplot(fig)

    st.caption("Built with Streamlit — use the sidebar to explore different plots and datasets.")


if __name__ == "__main__":
    main()
