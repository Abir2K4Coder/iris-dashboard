import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import plotly.express as px
import plotly.graph_objects as go

# Set the page title and layout
st.set_page_config(page_title="Iris Dataset Dashboard", layout="wide")

# --- Load the dataset ---
@st.cache_data
def load_data():
    """Load the Iris dataset and return a pandas DataFrame."""
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    return df

iris_df = load_data()

# --- Title and Description ---
st.title("🌺 Iris Dataset Dashboard")
st.markdown("This interactive dashboard allows you to explore the classic Iris flower dataset.")

# --- Sidebar for user input ---
st.sidebar.header("User Input Features")

# Feature selection for the scatter plot
x_axis = st.sidebar.selectbox('Select X-axis:', iris_df.columns[:-1])
y_axis = st.sidebar.selectbox('Select Y-axis:', iris_df.columns[:-1], index=1)
z_axis = st.sidebar.selectbox('Select Z-axis for 3D plot:', iris_df.columns[:-1], index=2)

# Species selection for filtering
selected_species = st.sidebar.multiselect(
    "Filter by species:",
    options=iris_df['species'].unique(),
    default=iris_df['species'].unique()
)

# New advanced filtering slider
st.sidebar.markdown("---")
st.sidebar.subheader("Advanced Data Filtering")
filter_feature = st.sidebar.selectbox('Filter by a numerical feature:', iris_df.columns[:-1])
min_val = float(iris_df[filter_feature].min())
max_val = float(iris_df[filter_feature].max())
value_range = st.sidebar.slider(
    f"Select a range for {filter_feature}:",
    min_value=min_val,
    max_value=max_val,
    value=(min_val, max_val)
)

# --- Data Filtering ---
filtered_df = iris_df[
    (iris_df['species'].isin(selected_species)) &
    (iris_df[filter_feature] >= value_range[0]) &
    (iris_df[filter_feature] <= value_range[1])
]

# --- Main Dashboard Layout ---
st.header("Visualizations")

# Three columns for visualizations
col1, col2, col3 = st.columns(3)

# --- Scatter Plot ---
with col1:
    st.subheader("Scatter Plot")
    fig_scatter, ax_scatter = plt.subplots()
    sns.scatterplot(
        x=x_axis,
        y=y_axis,
        hue='species',
        data=filtered_df,
        ax=ax_scatter
    )
    st.pyplot(fig_scatter)

# --- Histogram ---
with col2:
    st.subheader("Histogram")
    hist_feature = st.selectbox(
        "Select a feature for the histogram:",
        options=iris_df.columns[:-1],
        index=0
    )
    fig_hist, ax_hist = plt.subplots()
    sns.histplot(
        data=filtered_df,
        x=hist_feature,
        hue='species',
        multiple="stack",
        ax=ax_hist
    )
    st.pyplot(fig_hist)

# --- Box Plot ---
with col3:
    st.subheader("Box Plot")
    box_feature = st.selectbox(
        "Select a feature for the box plot:",
        options=iris_df.columns[:-1],
        index=2
    )
    fig_box, ax_box = plt.subplots()
    sns.boxplot(
        data=filtered_df,
        x='species',
        y=box_feature,
        ax=ax_box
    )
    st.pyplot(fig_box)

st.markdown("---")

# New row for the violin plot and 3D scatter plot
col4, col5 = st.columns(2)

# --- Violin Plot ---
with col4:
    st.subheader("Violin Plot")
    violin_feature = st.selectbox(
        "Select a feature for the violin plot:",
        options=iris_df.columns[:-1],
        index=3
    )
    fig_violin, ax_violin = plt.subplots()
    sns.violinplot(
        data=filtered_df,
        x='species',
        y=violin_feature,
        ax=ax_violin
    )
    st.pyplot(fig_violin)

# --- 3D Scatter Plot ---
with col5:
    st.subheader("3D Scatter Plot")
    st.markdown("Interact with the plot below to view the data from different angles.")
    fig_3d = px.scatter_3d(
        filtered_df,
        x=x_axis,
        y=y_axis,
        z=z_axis,
        color='species',
        title="3D Scatter Plot"
    )
    st.plotly_chart(fig_3d)

st.markdown("---")

# --- Correlation Heatmap and Pair Plot Section ---
st.subheader("Advanced Data Relationships")
col6, col7 = st.columns(2)

# --- Correlation Heatmap ---
with col6:
    st.markdown("### Correlation Heatmap")
    fig_corr, ax_corr = plt.subplots()
    corr_matrix = filtered_df.drop('species', axis=1).corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax_corr)
    st.pyplot(fig_corr)

# --- Pair Plot ---
with col7:
    st.markdown("### Pair Plot")
    st.markdown("Click the button to generate the plot showing relationships between all feature pairs.")
    if st.button("Generate Pair Plot"):
        with st.spinner('Generating pair plot...'):
            fig_pair = sns.pairplot(filtered_df, hue='species')
            st.pyplot(fig_pair)

st.markdown("---")

# --- Raw Data Display ---
st.subheader("Raw Data Table")
st.markdown("Displaying the filtered dataset.")
st.dataframe(filtered_df)

# --- Data Download Button ---
csv_data = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Filtered Data as CSV",
    data=csv_data,
    file_name='filtered_iris_data.csv',
    mime='text/csv',
)

# --- Simple Statistics ---
st.subheader("Data Statistics")
st.markdown("Summary statistics for the entire dataset.")
st.write(iris_df.describe())

# --- Statistics by Species ---
st.subheader("Data Statistics by Species")
st.markdown("Summary statistics for each species in the filtered dataset.")
st.write(filtered_df.groupby('species').describe())