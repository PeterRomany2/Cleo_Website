import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import timedelta, datetime
import streamlit.components.v1 as components
from plotly import graph_objects as go
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Cleo Laboratories Dashboard", layout="wide" ,page_icon="cclogo.png")









def check_login(username, password):
    return username == "sherif" and password == "cleo"

def login_page():
    st.title("Login Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if check_login(username, password):
            st.session_state.logged_in = True  # Set session state
            st.success("Login successful!")
            st.rerun()  # Redirect to the main app
        else:
            st.error("Invalid username or password")

if "logged_in" not in st.session_state or not st.session_state.logged_in:
    login_page()
    st.stop()  # Stop further execution of the page







lottie_html = """
<div style="position: fixed; top: 0; right: 0;">
<script src="https://unpkg.com/@lottiefiles/lottie-player@2.0.8/dist/lottie-player.js"></script><lottie-player src="https://lottie.host/360e8b4f-c02d-44db-ae3e-a65804e51260/wS4DxEonNm.json" background="##FFFFFF" speed="1" style="width: 300px; height: 300px" loop controls autoplay direction="1" mode="normal"></lottie-player>

</div>

<div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);">
<script src="https://unpkg.com/@lottiefiles/lottie-player@2.0.8/dist/lottie-player.js"></script><lottie-player src="https://lottie.host/db72789c-ae59-49a6-ba25-d87d50ae976a/yjyd9YxoBF.json" background="##FFFFFF" speed="1" style="width: 250px; height: 250px" loop  autoplay direction="1" mode="normal"></lottie-player>

</div>

<div style="position: fixed; top: 0; left: 0;">
<script src="https://unpkg.com/@lottiefiles/lottie-player@2.0.8/dist/lottie-player.js"></script><lottie-player src="https://lottie.host/cb3556f9-5948-4c56-b389-963134c139bd/UjO3YxzPgS.json" background="##FFFFFF" speed="1" style="width: 300px; height: 300px" loop controls autoplay direction="1" mode="normal"></lottie-player>

</div>
"""
components.html(lottie_html, height=200)

selected_date_column = 'DATE'


# Helper functions
@st.cache_data
def load_data():
    data = pd.read_csv("Cleo_Data.csv")
    data[selected_date_column] = pd.to_datetime(data[selected_date_column])

    # Rename columns here
    data['NET_CUSTOMERS'] = data['SUBSCRIBERS_GAINED'] - data['SUBSCRIBERS_LOST']


    return data


def custom_quarter(date):
    month = date.month
    year = date.year
    if month in [1, 2, 3]:
        return pd.Period(year=year, quarter=1, freq='Q')
    elif month in [4, 5, 6]:
        return pd.Period(year=year, quarter=2, freq='Q')
    elif month in [7, 8, 9]:
        return pd.Period(year=year, quarter=3, freq='Q')
    else:  # month in [10, 11, 12]
        return pd.Period(year=year, quarter=4, freq='Q')


def aggregate_data(df, freq):
    if freq == 'Q':
        df = df.copy()
        df['CUSTOM_Q'] = df[selected_date_column].apply(custom_quarter)
        df_agg = df.groupby('CUSTOM_Q').agg({
            'SALES': 'sum',
            'ORDERS': 'sum',
            'NET_CUSTOMERS': 'sum',
            'CUSTOMER_REVIEWS': 'sum',
            'COMMENTS': 'sum',
            'SHARES': 'sum',
        })
        return df_agg
    else:
        return df.resample(freq, on=selected_date_column).agg({
            'SALES': 'sum',
            'ORDERS': 'sum',
            'NET_CUSTOMERS': 'sum',
            'CUSTOMER_REVIEWS': 'sum',
            'COMMENTS': 'sum',
            'SHARES': 'sum',
        })


def get_weekly_data(df):
    return aggregate_data(df, 'W')


def get_monthly_data(df):
    return aggregate_data(df, 'M')


def get_quarterly_data(df):
    return aggregate_data(df, 'Q')


def format_with_commas(number):
    return f"{number:,}"


def create_metric_chart(df, column, color, chart_type, height=150, time_frame='Daily'):
    chart_data = df[[column]].copy()
    if time_frame == 'Quarterly':
        chart_data.index = chart_data.index.strftime('%Y Q%q ')
    if chart_type == 'Bar':
        st.bar_chart(chart_data, y=column, color=color, height=height)
    if chart_type == 'Area':
        st.area_chart(chart_data, y=column, color=color, height=height)


def is_period_complete(date, freq):
    today = datetime.now()
    if freq == 'D':
        return date.date() < today.date()
    elif freq == 'W':
        return date + timedelta(days=6) < today
    elif freq == 'M':
        next_month = date.replace(day=28) + timedelta(days=4)
        return next_month.replace(day=1) <= today
    elif freq == 'Q':
        current_quarter = custom_quarter(today)
        return date < current_quarter


def calculate_delta(df, column):
    if len(df) < 2:
        return 0, 0
    current_value = df[column].iloc[-1]
    previous_value = df[column].iloc[-2]
    delta = current_value - previous_value
    delta_percent = (delta / previous_value) * 100 if previous_value != 0 else 0
    return delta, delta_percent


def display_metric(col, title, value, df, column, color, time_frame):
    with col:
        with st.container(border=True):
            delta, delta_percent = calculate_delta(df, column)
            delta_str = f"{delta:+,.0f} ({delta_percent:+.2f}%)"
            st.metric(title, format_with_commas(value), delta=delta_str)
            create_metric_chart(df, column, color, time_frame=time_frame, chart_type=chart_selection)

            last_period = df.index[-1]
            freq = {'Daily': 'D', 'Weekly': 'W', 'Monthly': 'M', 'Quarterly': 'Q'}[time_frame]
            if not is_period_complete(last_period, freq):
                st.caption(
                    f"Note: The last {time_frame.lower()[:-2] if time_frame != 'Daily' else 'day'} is incomplete.")


# Load data
df = load_data()

# Set up input widgets
st.logo(image=r"cclogo.png",
        icon_image=r"1.png")

with st.sidebar:
    st.title("Cleo Laboratories Dashboard")
    st.header("⚙️ Settings")

    max_date = df[selected_date_column].max().date()
    default_start_date = df[selected_date_column].min().date()
    default_end_date = max_date
    start_date = st.date_input("Start date", default_start_date, min_value=df[selected_date_column].min().date(),
                               max_value=max_date)
    end_date = st.date_input("End date", default_end_date, min_value=df[selected_date_column].min().date(),
                             max_value=max_date)
    time_frame = st.selectbox("Select time frame",
                              ("Daily", "Weekly", "Monthly", "Quarterly"),
                              )
    chart_selection = st.selectbox("Select a chart type",
                                   ("Bar", "Area"))

# Prepare data based on selected time frame
if time_frame == 'Daily':
    df_display = df.set_index(selected_date_column)
elif time_frame == 'Weekly':
    df_display = get_weekly_data(df)
elif time_frame == 'Monthly':
    df_display = get_monthly_data(df)
elif time_frame == 'Quarterly':
    df_display = get_quarterly_data(df)

# Display Key Metrics
st.subheader("All-Time Statistics")

metrics = [
    ("Total Customers", "NET_CUSTOMERS", '#29b5e8'),
    ("Total Sales", "SALES", '#FF9F36'),
    ("Total Orders", "ORDERS", '#D45B90'),
    ("Total Customer Reviews", "CUSTOMER_REVIEWS", '#7D44CF')
]

if time_frame == 'Quarterly':
    start_quarter = custom_quarter(start_date)
    end_quarter = custom_quarter(end_date)
    mask = (df_display.index >= start_quarter) & (df_display.index <= end_quarter)
else:
    mask = (df_display.index >= pd.Timestamp(start_date)) & (df_display.index <= pd.Timestamp(end_date))
df_filtered = df_display.loc[mask]

cols = st.columns(4)
for col, (title, column, color) in zip(cols, metrics):
    display_metric(col, title, df_filtered[column].sum(), df_filtered, column, color, time_frame)

# DataFrame display
with st.expander('See DataFrame (Selected time frame)'):
    st.dataframe(df_filtered)




















# Load the Gapminder dataset
df = px.data.gapminder()

# Scatter plot (renamed to reflect skincare products)
fig1 = px.scatter(
    df,
    x="gdpPercap",  # Renamed to "Price"
    y="lifeExp",  # Renamed to "Effectiveness"
    size="pop",  # Renamed to "Product Size"
    color="continent",  # Renamed to "Skin Type"
    hover_name="country",  # Renamed to "Brand"
    hover_data={"gdpPercap": ":,.2f", "lifeExp": ":,.2f", "pop": ":,.0f"},  # Renamed to "Price", "Effectiveness", "Product Size"
    log_x=True,
    size_max=60,
    animation_frame="year",  # Animate over the "year" column
    animation_group="country",  # Group animation by country

)

# Rename x-axis and y-axis titles for scatter plot
fig1.update_layout(
    xaxis_title="Price (USD)",  # Renamed x-axis
    yaxis_title="Rating (scale 36.57-87.58)",  # Renamed y-axis
)

# Customize the hover labels for scatter plot to reflect skincare terminology
fig1.update_traces(
    hovertemplate=(
        "Country: %{hovertext}<br>"  # Keep the country in the hover
        "Price: %{x:.2f}<br>"
        "Rating: %{y:.2f}<br>"
    )
)

# Bar chart (renamed to reflect skincare products)
fig2 = px.bar(
    df,
    x="continent",  # Renamed to "Skin Type"
    y="pop",  # Renamed to "Product Size"
    color="continent",  # Renamed to "Skin Type"
    title="Sales by Region (Serum) Over Time",  # Renamed to reflect skincare theme
    labels={"pop": "Product Size", "continent": "Skin Type"},
    hover_data={"pop": ":,.0f"},  # Renamed to "Product Size"
    animation_frame="year",  # Animate over the "year" column
    range_y=[0, 4000000000]

)

# Rename x-axis and y-axis titles for bar chart
fig2.update_layout(
    xaxis_title="Region (Serum)",  # Renamed x-axis
    yaxis_title="Sales",  # Renamed y-axis
)

# Customize the hover labels for bar chart to reflect skincare terminology
fig2.update_traces(
    hovertemplate=(
        "Sales: %{y:,.0f}<br>"
    )
)

# Streamlit Tabs
tab1, tab2 = st.tabs(["Skincare Price vs. Rating", "Sales by Region (Serum)"])

with tab1:
    # Use the Streamlit theme for the scatter plot.
    st.plotly_chart(fig1, theme="streamlit", use_container_width=True)

with tab2:
    # Use the native Plotly theme for the bar chart.
    st.plotly_chart(fig2, theme="streamlit", use_container_width=True)



# Extended dataset for sales with company-specific metrics and trends over time
np.random.seed(42)

# Simulating sales data over 6 months (representing a trend analysis)
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
categories = ['Skincare', 'Haircare']
products = ['Moisturizer', 'Serum', 'Cleanser', 'Shampoo', 'Conditioner', 'Hair Oil']

# Creating a sales dataset with trend data
data_trends = []
for month in months:
    for category in categories:
        for product in products:
            sales = np.random.randint(1500, 6000)  # Simulate sales data
            sales_growth = np.random.uniform(1, 10)  # Simulated growth in percentage
            marketing_spend = np.random.randint(200, 700)  # Simulated marketing spend
            region = np.random.choice(['North', 'South', 'East', 'West'])  # Random regions
            customer_segment = np.random.choice(['Youth', 'Adults', 'Seniors'])  # Random customer segments
            data_trends.append([month, category, product, sales, sales_growth, marketing_spend, region, customer_segment])

# Convert the sales trend data into a pandas DataFrame
df_trends = pd.DataFrame(data_trends, columns=['Month', 'Category', 'Product', 'Sales', 'Sales_Growth', 'Marketing_Spend', 'Region', 'Customer_Segment'])

# Aggregate sales data by Month and Category to get one line per category
df_aggregated = df_trends.groupby(['Month', 'Category'])['Sales'].sum().reset_index()

# Trend Analysis: Line chart for aggregated sales over time for different categories
fig_trend = px.line(df_aggregated, x='Month', y='Sales', color='Category', title="Sales Trend Over Time",
                    markers=True, line_shape='linear', template='plotly_dark')

# Create a heatmap for geographical segmentation by region
fig_geo = px.density_heatmap(df_trends, x="Month", y="Region", z="Sales", color_continuous_scale="Viridis",
                             title="Sales Distribution by Region")

# Customer Segmentation: Bar chart for sales by customer segment
fig_customer = px.bar(df_trends.groupby(['Customer_Segment', 'Product']).agg({'Sales': 'sum'}).reset_index(),
                      x='Customer_Segment', y='Sales', color='Product', barmode='stack',
                      title="Sales by Customer Segment and Product")

# Display the visualizations side by side
col1, col2 = st.columns(2)

# Display Trend Analysis Chart in the first column
with col1:
    st.plotly_chart(fig_trend)

# Display Geographical Segmentation Heatmap in the second column
with col2:
    st.plotly_chart(fig_geo)

# Display Customer Segmentation Chart below the previous ones
st.plotly_chart(fig_customer)








# Extended dataset for sales with company-specific metrics
data_sales = {
    "Category": ["Skincare", "Skincare", "Skincare", "Haircare", "Haircare", "Haircare"],
    "Sub_category": ["Moisturizer", "Serum", "Cleanser", "Shampoo", "Conditioner", "Hair Oil"],
    "Product_type": ["Brand A", "Brand B", "Brand C", "Brand D", "Brand E", "Brand F"],
    "Sales": [5000, 3000, 2500, 4000, 3500, 2000],  # Sales figures for each product
    "Sales_Growth": [5.0, 2.0, 1.5, 4.5, 3.0, 1.0],  # Sales growth percentage over the last quarter
    "Marketing_Spend": [500, 400, 300, 600, 550, 350],  # Marketing spend per product
    "Profit_Margin": [30, 25, 22, 35, 30, 28],  # Profit margin as a percentage
    "Customer_Satisfaction": [4.6, 4.3, 4.2, 4.4, 4.5, 4.1],  # Customer satisfaction rating
}

# Convert the sales data into a pandas DataFrame
df_sales = pd.DataFrame(data_sales)

# Create the sunburst chart for skincare and haircare sales with additional metrics
fig_sales = px.sunburst(df_sales, path=['Category', 'Sub_category', 'Product_type'], values='Sales',
                        hover_data=['Sales_Growth', 'Marketing_Spend', 'Profit_Margin', 'Customer_Satisfaction'],
                        title="Skincare and Haircare Product Sales")

# Custom dataset for skincare products with frequency of use
data_frequency = {
    "Product": ["Moisturizer", "Sunscreen", "Serum", "Cleanser", "Toner",
                "Exfoliator", "Eye Cream", "Face Mask"],
    "Frequency": [120, 90, 60, 110, 80, 70, 50, 85],  # Frequency of purchase or use
    "Type": ["Moisturizer", "Sunscreen", "Serum", "Cleanser", "Toner",
             "Exfoliator", "Eye Cream", "Face Mask"],  # Product type for coloring
    "Rating": [4.5, 4.7, 4.3, 4.0, 4.2, 3.9, 4.6, 4.1]  # Product ratings out of 5
}

# Convert the frequency data to a pandas DataFrame
df_frequency = pd.DataFrame(data_frequency)

# Create the polar bar chart for skincare product frequency by type
fig_frequency = px.bar_polar(df_frequency,
                             r="Frequency",
                             theta="Product",
                             color="Frequency",
                             template="plotly_dark",
                             color_discrete_sequence=px.colors.sequential.Plasma_r,
                             title="Skincare Product Frequency by Type")

# Display the two charts side by side
col1, col2 = st.columns(2)

# Display the sunburst chart in the first column
with col1:
    st.plotly_chart(fig_sales)

# Display the polar chart in the second column
with col2:
    st.plotly_chart(fig_frequency)















# Create the figure
fig = go.Figure()

# Add the funnel traces for different skincare brands or regions

# Skincare Brand 1 (Montreal)
fig.add_trace(go.Funnel(
    name='Brand A (Region 1)',
    y=["Website visit", "Product reviews read", "Added to cart", "Purchase made", "Repeat purchase"],
    x=[1000, 800, 600, 400, 200],  # Example data for each stage of the funnel
    textinfo="value+percent initial"
))

# Skincare Brand 2 (Toronto)
fig.add_trace(go.Funnel(
    name='Brand B (Region 2)',
    orientation="h",
    y=["Website visit", "Product reviews read", "Added to cart", "Purchase made", "Repeat purchase"],
    x=[1200, 1000, 800, 500, 300],  # Example data for each stage of the funnel
    textposition="inside",
    textinfo="value+percent initial"
))

# Skincare Brand 3 (Vancouver)
fig.add_trace(go.Funnel(
    name='Brand C (Region 3)',
    orientation="h",
    y=["Website visit", "Product reviews read", "Added to cart", "Purchase made", "Repeat purchase"],
    x=[1500, 1200, 900, 600, 400],  # Example data for each stage of the funnel
    textposition="outside",
    textinfo="value+percent initial"
))


fig.update_layout(
    title="Sales Funnel Analysis for Skincare Brands by Region")

# Display the funnel chart in Streamlit
st.plotly_chart(fig, use_container_width=True)













# Function to create Sankey diagram for Skincare Company without COGS
def create_sankey():
    # Define the nodes for the Sankey diagram (categories)
    labels = [
        'Face Creams', 'Serums', 'Lotions',
        'Revenue', 'Marketing', 'Salaries',
        'Rent', 'Profit', 'Website', 'Pharmacy',
        'Raw Materials', 'Shipping', 'Packaging', 'Other Expenses'
    ]

    # Define the flow relationships (sources, targets, and values)
    # The product categories generate revenue, and the revenue is shown as coming from them
    sources = [8, 9, 3, 3, 3, 3, 4, 4, 10, 10, 11, 11, 12]
    targets = [3, 3, 4, 5, 6, 7, 4, 7, 5, 6, 8, 8, 7]
    values = [60000, 40000, 45000, 25000, 20000, 10000, 15000, 5000, 8000, 6000, 7000, 3000, 2000]

    # Create the Sankey diagram using Plotly
    sankey_diagram = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels  # Added more labels here
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values
        )
    ))

    # Update layout and title
    sankey_diagram.update_layout(title="Financial Flow Analysis for Skincare Company")

    return sankey_diagram


# Streamlit app
st.title('Cash Flow Visualization for Skincare Company')
sankey_diagram = create_sankey()
st.plotly_chart(sankey_diagram)













st.title('Customer Segmentation and Satisfaction Prediction for Skincare Products using ML')


# Step 1: Generate enhanced synthetic data for K-Means Clustering (Customer Segmentation)
np.random.seed(42)

# Create 500 data points for customer segmentation (age and purchase frequency)
n_samples = 500  # Decrease the number of samples

# Customer Age (18 to 70+ years)
X = np.random.randn(n_samples, 2)

# Segment 1: Younger customers (18-30) with moderate purchase frequency (1-5)
X[:150, 0] = X[:150, 0] * 5 + 22  # Age: centered around 22 years
X[:150, 1] = X[:150, 1] * 2 + 4  # Purchase Frequency: centered around 4 purchases

# Segment 2: Middle-aged customers (30-50) with higher purchase frequency (5-8)
X[150:300, 0] = X[150:300, 0] * 5 + 40  # Age: centered around 40 years
X[150:300, 1] = X[150:300, 1] * 2 + 6  # Purchase Frequency: centered around 6 purchases

# Segment 3: Older customers (50-70) with low purchase frequency (1-4)
X[300:, 0] = X[300:, 0] * 5 + 55  # Age: centered around 55 years
X[300:, 1] = X[300:, 1] * 2 + 3  # Purchase Frequency: centered around 3 purchases

# Step 2: Apply KMeans clustering (Customer Segments)
kmeans = KMeans(n_clusters=3)  # We want to divide the customers into 3 segments
kmeans.fit(X)

# Prepare the data for plotting
df = pd.DataFrame(X, columns=["Customer Age", "Purchase Frequency"])
df['Customer Segment'] = kmeans.labels_

# Create a custom color palette for clusters
cluster_colors = ['#FF5733', '#33FF57', '#3357FF']  # Red, Green, Blue palette

# Step 3: Create the K-Means clustering plot
fig_kmeans = go.Figure()

for cluster in np.unique(df['Customer Segment']):
    cluster_data = df[df['Customer Segment'] == cluster]
    fig_kmeans.add_trace(go.Scatter(
        x=cluster_data['Customer Age'],
        y=cluster_data['Purchase Frequency'],
        mode='markers',  # Show points as markers
        name=f"Customer Segment {cluster}",
        marker=dict(
            size=8,  # Reduced size of the markers to avoid overlap
            color=cluster_colors[cluster],  # Assign the color for each cluster
            line=dict(width=2, color='Black'),  # Border around each marker
            symbol='circle' if cluster == 0 else 'cross' if cluster == 1 else 'diamond',  # Different marker shapes
            opacity=0.8  # Slight transparency to avoid overlap
        ),
        hovertemplate="<b>Customer Segment: %{text}</b><br>Age: %{x}<br>Purchase Frequency: %{y}<extra></extra>",
        text=[f"Customer Segment {cluster}" for _ in range(len(cluster_data))]  # Add segment info to the tooltip
    ))

# Add the cluster centroids as a separate trace
centroids = kmeans.cluster_centers_
fig_kmeans.add_trace(go.Scatter(
    x=centroids[:, 0],
    y=centroids[:, 1],
    mode='markers+text',
    name="Centroids",
    marker=dict(
        size=15,  # Size of centroid markers
        color='white',  # Change centroid marker color to white
        symbol='star',
        line=dict(width=2, color='black')
    ),
    text=["Centroid"] * len(centroids),
    textposition="top center"
))

# Customize the layout with black background
fig_kmeans.update_layout(
    title="Customer Segmentation for Skincare Company",
    xaxis_title="Customer Age",
    yaxis_title="Purchase Frequency",
    showlegend=True,
    legend_title="Customer Segments",
    height=600,  # Adjust the height to fit the plot better
    width=800,  # Adjust the width for better display
    margin=dict(t=40, l=40, r=40, b=40),  # Adding some margin for better visualization
    plot_bgcolor='rgb(0, 0, 0)',  # Set plot background to black
    paper_bgcolor='rgb(0, 0, 0)',  # Set paper background to black
    xaxis=dict(showgrid=True, gridwidth=1, gridcolor='gray', range=[15, 75]),  # Expanded x-axis range to include all ages
    yaxis=dict(showgrid=True, gridwidth=1, gridcolor='gray', range=[0, 12]),  # Set realistic purchase frequency range
    font=dict(color='white')  # Set font color to white to contrast against the black background
)

# Step 4: Create classification data using Logistic Regression (Predicting Customer Satisfaction)
# Create synthetic labels (0 = Low Satisfaction, 1 = High Satisfaction)
y_class = np.zeros(n_samples)
y_class[150:350] = 1  # Let's assume customers in the age range 30-50 (and moderate purchase) have high satisfaction

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.3, random_state=42)

# Train a Logistic Regression model
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Create the decision boundary for visualization
xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100),
                     np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Step 5: Create the Logistic Regression classification plot
fig_class = go.Figure()

# Plot the decision boundary
fig_class.add_trace(go.Contour(
    z=Z, x=xx[0], y=yy[:, 0], colorscale='Viridis', opacity=0.5, showscale=False
))

# Plot the data points
fig_class.add_trace(go.Scatter(
    x=X[:, 0], y=X[:, 1], mode='markers',
    marker=dict(color=y_class, colorscale='Viridis', size=10, line=dict(width=1, color='black')),
    text=["Class {}".format(i) for i in y_class],
    hovertemplate="<b>Class: %{text}</b><br>X: %{x}<br>Y: %{y}<extra></extra>"
))

# Customize the layout for classification plot
fig_class.update_layout(
    title="Customer Satisfaction Prediction for Skincare Products",
    xaxis_title="Customer Age",
    yaxis_title="Purchase Frequency",
    showlegend=False,
    height=600,  # Adjust height for better fitting
    width=800,  # Adjust width for display
    margin=dict(t=40, l=40, r=40, b=40),  # Adding some margin
    plot_bgcolor='rgb(0, 0, 0)',  # Black background
    paper_bgcolor='rgb(0, 0, 0)',  # Black background
    xaxis=dict(showgrid=True, gridwidth=1, gridcolor='gray'),
    yaxis=dict(showgrid=True, gridwidth=1, gridcolor='gray'),
    font=dict(color='white')  # Font color set to white for contrast
)

# Step 6: Display both K-Means Clustering and Logistic Regression Classification plots using Streamlit
col1, col2 = st.columns(2)

# Display K-Means Clustering plot in the first column
with col1:
    st.plotly_chart(fig_kmeans)

# Display Logistic Regression Classification plot in the second column
with col2:
    st.plotly_chart(fig_class)












