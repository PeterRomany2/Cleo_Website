import streamlit as st
import streamlit.components.v1 as components

# Page Configuration
st.set_page_config(page_title="Ames Housing Overview", layout="wide")
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.warning("You need to log in to access this page.")
    st.stop()  # Stop further execution of the page












# Custom CSS for Enhanced Appearance
st.markdown("""
<style>
    .main { background-color: #f7f9fc; font-family: 'Arial', sans-serif; }
    h1 { color: white; text-align: center; font-size: 2.8em; margin-bottom: 30px; }
    h2 { color: white; font-size: 2em; margin-top: 30px; }
    h3 { color: #4a7ab7; font-size: 1.5em; margin-top: 25px; }
    .stMarkdown p { line-height: 1.6; font-size: 16px; color: #475569; margin-bottom: 15px; }
    .stMarkdown ul { margin-left: 20px; }
    .stMarkdown li { padding: 5px 0; }
    .report-section { background: #ffffff; padding: 30px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1); margin-bottom: 35px; }
    .highlight { border-right: 6px solid #0288d1; border-left: 6px solid #0288d1; padding: 15px 20px; margin: 20px 0; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)

# Title and Description
st.title("Ames Housing Overview")

# Overview Section
st.header("Overview")
with st.container():
    col1, col2 = st.columns([1.5, 1])
    with col1:
        with st.expander("Key Metrics", expanded=True):
            # Upper and Lower Bound
            st.subheader("Upper and Lower Bound")
            st.markdown("""
            - **Spread/Dispersion of Data Points:** This represents the real range around the mean.
            - **Range:** $115.18K to $261.95K
            """)

            # Mean
            st.subheader("Mean")
            st.markdown("""
            - **Mean Sale Price:** The balance point where most values are centered.
            - **Value:** $188.57K
            """)
    with col2:
        st.image(r"Picture2.png", caption="Sale Price Distribution")

# Coefficient of Variation (CV)
st.subheader("Coefficient of Variation (CV) Interpretation for Sale Price")
st.markdown("""
<div class="highlight">
<ul>
        <li><strong>Moderate Coefficient of Variation (CV):</strong> A <strong style="color:orchid;">CV of 39%</strong> indicates a moderate level of dispersion in sale prices. This means the sale prices are somewhat spread out from the mean, but not excessively so.
</li>
        <li><strong>Interpretation:</strong> The sale prices exhibit a moderate degree of variability across transactions. While there is some diversity in the sale prices, with both lower-priced and higher-priced properties, the spread is not extremely wide. This suggests a balanced real estate market with a reasonable range of property values.</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# CV Rate
st.subheader("CV Rate (This represents how much of the data falls within this range)")
st.markdown("""
<div class="highlight">
<ul>
        <li>For the Sale Price, <strong style="color:aqua;">Around 72.00%</strong> of the transactions fall within the boundaries of the mean.</li>
        <li><strong style="color:yellow;">Approximately 61.29%</strong> of the data points fall within a low range (551 out of 899 data points).</li>
        <li>The distribution is described as <strong style="color:lime;">"most in low,"</strong> indicating that the majority of data points are concentrated towards the lower end of the distribution where mode(most frequently occurring value) locate.</li>
        <li>This is further elaborated by specifying that the mode is less than both the median and mean(<strong style="color:orchid;">mode < median= 174K < mean= 188.57K</strong>), suggesting a concentration of values towards the lower end of the distribution.</li>
    </ul>
</div>
""", unsafe_allow_html=True)

st.markdown("""
---
""", unsafe_allow_html=True)


# Insights Section
st.header("Insights")
insights_container = st.container()
with insights_container:
        # Maximum and Minimum Sale Price
        st.subheader("Maximum and Minimum Sale Price")
        st.markdown("""
        - **Maximum Sale Price:** $755,000
        - **Minimum Sale Price:** $13,100
        """)

        # Total Count and Sum of Sale Prices
        st.subheader("Total Count and Sum of Sale Prices")
        st.markdown("""
        - **Total Count of Sale Prices:** 899
        - **Total Sum of Sale Prices:** $172,815,786
        """)

        # Top Sale Prices
        st.subheader("Top Sale Prices")
        st.markdown("""
        - **Top 5 Distinct(most frequently occurring) Sale Prices:** $145,000, $155,000, $140,000, $185,000, $147,000
        - **Mode(most common) for Sale Price:** $145,000 and $155,000
        - **Top 5 Largest Sale Prices:** $755,000, $745,000, $615,000, $591,587, $500,067
        - **Top 5 Smallest Sale Prices:** $13,100, $40,000, $46,500, $52,000, $55,000
        """)


st.markdown("""
---
""", unsafe_allow_html=True)









# Market Overview: House Sale Price Section
st.header("Market Overview: House Sale Price")




# Description of Pie Chart and Image Side by Side in One Row with Equal Columns
with st.container():
    col1, col2 = st.columns([1, 1])  # Both columns take equal space
    with col1:
        st.subheader("Overview")
        st.markdown("""
        - This Pie Chart clarifies the importance of the shape of the home’s area. The chart shows that more than 60% of sales go to homes with a regular shape, followed by 36% for homes with a slightly irregular shape.
        - This indicates that there is a lot of land that must be used for another project, unlike homes.
        """)
    with col2:
        st.subheader("Volume wise lot shape Market")
        st.image(r"Picture2.png", caption="House Shape Distribution in Sales")

# House Sale Industry Section

# Description of Sales by Year and Chart Image Side by Side in One Row with Equal Columns
with st.container():
    col1, col2 = st.columns([1, 1])  # Both columns take equal space
    with col1:
        st.subheader("House Sale Industry")

        st.markdown("""
        - The year 2008 indicates more sales than 2007, with sales reaching more than $100 million, compared to only $60 million in 2007.
        - We observe that regions such as 'Nridght' and 'Somerset' accounted for the most sales in 2007, while in 2008, the best-selling areas were 'NAmes' and 'CollgCr'.
        """)
    with col2:
        st.subheader("Sale Total Volume by Region 2007and2008")
        st.image(r"Picture2.png", caption="Sales Distribution by Year and Region")



st.markdown("""
---
<div style="text-align: center; font-size: 14px; color: #6b7280;">
This app was developed to explore the key statistics of the Ames Housing dataset. The insights provided can help better understand the distribution of sale prices in the market.
</div>
""", unsafe_allow_html=True)




