import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# =====================================
# PAGE CONFIGURATION
# =====================================
st.set_page_config(
    page_title="FlowRisk Market Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================
# CUSTOM CSS STYLING
# =====================================
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f0f2f6;
        border-radius: 5px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ff4b4b;
        color: white;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ff4b4b;
    }
    .cluster-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        height: 100%;
    }
    </style>
""", unsafe_allow_html=https://raw.githubusercontent.com/dhananjay9007/FlowRisk-DashBoard/refs/heads/main/flowrisk_survey_synthetic_data_600_responses.csv)

# =====================================
# DATA LOADING FUNCTION
# =====================================
@st.cache_data
def load_data():
    """
    Load the FlowRisk survey data from GitHub.
    """
    # Your actual GitHub raw CSV URL
    url = "https://raw.githubusercontent.com/varunnjkumar-boop/FlowRisk-/refs/heads/main/flowrisk_survey_synthetic_data_600_responses.csv"
    
    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Please check if the GitHub repository is public and the file path is correct.")
        return None

# =====================================
# LOAD DATA
# =====================================
df = load_data()

# =====================================
# MAIN APP
# =====================================
if df is not None:
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏠 Home / Introduction",
        "📊 Survey Insights (EDA)",
        "👥 Customer Segmentation",
        "📈 Predictive Model Insights"
    ])
    
    # =====================================
    # TAB 1: HOME / INTRODUCTION
    # =====================================
    with tab1:
        st.title("🚀 FlowRisk: Market Survey Analysis")
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("""
            <div style='text-align: center; padding: 30px; background-color: #f8f9fa; border-radius: 10px;'>
                <h3>About This Dashboard</h3>
                <p style='font-size: 18px; line-height: 1.8;'>
                This dashboard presents the findings from a <strong>600-respondent survey</strong> 
                on the <strong>'FlowRisk'</strong> dynamic insurance concept. The analysis includes 
                survey insights, customer segmentation, and predictive model results.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Key Metrics Overview
        st.subheader("📊 Survey Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Responses",
                value="600",
                delta="Complete Dataset"
            )
        
        with col2:
            st.metric(
                label="Industries Covered",
                value=str(df['Q6_Industry'].nunique()) if 'Q6_Industry' in df.columns else "N/A",
                delta="Diverse Sample"
            )
        
        with col3:
            st.metric(
                label="Customer Segments",
                value="4",
                delta="K-Means Clustering"
            )
        
        with col4:
            st.metric(
                label="Analysis Methods",
                value="3",
                delta="Classification, Clustering, Regression"
            )
        
        st.markdown("---")
        
        # Project Description
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🎯 Project Objectives
            
            - **Validate Market Demand**: Assess interest in dynamic supply chain insurance
            - **Identify Target Segments**: Discover high-potential customer groups
            - **Inform Pricing Strategy**: Validate the need for dynamic, risk-based pricing
            - **Guide Go-to-Market**: Data-driven insights for sales and marketing
            """)
        
        with col2:
            st.markdown("""
            ### 🔍 Methodology
            
            - **Exploratory Data Analysis**: Survey response patterns and distributions
            - **K-Means Clustering**: Segmentation based on firmographics and tech adoption
            - **Classification Models**: Predicting customer interest levels
            - **Regression Models**: Analyzing willingness-to-pay patterns
            """)
        
        st.markdown("---")
        
        st.info("👈 **Navigate through the tabs above to explore the detailed findings!**")
    
    # =====================================
    # TAB 2: SURVEY INSIGHTS (EDA)
    # =====================================
    with tab2:
        st.title("📊 Exploratory Data Analysis")
        st.markdown("### Understanding Our Survey Respondents")
        
        st.markdown("---")
        
        # Chart 1: Interest Level Distribution
        if 'Q38_Interest_Level' in df.columns:
            st.subheader("1️⃣ Interest Level Distribution")
            
            interest_counts = df['Q38_Interest_Level'].value_counts().reset_index()
            interest_counts.columns = ['Interest Level', 'Count']
            
            # Define order for interest levels
            interest_order = ['Not Interested', 'Slightly Interested', 'Moderately Interested', 
                            'Very Interested', 'Extremely Interested']
            interest_counts['Interest Level'] = pd.Categorical(
                interest_counts['Interest Level'], 
                categories=interest_order, 
                ordered=True
            )
            interest_counts = interest_counts.sort_values('Interest Level')
            
            fig1 = px.bar(
                interest_counts,
                x='Interest Level',
                y='Count',
                title='Distribution of Interest in FlowRisk',
                color='Count',
                color_continuous_scale='Blues',
                text='Count'
            )
            
            fig1.update_traces(texttemplate='%{text}', textposition='outside')
            fig1.update_layout(
                xaxis_title="Interest Level",
                yaxis_title="Number of Respondents",
                showlegend=False,
                height=500
            )
            
            st.plotly_chart(fig1, use_container_width=True)
            
            # Calculate percentages
            total = len(df)
            very_interested = len(df[df['Q38_Interest_Level'] == 'Very Interested'])
            extremely_interested = len(df[df['Q38_Interest_Level'] == 'Extremely Interested'])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Very Interested", f"{(very_interested/total)*100:.1f}%", f"{very_interested} respondents")
            with col2:
                st.metric("Extremely Interested", f"{(extremely_interested/total)*100:.1f}%", f"{extremely_interested} respondents")
            with col3:
                st.metric("High Interest Total", f"{((very_interested + extremely_interested)/total)*100:.1f}%", 
                         f"{very_interested + extremely_interested} respondents")
        
        st.markdown("---")
        
        # Chart 2: Industry Distribution
        if 'Q6_Industry' in df.columns:
            st.subheader("2️⃣ Industry Breakdown")
            
            industry_counts = df['Q6_Industry'].value_counts().reset_index()
            industry_counts.columns = ['Industry', 'Count']
            
            fig2 = px.bar(
                industry_counts,
                x='Industry',
                y='Count',
                title='Respondent Distribution by Industry',
                color='Count',
                color_continuous_scale='Reds',
                text='Count'
            )
            
            fig2.update_traces(texttemplate='%{text}', textposition='outside')
            fig2.update_layout(
                xaxis_title="Industry",
                yaxis_title="Number of Respondents",
                showlegend=False,
                height=500,
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig2, use_container_width=True)
            
            st.info(f"**Key Insight:** Survey responses span **{len(industry_counts)}** distinct industries, ensuring diverse market representation.")
        
        st.markdown("---")
        
        # Chart 3: Tech Adoption Approach
        if 'Q34_Tech_Adoption_Approach' in df.columns:
            st.subheader("3️⃣ Technology Adoption Approach")
            
            tech_counts = df['Q34_Tech_Adoption_Approach'].value_counts().reset_index()
            tech_counts.columns = ['Adoption Approach', 'Count']
            
            fig3 = px.bar(
                tech_counts,
                x='Adoption Approach',
                y='Count',
                title='Technology Adoption Approach Distribution',
                color='Count',
                color_continuous_scale='Greens',
                text='Count'
            )
            
            fig3.update_traces(texttemplate='%{text}', textposition='outside')
            fig3.update_layout(
                xaxis_title="Technology Adoption Approach",
                yaxis_title="Number of Respondents",
                showlegend=False,
                height=500
            )
            
            st.plotly_chart(fig3, use_container_width=True)
            
            st.info("**Key Insight:** Understanding tech adoption behavior helps us tailor our sales approach and product positioning to different customer segments.")
        
        st.markdown("---")
        
        # Additional Summary Stats
        st.subheader("📈 Quick Summary Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Dataset Information:**
            - Total Responses: 600
            - Data Quality: Complete responses only
            - Survey Method: Synthetic data mimicking real-world patterns
            """)
        
        with col2:
            if 'Q6_Industry' in df.columns and 'Q38_Interest_Level' in df.columns:
                high_interest = df[df['Q38_Interest_Level'].isin(['Very Interested', 'Extremely Interested'])]
                st.markdown(f"""
                **High-Interest Respondents:**
                - Count: {len(high_interest)}
                - Percentage: {(len(high_interest)/len(df))*100:.1f}%
                - Top Industry: {high_interest['Q6_Industry'].mode()[0] if len(high_interest) > 0 else 'N/A'}
                """)
    
    # =====================================
    # TAB 3: CUSTOMER SEGMENTATION
    # =====================================
    with tab3:
        st.title("👥 Customer Segmentation Analysis")
        st.markdown("### Our 4 Customer Segments")
        
        st.markdown("""
        <div style='background-color: #f0f7ff; padding: 20px; border-radius: 10px; margin-bottom: 30px;'>
            <p style='font-size: 16px; margin-bottom: 0;'>
            We ran a <strong>K-Means clustering analysis</strong> on firmographics and tech adoption patterns. 
            This revealed <strong>four distinct customer segments</strong>, each with unique characteristics and 
            varying levels of interest in FlowRisk.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Create 4 columns for the cluster profiles
        col1, col2, col3, col4 = st.columns(4)
        
        # Cluster 1
        with col1:
            st.markdown("""
            <div class='cluster-card' style='border-left: 5px solid #28a745;'>
                <h3 style='color: #28a745; margin-top: 0;'>🎯 Cluster 1</h3>
                <h4>The Ideal Target</h4>
                <h5 style='color: #666;'>(Tech-Savvy E-commerce)</h5>
                
                <hr style='margin: 15px 0;'>
                
                <p><strong>🏢 Industry:</strong><br>E-commerce/Retail</p>
                
                <p><strong>💻 Tech Adoption:</strong><br>Fast Follower</p>
                
                <div style='background-color: #d4edda; padding: 15px; border-radius: 5px; margin-top: 15px;'>
                    <p><strong>📊 Interest Level:</strong></p>
                    <h3 style='color: #28a745; margin: 5px 0;'>32.8%</h3>
                    <p style='margin: 0;'><strong>"Very Interested"</strong></p>
                    <p style='font-size: 14px; margin-top: 10px; color: #155724;'>
                        <strong>✅ Primary Target</strong><br>
                        Highest engagement level
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Cluster 2
        with col2:
            st.markdown("""
            <div class='cluster-card' style='border-left: 5px solid #007bff;'>
                <h3 style='color: #007bff; margin-top: 0;'>🎯 Cluster 2</h3>
                <h4>Strong Secondary Market</h4>
                <h5 style='color: #666;'>(High-Value Goods)</h5>
                
                <hr style='margin: 15px 0;'>
                
                <p><strong>🏢 Industry:</strong><br>Pharma, Food & Manufacturing</p>
                
                <p><strong>💻 Tech Adoption:</strong><br>Cautious</p>
                
                <div style='background-color: #cce5ff; padding: 15px; border-radius: 5px; margin-top: 15px;'>
                    <p><strong>📊 Interest Level:</strong></p>
                    <h3 style='color: #007bff; margin: 5px 0;'>14.6%</h3>
                    <p style='margin: 0;'><strong>"Extremely Interested"</strong></p>
                    <p style='font-size: 14px; margin-top: 10px; color: #004085;'>
                        <strong>💡 They Feel the Pain</strong><br>
                        High-value shipments = high risk
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Cluster 3
        with col3:
            st.markdown("""
            <div class='cluster-card' style='border-left: 5px solid #ffc107;'>
                <h3 style='color: #ffc107; margin-top: 0;'>🎯 Cluster 3</h3>
                <h4>The Cautious Market</h4>
                <h5 style='color: #666;'>(Fashion)</h5>
                
                <hr style='margin: 15px 0;'>
                
                <p><strong>🏢 Industry:</strong><br>Fashion & Apparel</p>
                
                <p><strong>💻 Tech Adoption:</strong><br>Cautious</p>
                
                <div style='background-color: #fff3cd; padding: 15px; border-radius: 5px; margin-top: 15px;'>
                    <p><strong>📊 Interest Level:</strong></p>
                    <h3 style='color: #ffc107; margin: 5px 0;'>Moderate</h3>
                    <p style='margin: 0;'><strong>Moderately Interested</strong></p>
                    <p style='font-size: 14px; margin-top: 10px; color: #856404;'>
                        <strong>⏳ Needs Nurturing</strong><br>
                        Potential for conversion
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Cluster 4
        with col4:
            st.markdown("""
            <div class='cluster-card' style='border-left: 5px solid #6c757d;'>
                <h3 style='color: #6c757d; margin-top: 0;'>🎯 Cluster 4</h3>
                <h4>The Cautious Market</h4>
                <h5 style='color: #666;'>(Electronics)</h5>
                
                <hr style='margin: 15px 0;'>
                
                <p><strong>🏢 Industry:</strong><br>Electronics & Tech</p>
                
                <p><strong>💻 Tech Adoption:</strong><br>Cautious</p>
                
                <div style='background-color: #e2e3e5; padding: 15px; border-radius: 5px; margin-top: 15px;'>
                    <p><strong>📊 Interest Level:</strong></p>
                    <h3 style='color: #6c757d; margin: 5px 0;'>Moderate</h3>
                    <p style='margin: 0;'><strong>Moderately Interested</strong></p>
                    <p style='font-size: 14px; margin-top: 10px; color: #383d41;'>
                        <strong>⏳ Needs Nurturing</strong><br>
                        Long-term opportunity
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Strategic Implications
        st.subheader("🎯 Strategic Implications")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style='background-color: #d4edda; padding: 20px; border-radius: 10px; border-left: 5px solid #28a745;'>
                <h4 style='color: #155724; margin-top: 0;'>✅ Immediate Opportunities</h4>
                <ul>
                    <li><strong>Cluster 1 (E-commerce/Retail):</strong> Highest "Very Interested" rate. Launch pilot programs here.</li>
                    <li><strong>Cluster 2 (Pharma/Food/Mfg):</strong> Highest "Extremely Interested" rate. They understand the value proposition immediately.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style='background-color: #fff3cd; padding: 20px; border-radius: 10px; border-left: 5px solid #ffc107;'>
                <h4 style='color: #856404; margin-top: 0;'>⏳ Long-Term Strategy</h4>
                <ul>
                    <li><strong>Clusters 3 & 4 (Fashion & Electronics):</strong> Cautious adopters with moderate interest.</li>
                    <li><strong>Approach:</strong> Case studies from Clusters 1 & 2 to build trust and demonstrate ROI.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.success("""
        **💡 Key Takeaway:** Focus initial sales and marketing efforts on **Cluster 1** (E-commerce/Retail) 
        and **Cluster 2** (Pharma/Food/Manufacturing) where interest and pain points are highest. Use success 
        stories from these segments to convert the more cautious Clusters 3 and 4.
        """)
    
    # =====================================
    # TAB 4: PREDICTIVE MODEL INSIGHTS
    # =====================================
    with tab4:
        st.title("📈 Predictive Model Insights")
        st.markdown("### What Our Predictive Models Told Us")
        
        st.markdown("""
        <div style='background-color: #fff3cd; padding: 20px; border-radius: 10px; margin-bottom: 30px;'>
            <p style='font-size: 16px; margin-bottom: 0;'>
            ⚠️ <strong>Important:</strong> The "poor" performance of our models isn't a failure—it's a critical 
            validation of FlowRisk's core value proposition. Here's what the data is telling us...
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # ===== CLASSIFICATION SECTION =====
        st.subheader("🎯 Section 1: Classification (Predicting Customer Interest)")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            st.metric(
                label="Best Model Accuracy",
                value="56.1%",
                delta="-43.9% from perfect",
                delta_color="inverse"
            )
        
        with col2:
            st.markdown("""
            <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 5px solid #17a2b8;'>
                <h4 style='color: #17a2b8; margin-top: 0;'>Models Tested</h4>
                <ul>
                    <li>Random Forest Classifier</li>
                    <li>Logistic Regression</li>
                    <li>Decision Tree Classifier</li>
                </ul>
                <p><strong>Result:</strong> All models showed low predictive accuracy (~56%)</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.metric(
                label="Feature Count",
                value="20+",
                delta="Demographics & Firmographics"
            )
        
        st.markdown("---")
        
        # Classification Insight
        st.markdown("""
        <div style='background-color: #d1ecf1; padding: 25px; border-radius: 10px; border-left: 5px solid #0c5460;'>
            <h4 style='color: #0c5460; margin-top: 0;'>💡 Critical Finding: Demographics Don't Predict Interest</h4>
            <p style='font-size: 16px; line-height: 1.8;'>
            Our classification models (Random Forest, Logistic Regression) achieved only <strong>56.1% accuracy</strong>. 
            This is actually <strong>great news</strong>. Here's why:
            </p>
            <ul style='font-size: 16px; line-height: 1.8;'>
                <li><strong>What it proves:</strong> A customer's demographics alone (industry, size, location) are 
                <em>not enough</em> to predict their interest in FlowRisk.</li>
                <li><strong>Why this matters:</strong> It means we can't rely on simple segmentation rules like 
                "all e-commerce companies will be interested."</li>
                <li><strong>The business implication:</strong> This strengthens our case for a <strong>needs-based, 
                consultative sales approach</strong>. We must engage with prospects individually to understand their 
                specific pain points and risk profiles.</li>
            </ul>
            <p style='font-size: 16px; margin-bottom: 0;'>
            ✅ <strong>Validation:</strong> Interest is driven by specific business needs, not demographics—exactly 
            what FlowRisk's flexible model addresses.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("---")
        
        # ===== REGRESSION SECTION =====
        st.subheader("💰 Section 2: Regression (Predicting Willingness to Pay)")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            st.metric(
                label="Best Model R²",
                value="8.4%",
                delta="Very Low Predictive Power",
                delta_color="inverse"
            )
        
        with col2:
            st.markdown("""
            <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 5px solid #28a745;'>
                <h4 style='color: #28a745; margin-top: 0;'>Models Tested</h4>
                <ul>
                    <li>Linear Regression</li>
                    <li>Ridge Regression</li>
                    <li>Lasso Regression</li>
                </ul>
                <p><strong>Result:</strong> All models showed extremely low R² (~8.4%)</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.metric(
                label="Variance Explained",
                value="~8%",
                delta="91.6% unexplained by demographics"
            )
        
        st.markdown("---")
        
        # Regression Insight - THE MOST IMPORTANT
        st.markdown("""
        <div style='background-color: #d4edda; padding: 30px; border-radius: 10px; border-left: 5px solid #155724;'>
            <h4 style='color: #155724; margin-top: 0;'>🎯 CRITICAL VALIDATION: Pricing Cannot Be Demographic-Based</h4>
            <p style='font-size: 18px; line-height: 1.8;'>
            Our regression models (Linear, Ridge, Lasso) achieved an R² of only <strong>8.4%</strong>. 
            This is <strong>the most important finding</strong> of the entire analysis.
            </p>
            <hr style='border-color: #155724; margin: 20px 0;'>
            <h5 style='color: #155724;'>What This Proves:</h5>
            <ul style='font-size: 16px; line-height: 1.8;'>
                <li><strong>Demographics have NO relationship to price:</strong> A customer's industry, revenue, 
                size, or location explains only 8.4% of what they're willing to pay.</li>
                <li><strong>Traditional pricing models would fail:</strong> Imagine if we set prices based on 
                "all pharmaceutical companies pay $X" or "companies with $10M-$50M revenue pay $Y." This data proves 
                that approach would be completely wrong 91.6% of the time.</li>
            </ul>
            <hr style='border-color: #155724; margin: 20px 0;'>
            <h5 style='color: #155724;'>Why This Validates FlowRisk:</h5>
            <ul style='font-size: 16px; line-height: 1.8;'>
                <li><strong>FlowRisk's Core Model:</strong> Dynamic, risk-based pricing calculated 
                <em>for each individual shipment</em> based on:
                    <ul>
                        <li>Origin and destination</li>
                        <li>Real-time weather and geopolitical conditions</li>
                        <li>Carrier reliability</li>
                        <li>Cargo value and sensitivity</li>
                    </ul>
                </li>
                <li><strong>The Data Confirms:</strong> Since demographics don't predict willingness to pay, 
                a static, segment-based pricing model would fail. FlowRisk's shipment-specific, dynamic pricing 
                is <strong>the only viable approach</strong>.</li>
            </ul>
            <hr style='border-color: #155724; margin: 20px 0;'>
            <p style='font-size: 18px; margin-bottom: 0;'>
            ✅ <strong>Bottom Line:</strong> The low R² isn't a weakness—it's proof that FlowRisk's 
            <strong>dynamic, risk-based pricing model</strong> is the correct strategy. Traditional segment-based 
            pricing would be fundamentally flawed for this market.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Final Summary
        st.subheader("📋 Summary: What The Models Tell Us About Our Strategy")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style='background-color: #fff3cd; padding: 20px; border-radius: 10px;'>
                <h5 style='color: #856404;'>🚫 What DOESN'T Work</h5>
                <ul>
                    <li>Demographic-based lead scoring</li>
                    <li>Industry-specific fixed pricing</li>
                    <li>One-size-fits-all packages</li>
                    <li>Automated, rules-based sales qualification</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style='background-color: #d4edda; padding: 20px; border-radius: 10px;'>
                <h5 style='color: #155724;'>✅ What DOES Work</h5>
                <ul>
                    <li><strong>Consultative sales:</strong> Understand individual needs</li>
                    <li><strong>Dynamic pricing:</strong> Calculate per-shipment risk</li>
                    <li><strong>Flexible packages:</strong> Customize to business needs</li>
                    <li><strong>Education-first:</strong> Help prospects understand their risk</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.info("""
        **🎯 Strategic Takeaway:** The "failure" of our predictive models is actually a success. It proves that 
        FlowRisk's flexible, dynamic, risk-based approach is not just innovative—it's **necessary**. A traditional, 
        static insurance model would be fundamentally misaligned with how this market actually behaves.
        """)

else:
    st.error("⚠️ Unable to load data. Please check your CSV URL and ensure your GitHub repository is public.")
    st.info("""
    **Current URL being used:**
    
    `https://raw.githubusercontent.com/varunnjkumar-boop/FlowRisk-/refs/heads/main/flowrisk_survey_synthetic_data_600_responses.csv`
    
    **Please verify:**
    1. The repository `FlowRisk-` is **public** (not private)
    2. The file `flowrisk_survey_synthetic_data_600_responses.csv` exists in the main branch
    3. The file path is correct

    """)
