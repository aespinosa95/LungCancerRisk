"""
Shared styles and navigation components for Lung Cancer Analysis Dashboard
"""

def get_common_styles():
    """
    Returns the common CSS styles for all pages in the dashboard.
    Uses a professional blue medical theme.
    """
    return """
<style>
    /* Import Professional Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    
    /* Main Background - Soft Medical Theme */
    .main {
        background: linear-gradient(160deg, #f8fafc 0%, #f1f5f9 50%, #e0f2fe 100%);
        background-attachment: fixed;
        color: #1e293b;
    }
    
    /* Subtle Medical Pattern Overlay */
    .main::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(circle at 15% 25%, rgba(37, 99, 235, 0.03) 0%, transparent 40%),
            radial-gradient(circle at 85% 75%, rgba(14, 116, 144, 0.02) 0%, transparent 40%);
        pointer-events: none;
        z-index: 0;
    }
    
    /* Navigation Menu */
    .nav-container {
        background: white;
        padding: 16px 32px;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        margin-bottom: 32px;
        border-bottom: 3px solid #3b82f6;
    }
    
    .nav-menu {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        justify-content: center;
        align-items: center;
    }
    
    .nav-button {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 10px 20px;
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border: 2px solid #bfdbfe;
        border-radius: 10px;
        color: #1e40af;
        font-weight: 600;
        font-size: 0.95rem;
        text-decoration: none;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .nav-button:hover {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border-color: #3b82f6;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    
    .nav-button.active {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: white;
        border-color: #1d4ed8;
    }
    
    /* Hero Header Section */
    .hero-section {
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
        padding: 48px 32px;
        border-radius: 20px;
        box-shadow: 0 20px 60px rgba(37, 99, 235, 0.2);
        margin: 0 0 48px 0;
        position: relative;
        overflow: hidden;
    }
    
    .hero-section::before {
        content: "";
        position: absolute;
        top: -50%;
        right: -10%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(255, 255, 255, 0.15) 0%, transparent 70%);
        border-radius: 50%;
    }
    
    .hero-section::after {
        content: "";
        position: absolute;
        bottom: -30%;
        left: -5%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(255, 255, 255, 0.1) 0%, transparent 70%);
        border-radius: 50%;
    }
    
    .hero-content {
        position: relative;
        z-index: 1;
        text-align: center;
        max-width: 1000px;
        margin: 0 auto;
    }
    
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        color: white;
        margin-bottom: 16px;
        line-height: 1.2;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .hero-subtitle {
        font-size: 1.15rem;
        color: rgba(255, 255, 255, 0.95);
        line-height: 1.6;
        font-weight: 500;
        max-width: 800px;
        margin: 0 auto;
    }
    
    .hero-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        padding: 8px 20px;
        border-radius: 30px;
        font-size: 14px;
        font-weight: 600;
        color: white;
        margin-top: 20px;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.75rem;
        font-weight: 700;
        color: #1e293b;
        margin: 50px 0 28px 0;
        padding-bottom: 12px;
        border-bottom: 3px solid #dbeafe;
        display: flex;
        align-items: center;
        gap: 12px;
        position: relative;
    }
    
    .section-header::before {
        content: "";
        width: 8px;
        height: 28px;
        background: linear-gradient(180deg, #3b82f6 0%, #2563eb 100%);
        border-radius: 4px;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.3);
    }
    
    .section-subtitle {
        font-size: 1rem;
        color: #64748b;
        margin-top: -20px;
        margin-bottom: 30px;
        font-weight: 500;
    }
    
    /* KPI Cards */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 20px;
        margin: 32px 0 48px 0;
    }
    
    @media (max-width: 1200px) {
        .kpi-grid {
            grid-template-columns: repeat(2, 1fr);
        }
    }
    
    @media (max-width: 768px) {
        .kpi-grid {
            grid-template-columns: 1fr;
        }
    }
    
    .kpi-card {
        background: white;
        padding: 28px 24px;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06);
        border: 1px solid #f1f5f9;
        position: relative;
        overflow: hidden;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border-top: 4px solid #3b82f6;
    }
    
    .kpi-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 12px 32px rgba(37, 99, 235, 0.15);
        border-color: #bfdbfe;
    }
    
    .kpi-icon {
        width: 48px;
        height: 48px;
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
        margin-bottom: 16px;
    }
    
    .kpi-title {
        font-size: 13px;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }
    
    .kpi-value {
        font-size: 2rem;
        font-weight: 800;
        color: #2563eb;
        line-height: 1.2;
    }
    
    /* Stats Cards */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 20px;
        margin: 24px 0;
    }
    
    .stat-card {
        background: white;
        padding: 24px;
        border-radius: 14px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.05);
        border: 1px solid #f1f5f9;
        border-top: 3px solid #3b82f6;
        transition: all 0.3s ease;
        text-align: center;
    }
    
    .stat-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(37, 99, 235, 0.12);
    }
    
    .stat-icon {
        font-size: 2.5rem;
        margin-bottom: 12px;
    }
    
    .stat-label {
        font-size: 0.85rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }
    
    .stat-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #2563eb;
        line-height: 1.2;
    }
    
    /* Control Panel */
    .control-panel {
        background: white;
        padding: 28px 32px;
        border-radius: 18px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06);
        border: 1px solid #f1f5f9;
        margin-bottom: 40px;
        border-top: 4px solid #3b82f6;
    }
    
    .control-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    /* Chart Containers */
    .chart-container {
        background: white;
        padding: 32px;
        border-radius: 18px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06);
        border: 1px solid #f1f5f9;
        margin-bottom: 32px;
        transition: all 0.3s ease;
    }
    
    .chart-container:hover {
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
    }
    
    .chart-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 20px;
        padding-bottom: 12px;
        border-bottom: 2px solid #f8fafc;
    }
    
    /* Data Table */
    .data-table-container {
        background: white;
        padding: 32px;
        border-radius: 18px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06);
        border: 1px solid #f1f5f9;
        margin-bottom: 40px;
    }
    
    /* Info Boxes */
    .info-box {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border-left: 4px solid #3b82f6;
        padding: 20px 24px;
        border-radius: 12px;
        margin: 24px 0;
        font-size: 0.95rem;
        color: #1e40af;
        line-height: 1.6;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-left: 4px solid #f59e0b;
        padding: 20px 24px;
        border-radius: 12px;
        margin: 24px 0;
        font-size: 0.95rem;
        color: #78350f;
        line-height: 1.6;
    }
    
    /* Variable Type Badges */
    .variable-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-left: 12px;
    }
    
    .badge-numerical {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        color: #1e40af;
    }
    
    .badge-categorical {
        background: linear-gradient(135deg, #fce7f3 0%, #fbcfe8 100%);
        color: #9f1239;
    }
    
    /* Risk Level Badges */
    .risk-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 4px;
    }
    
    .badge-low {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        color: #065f46;
    }
    
    .badge-medium {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        color: #78350f;
    }
    
    .badge-high {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        color: #991b1b;
    }
    
    /* Cluster Badges */
    .cluster-badge {
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 2px solid;
    }
    
    .badge-cluster-0 {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        color: #1e40af;
        border-color: #93c5fd;
    }
    
    .badge-cluster-1 {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        color: #991b1b;
        border-color: #fca5a5;
    }
    
    .badge-cluster-2 {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        color: #065f46;
        border-color: #6ee7b7;
    }
    
    .badge-cluster-3 {
        background: linear-gradient(135deg, #fce7f3 0%, #fbcfe8 100%);
        color: #831843;
        border-color: #f9a8d4;
    }
    
    .badge-cluster-4 {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        color: #78350f;
        border-color: #fcd34d;
    }
    
    .badge-cluster-5 {
        background: linear-gradient(135deg, #e9d5ff 0%, #d8b4fe 100%);
        color: #581c87;
        border-color: #c084fc;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border-right: 1px solid #e2e8f0;
    }
    
    [data-testid="stSidebar"] h1 {
        color: #2563eb;
        font-size: 1.5rem;
        font-weight: 800;
        margin-bottom: 24px;
        padding-bottom: 16px;
        border-bottom: 2px solid #dbeafe;
    }
    
    /* Streamlit Overrides */
    .stSelectbox label, .stMultiSelect label, .stSlider label, .stCheckbox label, .stRadio label {
        color: #1e293b !important;
        font-weight: 600 !important;
        font-size: 14px !important;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    /* Footer */
    .custom-footer {
        background: white;
        border-top: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 40px 32px;
        margin-top: 60px;
        text-align: center;
        box-shadow: 0 -4px 20px rgba(0, 0, 0, 0.04);
    }
    
    .footer-content {
        max-width: 800px;
        margin: 0 auto;
    }
    
    .footer-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 16px;
    }
    
    .footer-text {
        color: #64748b;
        font-size: 0.95rem;
        line-height: 1.6;
        margin-bottom: 24px;
    }
    
    .footer-divider {
        width: 60px;
        height: 3px;
        background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
        margin: 24px auto;
        border-radius: 2px;
    }
    
    .social-section {
        margin-top: 28px;
    }
    
    .social-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 16px;
    }
    
    .social-links {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 20px;
    }
    
    .social-link {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 56px;
        height: 56px;
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        border-radius: 14px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        text-decoration: none;
        border: 2px solid transparent;
    }
    
    .social-link:hover {
        transform: translateY(-4px) scale(1.05);
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        border-color: #93c5fd;
        box-shadow: 0 8px 24px rgba(37, 99, 235, 0.25);
    }
    
    .copyright {
        margin-top: 28px;
        padding-top: 24px;
        border-top: 1px solid #f1f5f9;
        color: #94a3b8;
        font-size: 0.85rem;
    }
</style>
"""


def get_navigation_menu(active_page="Home"):
    """
    Returns the HTML for the navigation menu with the specified active page.
    
    Args:
        active_page (str): The name of the currently active page.
                          Options: "Home", "Distribution", "Correlations", "Clusters", "Relaciones", "Prediction"
    
    Returns:
        str: HTML string for the navigation menu
    """

    pages = {
    "Home": {"emoji": "🏠", "url": "/"},
    "Distribution": {"emoji": "📊", "url": "/Distribution"},
    "Correlations": {"emoji": "🔄", "url": "/Correlations"},
    "Statistical Tests": {"emoji": "📈", "url": "/Statistical_Tests"},
    "Clusters": {"emoji": "🎯", "url": "/Clusters"},
    "Prediction": {"emoji": "🧬", "url": "/Prediction"},
    "Model Performance": {"emoji": "🎓", "url": "/Model_Performance"}
    }


    buttons_html = ""
    for page_name, page_info in pages.items():
        active_class = " active" if page_name == active_page else ""
        buttons_html += f'''<a href="{page_info['url']}" target="_self" class="nav-button{active_class}">{page_info['emoji']} {page_name}</a>
        '''
    
    return f'''<div class="nav-container">
    <div class="nav-menu">{buttons_html}</div>
</div>
'''


def apply_page_style(active_page="Home"):
    """
    Convenience function to apply both styles and navigation menu to a page.
    
    Args:
        active_page (str): The name of the currently active page
    
    Usage:
        import streamlit as st
        from utils.styles import apply_page_style
        
        apply_page_style("Distribution")
    """
    import streamlit as st
    st.markdown(get_common_styles(), unsafe_allow_html=True)
    st.markdown(get_navigation_menu(active_page), unsafe_allow_html=True)