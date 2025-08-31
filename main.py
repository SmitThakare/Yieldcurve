# Finance Data Visualization App
# A full-stack application for Treasury yield curves and macroeconomic sentiment analysis

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import requests
import sqlite3
import asyncio
import aiohttp
from textblob import TextBlob
import re
import json
from typing import Dict, List, Tuple, Optional
import time
from threading import Thread
import schedule
from concurrent.futures import ThreadPoolExecutor
import logging
from dataclasses import dataclass
import hashlib

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
@dataclass
class Config:
    FRED_API_KEY = "523c8dd40e837ee09530c0e9decac6bb"  # Get from https://fred.stlouisfed.org/
    DB_PATH = "finance_data.db"
    CACHE_DURATION = 3600  # 1 hour in seconds
    UPDATE_INTERVAL = 300  # 5 minutes in seconds

config = Config()

# Database Schema and Operations
class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Yield curve data
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS yield_curves (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                maturity TEXT NOT NULL,
                yield_rate REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(date, maturity)
            )
        """)
        
        # News and Fed speeches
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS text_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT,
                url TEXT,
                published_date TEXT NOT NULL,
                raw_sentiment REAL,
                processed_sentiment REAL,
                entities TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Daily sentiment aggregates
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sentiment_aggregates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                source_type TEXT NOT NULL,
                avg_sentiment REAL NOT NULL,
                sentiment_count INTEGER NOT NULL,
                volatility REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(date, source_type)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def insert_yield_data(self, data: pd.DataFrame):
        """Insert yield curve data"""
        conn = sqlite3.connect(self.db_path)
        data.to_sql('yield_curves', conn, if_exists='append', index=False)
        conn.close()
    
    def insert_text_data(self, source: str, title: str, content: str, url: str, 
                        published_date: str, sentiment_scores: Dict):
        """Insert text data with sentiment scores"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO text_data 
            (source, title, content, url, published_date, raw_sentiment, processed_sentiment, entities)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (source, title, content, url, published_date, 
              sentiment_scores.get('raw'), sentiment_scores.get('processed'),
              json.dumps(sentiment_scores.get('entities', []))))
        
        conn.commit()
        conn.close()
    
    def get_yield_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Retrieve yield curve data for date range"""
        conn = sqlite3.connect(self.db_path)
        query = """
            SELECT date, maturity, yield_rate 
            FROM yield_curves 
            WHERE date BETWEEN ? AND ?
            ORDER BY date, maturity
        """
        df = pd.read_sql_query(query, conn, params=(start_date, end_date))
        conn.close()
        return df
    
    def get_sentiment_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Retrieve sentiment aggregate data"""
        conn = sqlite3.connect(self.db_path)
        query = """
            SELECT date, source_type, avg_sentiment, sentiment_count, volatility
            FROM sentiment_aggregates 
            WHERE date BETWEEN ? AND ?
            ORDER BY date
        """
        df = pd.read_sql_query(query, conn, params=(start_date, end_date))
        conn.close()
        return df

# Data Fetchers
class TreasuryDataFetcher:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.stlouisfed.org/fred/series/observations"
        
        # Treasury yield series IDs from FRED
        self.yield_series = {
            "1M": "DGS1MO",
            "3M": "DGS3MO", 
            "6M": "DGS6MO",
            "1Y": "DGS1",
            "2Y": "DGS2",
            "3Y": "DGS3",
            "5Y": "DGS5",
            "7Y": "DGS7",
            "10Y": "DGS10",
            "20Y": "DGS20",
            "30Y": "DGS30"
        }
    
    async def fetch_yield_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch Treasury yield data from FRED API"""
        all_data = []
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for maturity, series_id in self.yield_series.items():
                task = self._fetch_series(session, series_id, maturity, start_date, end_date)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, pd.DataFrame) and not result.empty:
                    all_data.append(result)
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()
    
    async def _fetch_series(self, session: aiohttp.ClientSession, series_id: str, 
                           maturity: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch individual yield series"""
        params = {
            'series_id': series_id,
            'api_key': self.api_key,
            'file_type': 'json',
            'observation_start': start_date,
            'observation_end': end_date
        }
        
        try:
            async with session.get(self.base_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    observations = data.get('observations', [])
                    
                    df_data = []
                    for obs in observations:
                        if obs['value'] != '.':  # Filter out missing values
                            df_data.append({
                                'date': obs['date'],
                                'maturity': maturity,
                                'yield_rate': float(obs['value'])
                            })
                    
                    return pd.DataFrame(df_data)
        except Exception as e:
            logger.error(f"Error fetching {series_id}: {e}")
            return pd.DataFrame()

class NewsDataFetcher:
    def __init__(self):
        self.fed_rss_urls = [
            "https://www.federalreserve.gov/feeds/press_all.xml",
            "https://www.federalreserve.gov/feeds/speeches.xml"
        ]
        
        self.financial_rss_urls = [
            "https://feeds.bloomberg.com/markets/news.rss",
            "https://www.reuters.com/business/finance/rss",
            "https://feeds.finance.yahoo.com/rss/2.0/headline"
        ]
    
    async def fetch_fed_data(self, days_back: int = 7) -> List[Dict]:
        """Fetch Federal Reserve speeches and press releases"""
        # Simplified implementation - in production, use feedparser
        articles = []
        
        # Mock Fed data for demonstration
        mock_fed_articles = [
            {
                'source': 'Federal Reserve',
                'title': 'Federal Reserve maintains target range for federal funds rate',
                'content': 'The Federal Open Market Committee decided to maintain the target range for the federal funds rate at 5-1/4 to 5-1/2 percent.',
                'url': 'https://www.federalreserve.gov/newsevents/pressreleases/monetary20231101a.htm',
                'published_date': (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            },
            {
                'source': 'Federal Reserve',
                'title': 'Chair Powell discusses economic outlook and monetary policy',
                'content': 'Federal Reserve Chair Jerome Powell discussed the current economic outlook, emphasizing data-dependent approach to monetary policy.',
                'url': 'https://www.federalreserve.gov/newsevents/speech/powell20231102a.htm',
                'published_date': (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')
            }
        ]
        
        return mock_fed_articles
    
    async def fetch_financial_news(self, days_back: int = 7) -> List[Dict]:
        """Fetch financial news articles"""
        # Mock financial news data for demonstration
        mock_news_articles = [
            {
                'source': 'Financial News',
                'title': 'Treasury yields climb as investors assess Fed policy outlook',
                'content': 'U.S. Treasury yields rose across maturities as investors weighed the Federal Reserve latest monetary policy stance.',
                'url': 'https://example.com/treasury-yields-climb',
                'published_date': (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            },
            {
                'source': 'Financial News', 
                'title': 'Economic data suggests continued resilience in labor markets',
                'content': 'Recent employment data indicates sustained strength in the U.S. labor market, supporting economic growth expectations.',
                'url': 'https://example.com/labor-market-resilience',
                'published_date': (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')
            }
        ]
        
        return mock_news_articles

# NLP Sentiment Analysis Pipeline
class SentimentAnalyzer:
    def __init__(self):
        self.economic_terms = [
            'inflation', 'deflation', 'recession', 'recovery', 'growth', 'gdp',
            'unemployment', 'employment', 'interest rate', 'monetary policy',
            'fiscal policy', 'quantitative easing', 'tapering', 'hawkish', 'dovish'
        ]
    
    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment using TextBlob and economic context"""
        # Basic sentiment analysis
        blob = TextBlob(text)
        raw_sentiment = blob.sentiment.polarity
        
        # Economic context weighting
        economic_context_score = self._calculate_economic_context(text)
        processed_sentiment = (raw_sentiment * 0.7) + (economic_context_score * 0.3)
        
        # Extract entities (simplified)
        entities = self._extract_entities(text)
        
        return {
            'raw': raw_sentiment,
            'processed': processed_sentiment,
            'entities': entities,
            'confidence': abs(processed_sentiment)
        }
    
    def _calculate_economic_context(self, text: str) -> float:
        """Calculate sentiment based on economic terminology"""
        text_lower = text.lower()
        positive_terms = ['growth', 'recovery', 'strong', 'robust', 'stable', 'improving']
        negative_terms = ['recession', 'decline', 'weak', 'concerns', 'uncertainty', 'volatility']
        
        positive_count = sum(1 for term in positive_terms if term in text_lower)
        negative_count = sum(1 for term in negative_terms if term in text_lower)
        
        total_terms = positive_count + negative_count
        if total_terms == 0:
            return 0.0
        
        return (positive_count - negative_count) / total_terms
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract economic entities from text"""
        entities = []
        text_lower = text.lower()
        
        for term in self.economic_terms:
            if term in text_lower:
                entities.append(term)
        
        return entities

# Data Processing and Caching
class DataProcessor:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.treasury_fetcher = TreasuryDataFetcher(config.FRED_API_KEY)
        self.news_fetcher = NewsDataFetcher()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.cache = {}
        self.cache_timestamps = {}
    
    def get_cache_key(self, *args) -> str:
        """Generate cache key from arguments"""
        return hashlib.md5(str(args).encode()).hexdigest()
    
    def is_cache_valid(self, key: str) -> bool:
        """Check if cache is still valid"""
        if key not in self.cache_timestamps:
            return False
        return time.time() - self.cache_timestamps[key] < config.CACHE_DURATION
    
    async def update_yield_data(self):
        """Update Treasury yield curve data"""
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        try:
            df = await self.treasury_fetcher.fetch_yield_data(start_date, end_date)
            if not df.empty:
                self.db_manager.insert_yield_data(df)
                logger.info(f"Updated {len(df)} yield data points")
        except Exception as e:
            logger.error(f"Error updating yield data: {e}")
    
    async def update_sentiment_data(self):
        """Update sentiment data from news and Fed sources"""
        try:
            # Fetch Fed data
            fed_articles = await self.news_fetcher.fetch_fed_data()
            for article in fed_articles:
                sentiment_scores = self.sentiment_analyzer.analyze_sentiment(
                    f"{article['title']} {article['content']}"
                )
                self.db_manager.insert_text_data(
                    article['source'], article['title'], article['content'],
                    article['url'], article['published_date'], sentiment_scores
                )
            
            # Fetch financial news
            news_articles = await self.news_fetcher.fetch_financial_news()
            for article in news_articles:
                sentiment_scores = self.sentiment_analyzer.analyze_sentiment(
                    f"{article['title']} {article['content']}"
                )
                self.db_manager.insert_text_data(
                    article['source'], article['title'], article['content'],
                    article['url'], article['published_date'], sentiment_scores
                )
            
            logger.info(f"Updated sentiment data: {len(fed_articles)} Fed articles, {len(news_articles)} news articles")
        except Exception as e:
            logger.error(f"Error updating sentiment data: {e}")
    
    def get_processed_data(self, start_date: str, end_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get processed yield and sentiment data with caching"""
        cache_key = self.get_cache_key('processed_data', start_date, end_date)
        
        if self.is_cache_valid(cache_key):
            return self.cache[cache_key]
        
        # Fetch yield data
        yield_df = self.db_manager.get_yield_data(start_date, end_date)
        
        # Fetch sentiment data
        sentiment_df = self.db_manager.get_sentiment_data(start_date, end_date)
        
        # Cache results
        self.cache[cache_key] = (yield_df, sentiment_df)
        self.cache_timestamps[cache_key] = time.time()
        
        return yield_df, sentiment_df

# Background Task Scheduler
class BackgroundScheduler:
    def __init__(self, data_processor: DataProcessor):
        self.data_processor = data_processor
        self.running = False
    
    def start_scheduler(self):
        """Start background data updates"""
        if not self.running:
            self.running = True
            schedule.every(5).minutes.do(self._run_updates)
            
            def run_schedule():
                while self.running:
                    schedule.run_pending()
                    time.sleep(60)
            
            Thread(target=run_schedule, daemon=True).start()
            logger.info("Background scheduler started")
    
    def _run_updates(self):
        """Run async updates in background"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            loop.run_until_complete(self.data_processor.update_yield_data())
            loop.run_until_complete(self.data_processor.update_sentiment_data())
            
            loop.close()
        except Exception as e:
            logger.error(f"Background update error: {e}")

# Streamlit Application
class FinanceApp:
    def __init__(self):
        self.db_manager = DatabaseManager(config.DB_PATH)
        self.data_processor = DataProcessor(self.db_manager)
        self.scheduler = BackgroundScheduler(self.data_processor)
        
        # Initialize with sample data if database is empty
        self._initialize_sample_data()
        
        # Start background updates
        self.scheduler.start_scheduler()
    
    def _initialize_sample_data(self):
        """Initialize with sample data for demonstration"""
        # Generate sample yield curve data
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
        maturities = ['3M', '6M', '1Y', '2Y', '5Y', '10Y', '30Y']
        base_yields = {'3M': 4.5, '6M': 4.7, '1Y': 4.9, '2Y': 4.8, '5Y': 4.6, '10Y': 4.4, '30Y': 4.5}
        
        sample_yield_data = []
        for date in dates[::7]:  # Weekly data
            for maturity in maturities:
                base_yield = base_yields[maturity]
                noise = np.random.normal(0, 0.1)
                yield_rate = max(0, base_yield + noise)
                
                sample_yield_data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'maturity': maturity,
                    'yield_rate': yield_rate
                })
        
        df_yield = pd.DataFrame(sample_yield_data)
        
        try:
            self.db_manager.insert_yield_data(df_yield)
            logger.info("Sample yield data initialized")
        except:
            pass  # Data already exists
        
        # Generate sample sentiment data
        sample_sentiment_data = []
        for date in dates[::3]:  # Every 3 days
            for source_type in ['Federal Reserve', 'Financial News']:
                sentiment = np.random.normal(0, 0.3)
                sample_sentiment_data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'source_type': source_type,
                    'avg_sentiment': sentiment,
                    'sentiment_count': np.random.randint(5, 20),
                    'volatility': abs(np.random.normal(0, 0.1))
                })
        
        # Insert sentiment aggregates (simplified for sample data)
        conn = sqlite3.connect(self.db_manager.db_path)
        df_sentiment = pd.DataFrame(sample_sentiment_data)
        try:
            df_sentiment.to_sql('sentiment_aggregates', conn, if_exists='append', index=False)
            logger.info("Sample sentiment data initialized")
        except:
            pass  # Data already exists
        finally:
            conn.close()
    
    def run(self):
        """Main Streamlit application"""
        st.set_page_config(
            page_title="Finance Data Visualization",
            page_icon="📈",
            layout="wide"
        )
        
        st.title("📈 Treasury Yield Curves & Macroeconomic Sentiment")
        st.markdown("*Real-time analysis of U.S. Treasury yields with AI-powered sentiment insights*")
        
        # Sidebar controls
        st.sidebar.header("📊 Controls")
        
        # Date range selector
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime(2023, 1, 1),
                min_value=datetime(2023, 1, 1),
                max_value=datetime(2024, 1, 1)
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime(2024, 1, 1),
                min_value=datetime(2023, 1, 1),
                max_value=datetime(2024, 1, 1)
            )
        
        # Filters
        sentiment_sources = st.sidebar.multiselect(
            "Sentiment Sources",
            options=["Federal Reserve", "Financial News"],
            default=["Federal Reserve", "Financial News"]
        )
        
        sentiment_threshold = st.sidebar.slider(
            "Sentiment Filter",
            min_value=-1.0,
            max_value=1.0,
            value=(-1.0, 1.0),
            step=0.1,
            format="%.1f"
        )
        
        # Refresh button
        if st.sidebar.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
        
        # Historical comparison
        st.sidebar.header("📅 Historical Comparison")
        comparison_periods = st.sidebar.multiselect(
            "Compare with",
            options=["2008 Financial Crisis", "2020 Pandemic", "2022 Inflation Surge"],
            default=[]
        )
        
        # Main dashboard
        try:
            yield_df, sentiment_df = self.data_processor.get_processed_data(
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
            
            if yield_df.empty:
                st.warning("No yield curve data available for the selected period.")
                return
            
            # Filter sentiment data
            if not sentiment_df.empty:
                sentiment_df = sentiment_df[
                    (sentiment_df['source_type'].isin(sentiment_sources)) &
                    (sentiment_df['avg_sentiment'] >= sentiment_threshold[0]) &
                    (sentiment_df['avg_sentiment'] <= sentiment_threshold[1])
                ]
            
            # Create main visualization
            self._create_main_charts(yield_df, sentiment_df)
            
            # Additional analysis sections
            col1, col2 = st.columns(2)
            
            with col1:
                self._create_yield_curve_analysis(yield_df)
            
            with col2:
                self._create_sentiment_analysis(sentiment_df)
            
            # Expandable sections
            with st.expander("🔍 Detailed Data Explorer"):
                self._create_data_explorer(yield_df, sentiment_df)
            
            with st.expander("📰 Recent News & Sentiment Samples"):
                self._create_news_samples()
            
            with st.expander("📊 Statistical Summary"):
                self._create_statistical_summary(yield_df, sentiment_df)
        
        except Exception as e:
            st.error(f"Error loading data: {e}")
            logger.error(f"Dashboard error: {e}")
    
    def _create_main_charts(self, yield_df: pd.DataFrame, sentiment_df: pd.DataFrame):
        """Create main yield curve and sentiment overlay charts"""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('U.S. Treasury Yield Curves', 'Macroeconomic Sentiment Trends'),
            vertical_spacing=0.1,
            specs=[[{"secondary_y": False}], [{"secondary_y": True}]]
        )
        
        # Yield curve plot
        if not yield_df.empty:
            # Get latest yield curve
            latest_date = yield_df['date'].max()
            latest_yields = yield_df[yield_df['date'] == latest_date]
            
            # Sort by maturity for proper curve
            maturity_order = ['1M', '3M', '6M', '1Y', '2Y', '3Y', '5Y', '7Y', '10Y', '20Y', '30Y']
            latest_yields = latest_yields.set_index('maturity').reindex(maturity_order).reset_index()
            latest_yields = latest_yields.dropna()
            
            fig.add_trace(
                go.Scatter(
                    x=latest_yields['maturity'],
                    y=latest_yields['yield_rate'],
                    mode='lines+markers',
                    name=f'Yield Curve ({latest_date})',
                    line=dict(color='#1f77b4', width=3),
                    marker=dict(size=8)
                ),
                row=1, col=1
            )
        
        # Sentiment trends
        if not sentiment_df.empty:
            for source in sentiment_df['source_type'].unique():
                source_data = sentiment_df[sentiment_df['source_type'] == source]
                source_data = source_data.sort_values('date')
                
                color = '#ff7f0e' if source == 'Federal Reserve' else '#2ca02c'
                
                fig.add_trace(
                    go.Scatter(
                        x=pd.to_datetime(source_data['date']),
                        y=source_data['avg_sentiment'],
                        mode='lines+markers',
                        name=f'{source} Sentiment',
                        line=dict(color=color, width=2),
                        marker=dict(size=4)
                    ),
                    row=2, col=1
                )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="Treasury Yields & Market Sentiment Dashboard"
        )
        
        fig.update_xaxes(title_text="Maturity", row=1, col=1)
        fig.update_yaxes(title_text="Yield (%)", row=1, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Sentiment Score", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _create_yield_curve_analysis(self, yield_df: pd.DataFrame):
        """Create yield curve analysis section"""
        st.subheader("📈 Yield Curve Analysis")
        
        if yield_df.empty:
            st.info("No yield data available")
            return
        
        # Calculate yield curve metrics
        latest_date = yield_df['date'].max()
        latest_yields = yield_df[yield_df['date'] == latest_date]
        
        if not latest_yields.empty:
            # Find 10Y and 2Y yields for spread calculation
            y10 = latest_yields[latest_yields['maturity'] == '10Y']['yield_rate']
            y2 = latest_yields[latest_yields['maturity'] == '2Y']['yield_rate']
            
            if not y10.empty and not y2.empty:
                spread_10_2 = y10.iloc[0] - y2.iloc[0]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "10Y-2Y Spread",
                        f"{spread_10_2:.2f}%",
                        delta=f"{'Inverted' if spread_10_2 < 0 else 'Normal'}"
                    )
                
                with col2:
                    avg_yield = latest_yields['yield_rate'].mean()
                    st.metric("Average Yield", f"{avg_yield:.2f}%")
                
                with col3:
                    yield_volatility = yield_df.groupby('maturity')['yield_rate'].std().mean()
                    st.metric("Avg Volatility", f"{yield_volatility:.2f}%")
        
        # Historical yield chart for specific maturity
        selected_maturity = st.selectbox(
            "Select Maturity for Historical View",
            options=yield_df['maturity'].unique(),
            index=0
        )
        
        maturity_data = yield_df[yield_df['maturity'] == selected_maturity].copy()
        maturity_data['date'] = pd.to_datetime(maturity_data['date'])
        maturity_data = maturity_data.sort_values('date')
        
        fig = px.line(
            maturity_data, 
            x='date', 
            y='yield_rate',
            title=f'{selected_maturity} Treasury Yield Historical Trend'
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    def _create_sentiment_analysis(self, sentiment_df: pd.DataFrame):
        """Create sentiment analysis section"""
        st.subheader("💭 Sentiment Analysis")
        
        if sentiment_df.empty:
            st.info("No sentiment data available")
            return
        
        # Sentiment metrics
        avg_sentiment = sentiment_df['avg_sentiment'].mean()
        sentiment_std = sentiment_df['avg_sentiment'].std()
        
        col1, col2 = st.columns(2)
        
        with col1:
            sentiment_color = "green" if avg_sentiment > 0 else "red"
            st.metric(
                "Overall Sentiment",
                f"{avg_sentiment:.3f}",
                delta=f"{'Positive' if avg_sentiment > 0 else 'Negative'}"
            )
        
        with col2:
            st.metric("Sentiment Volatility", f"{sentiment_std:.3f}")
        
        # Sentiment distribution
        fig = px.histogram(
            sentiment_df,
            x='avg_sentiment',
            color='source_type',
            title='Sentiment Score Distribution',
            nbins=20,
            barmode='overlay',
            opacity=0.7
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    def _create_data_explorer(self, yield_df: pd.DataFrame, sentiment_df: pd.DataFrame):
        """Create detailed data explorer"""
        st.subheader("🔍 Data Explorer")
        
        tab1, tab2 = st.tabs(["Yield Data", "Sentiment Data"])
        
        with tab1:
            if not yield_df.empty:
                st.dataframe(yield_df.head(100), use_container_width=True)
                
                # Download button for yield data
                csv_yield = yield_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Yield Data",
                    data=csv_yield,
                    file_name=f"yield_data_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No yield data to display")
        
        with tab2:
            if not sentiment_df.empty:
                st.dataframe(sentiment_df.head(100), use_container_width=True)
                
                # Download button for sentiment data
                csv_sentiment = sentiment_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Sentiment Data",
                    data=csv_sentiment,
                    file_name=f"sentiment_data_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No sentiment data to display")
    
    def _create_news_samples(self):
        """Display recent news samples with sentiment highlights"""
        st.subheader("📰 Recent News & Analysis")
        
        # Fetch recent text data from database
        conn = sqlite3.connect(self.db_manager.db_path)
        query = """
            SELECT source, title, content, raw_sentiment, processed_sentiment, 
                   entities, published_date, url
            FROM text_data 
            ORDER BY created_at DESC 
            LIMIT 10
        """
        
        try:
            recent_news = pd.read_sql_query(query, conn)
            conn.close()
            
            if not recent_news.empty:
                for idx, row in recent_news.iterrows():
                    with st.container():
                        st.markdown(f"**{row['source']}** - {row['published_date']}")
                        st.markdown(f"**{row['title']}**")
                        
                        # Sentiment indicators
                        col1, col2, col3 = st.columns([1, 1, 2])
                        
                        with col1:
                            sentiment_color = "green" if row['processed_sentiment'] > 0 else "red"
                            st.markdown(
                                f"<span style='color: {sentiment_color}'>Sentiment: {row['processed_sentiment']:.3f}</span>",
                                unsafe_allow_html=True
                            )
                        
                        with col2:
                            confidence = abs(row['processed_sentiment'])
                            st.progress(confidence, text=f"Confidence: {confidence:.2f}")
                        
                        with col3:
                            if row['entities'] and row['entities'] != '[]':
                                try:
                                    entities = json.loads(row['entities'])
                                    if entities:
                                        st.markdown(f"**Keywords:** {', '.join(entities[:5])}")
                                except:
                                    pass
                        
                        # Content preview
                        if row['content']:
                            preview = row['content'][:200] + "..." if len(row['content']) > 200 else row['content']
                            st.markdown(f"*{preview}*")
                        
                        if row['url']:
                            st.markdown(f"[Read full article]({row['url']})")
                        
                        st.divider()
            else:
                st.info("No recent news data available")
        
        except Exception as e:
            st.error(f"Error loading news data: {e}")
    
    def _create_statistical_summary(self, yield_df: pd.DataFrame, sentiment_df: pd.DataFrame):
        """Create statistical summary section"""
        st.subheader("📊 Statistical Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Yield Curve Statistics**")
            if not yield_df.empty:
                yield_stats = yield_df.groupby('maturity')['yield_rate'].agg([
                    'mean', 'std', 'min', 'max', 'count'
                ]).round(3)
                st.dataframe(yield_stats)
            else:
                st.info("No yield data for statistics")
        
        with col2:
            st.markdown("**Sentiment Statistics**")
            if not sentiment_df.empty:
                sentiment_stats = sentiment_df.groupby('source_type')['avg_sentiment'].agg([
                    'mean', 'std', 'min', 'max', 'count'
                ]).round(3)
                st.dataframe(sentiment_stats)
                
                # Correlation analysis
                if len(sentiment_df['source_type'].unique()) > 1:
                    sentiment_pivot = sentiment_df.pivot_table(
                        index='date', 
                        columns='source_type', 
                        values='avg_sentiment'
                    )
                    correlation = sentiment_pivot.corr()
                    
                    st.markdown("**Cross-Source Sentiment Correlation**")
                    fig = px.imshow(
                        correlation,
                        title="Sentiment Correlation Matrix",
                        color_continuous_scale='RdBu_r',
                        aspect='auto'
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No sentiment data for statistics")


# FastAPI Backend (Optional - for production deployment)
class FastAPIBackend:
    """
    Optional FastAPI backend for production deployment.
    This would run separately from Streamlit in a production environment.
    """
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    def create_app(self):
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        
        app = FastAPI(title="Finance Data API")
        
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        @app.get("/yield-curve/{start_date}/{end_date}")
        async def get_yield_curve(start_date: str, end_date: str):
            try:
                data = self.db_manager.get_yield_data(start_date, end_date)
                return data.to_dict('records')
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/sentiment/{start_date}/{end_date}")
        async def get_sentiment(start_date: str, end_date: str):
            try:
                data = self.db_manager.get_sentiment_data(start_date, end_date)
                return data.to_dict('records')
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        return app


# Main Application Runner
def main():
    """Main application entry point"""
    try:
        # Initialize and run the Streamlit app
        app = FinanceApp()
        app.run()
    
    except Exception as e:
        st.error(f"Application failed to start: {e}")
        logger.error(f"Application startup error: {e}")


# For running with uvicorn (FastAPI backend)
def create_fastapi_app():
    """Create FastAPI app for production deployment"""
    db_manager = DatabaseManager(config.DB_PATH)
    backend = FastAPIBackend(db_manager)
    return backend.create_app()


if __name__ == "__main__":
    main()

