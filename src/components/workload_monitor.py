from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import asdict
import logging
import threading
import time
from collections import deque
from scipy import stats
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')

from ..models.data_models import Metrics
from ..utils.metrics import MetricsCollector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Pattern:
    """Class representing workload patterns"""
    def __init__(self, pattern_type: str, period: int, strength: float):
        self.pattern_type = pattern_type
        self.period = period
        self.strength = strength
        self.last_updated = datetime.now()

class Anomaly:
    """Class representing workload anomalies"""
    def __init__(self, metric: str, timestamp: datetime, value: float, score: float):
        self.metric = metric
        self.timestamp = timestamp
        self.value = value
        self.score = score
        self.analyzed = False

class WorkloadMonitor:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Workload Monitor with configuration
        
        Args:
            config: Configuration dictionary containing:
                - history_size: Size of metrics history
                - analysis_interval: Interval for analysis updates
                - prediction_horizons: List of prediction horizons
                - anomaly_threshold: Threshold for anomaly detection
                - pattern_significance: Minimum pattern significance
                - seasonal_periods: List of potential seasonal periods
        """
        self.config = config
        self.metrics_collector = MetricsCollector()
        self.history: deque = deque(maxlen=config.get('history_size', 10000))
        self.patterns: Dict[str, List[Pattern]] = {}
        self.anomalies: List[Anomaly] = []
        self.predictions: Dict[str, Dict[int, List[float]]] = {}
        self.scaler = StandardScaler()
        
        # Analysis results
        self.current_analysis = {
            'patterns': {},
            'anomalies': [],
            'predictions': {},
            'statistics': {}
        }
        
        # Threading
        self.analysis_lock = threading.Lock()
        self.running = True
        self.analysis_thread = threading.Thread(target=self._background_analysis)
        self.analysis_thread.daemon = True
        self.analysis_thread.start()

    def analyze_workload(self, history: List[Tuple[datetime, Metrics]], 
                        window_size: int) -> Dict[str, Any]:
        """
        Implementation of Algorithm 6: Advanced Workload Analysis and Prediction
        
        Args:
            history: List of timestamped metrics
            window_size: Analysis window size
        
        Returns:
            Dict containing analysis results
        """
        try:
            with self.analysis_lock:
                # Prepare data
                df = self._prepare_time_series_data(history)
                
                # Extract patterns
                patterns = self._detect_patterns(df)
                
                # Detect anomalies
                anomalies = self._detect_anomalies(df)
                
                # Generate predictions
                predictions = self._generate_predictions(df, patterns)
                
                # Calculate statistics
                statistics = self._calculate_statistics(df)
                
                # Update current analysis
                self.current_analysis = {
                    'patterns': patterns,
                    'anomalies': anomalies,
                    'predictions': predictions,
                    'statistics': statistics,
                    'timestamp': datetime.now()
                }
                
                return self.current_analysis
                
        except Exception as e:
            logger.error(f"Error in workload analysis: {str(e)}")
            return self.current_analysis

    def _prepare_time_series_data(self, 
                                history: List[Tuple[datetime, Metrics]]) -> pd.DataFrame:
        """Prepare time series data for analysis"""
        try:
            # Convert history to DataFrame
            data = []
            for timestamp, metrics in history:
                metrics_dict = asdict(metrics)
                metrics_dict['timestamp'] = timestamp
                data.append(metrics_dict)
            
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            
            # Resample to regular intervals
            df = df.resample('1min').mean().interpolate()
            
            # Add time-based features
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            df['is_business_hour'] = df['hour'].between(9, 17).astype(int)
            
            return df
            
        except Exception as e:
            logger.error(f"Error preparing time series data: {str(e)}")
            return pd.DataFrame()

    def _detect_patterns(self, df: pd.DataFrame) -> Dict[str, List[Pattern]]:
        """Detect workload patterns in the data"""
        patterns = {}
        
        try:
            for column in df.select_dtypes(include=[np.number]).columns:
                if column in ['hour', 'day_of_week', 'is_weekend', 'is_business_hour']:
                    continue
                
                series = df[column].dropna()
                if len(series) < 24:  # Minimum required data
                    continue
                
                patterns[column] = []
                
                # Detect daily patterns
                daily_pattern = self._detect_seasonal_pattern(series, 24)
                if daily_pattern:
                    patterns[column].append(daily_pattern)
                
                # Detect weekly patterns
                weekly_pattern = self._detect_seasonal_pattern(series, 24 * 7)
                if weekly_pattern:
                    patterns[column].append(weekly_pattern)
                
                # Detect trends
                trend_pattern = self._detect_trend_pattern(series)
                if trend_pattern:
                    patterns[column].append(trend_pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting patterns: {str(e)}")
            return patterns

    def _detect_seasonal_pattern(self, series: pd.Series, 
                               period: int) -> Optional[Pattern]:
        """Detect seasonal patterns with specified period"""
        try:
            if len(series) < period * 2:
                return None
            
            # Calculate autocorrelation
            acf = stats.acf(series, nlags=period)
            
            # Check if there's significant correlation at the period
            if abs(acf[period-1]) > self.config.get('pattern_significance', 0.3):
                return Pattern(
                    pattern_type='seasonal',
                    period=period,
                    strength=abs(acf[period-1])
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting seasonal pattern: {str(e)}")
            return None

    def _detect_trend_pattern(self, series: pd.Series) -> Optional[Pattern]:
        """Detect trend patterns in the data"""
        try:
            # Calculate trend using linear regression
            x = np.arange(len(series))
            slope, intercept, r_value, _, _ = stats.linregress(x, series)
            
            # Check if trend is significant
            if abs(r_value) > self.config.get('pattern_significance', 0.3):
                return Pattern(
                    pattern_type='trend',
                    period=0,  # No period for trends
                    strength=abs(r_value)
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting trend pattern: {str(e)}")
            return None

    def _detect_anomalies(self, df: pd.DataFrame) -> List[Anomaly]:
        """Detect anomalies in the data"""
        anomalies = []
        
        try:
            for column in df.select_dtypes(include=[np.number]).columns:
                if column in ['hour', 'day_of_week', 'is_weekend', 'is_business_hour']:
                    continue
                
                series = df[column].dropna()
                if len(series) < 30:  # Minimum required data
                    continue
                
                # Calculate rolling statistics
                rolling_mean = series.rolling(window=30).mean()
                rolling_std = series.rolling(window=30).std()
                
                # Z-score based anomaly detection
                z_scores = (series - rolling_mean) / rolling_std
                
                # Identify anomalies
                threshold = self.config.get('anomaly_threshold', 3)
                anomaly_indices = np.where(abs(z_scores) > threshold)[0]
                
                for idx in anomaly_indices:
                    anomalies.append(Anomaly(
                        metric=column,
                        timestamp=series.index[idx],
                        value=series.iloc[idx],
                        score=abs(z_scores.iloc[idx])
                    ))
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {str(e)}")
            return anomalies

    def _generate_predictions(self, df: pd.DataFrame, 
                            patterns: Dict[str, List[Pattern]]) -> Dict[str, Dict[int, List[float]]]:
        """Generate predictions for different horizons"""
        predictions = {}
        
        try:
            for column in df.select_dtypes(include=[np.number]).columns:
                if column in ['hour', 'day_of_week', 'is_weekend', 'is_business_hour']:
                    continue
                
                series = df[column].dropna()
                if len(series) < 24:  # Minimum required data
                    continue
                
                predictions[column] = {}
                
                # Get column patterns
                column_patterns = patterns.get(column, [])
                
                # Select appropriate model based on patterns
                if any(p.pattern_type == 'seasonal' for p in column_patterns):
                    # Use SARIMA for seasonal data
                    predictions[column] = self._predict_sarima(series)
                else:
                    # Use Holt-Winters for non-seasonal data
                    predictions[column] = self._predict_holt_winters(series)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")
            return predictions

    def _predict_sarima(self, series: pd.Series) -> Dict[int, List[float]]:
        """Generate predictions using SARIMA model"""
        predictions = {}
        
        try:
            # Fit SARIMA model
            model = SARIMAX(
                series,
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 24)  # Daily seasonality
            ).fit(disp=False)
            
            # Generate predictions for different horizons
            for horizon in self.config.get('prediction_horizons', [1, 6, 24]):
                forecast = model.forecast(horizon)
                predictions[horizon] = forecast.tolist()
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in SARIMA prediction: {str(e)}")
            return predictions

    def _predict_holt_winters(self, series: pd.Series) -> Dict[int, List[float]]:
        """Generate predictions using Holt-Winters model"""
        predictions = {}
        
        try:
            # Fit Holt-Winters model
            model = ExponentialSmoothing(
                series,
                trend='add',
                seasonal=None
            ).fit()
            
            # Generate predictions for different horizons
            for horizon in self.config.get('prediction_horizons', [1, 6, 24]):
                forecast = model.forecast(horizon)
                predictions[horizon] = forecast.tolist()
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in Holt-Winters prediction: {str(e)}")
            return predictions

    def _calculate_statistics(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculate statistical metrics for the workload"""
        statistics = {}
        
        try:
            for column in df.select_dtypes(include=[np.number]).columns:
                if column in ['hour', 'day_of_week', 'is_weekend', 'is_business_hour']:
                    continue
                
                series = df[column].dropna()
                if len(series) < 2:
                    continue
                
                statistics[column] = {
                    'mean': series.mean(),
                    'std': series.std(),
                    'min': series.min(),
                    'max': series.max(),
                    'median': series.median(),
                    'skew': series.skew(),
                    'kurtosis': series.kurtosis(),
                    'p95': np.percentile(series, 95),
                    'p99': np.percentile(series, 99)
                }
            
            return statistics
            
        except Exception as e:
            logger.error(f"Error calculating statistics: {str(e)}")
            return statistics

    def _background_analysis(self):
        """Background thread for periodic workload analysis"""
        while self.running:
            try:
                time.sleep(self.config.get('analysis_interval', 60))
                
                # Convert history to list of tuples
                history = list(self.history)
                
                # Perform analysis
                self.analyze_workload(
                    history,
                    self.config.get('window_size', 24)
                )
                
            except Exception as e:
                logger.error(f"Error in background analysis: {str(e)}")

    def add_metrics(self, metrics: Metrics):
        """Add new metrics to the history"""
        try:
            self.history.append((datetime.now(), metrics))
        except Exception as e:
            logger.error(f"Error adding metrics: {str(e)}")

    def get_current_analysis(self) -> Dict[str, Any]:
        """Get the most recent analysis results"""
        return self.current_analysis

    def cleanup(self):
        """Cleanup resources and stop background thread"""
        self.running = False
        if self.analysis_thread.is_alive():
            self.analysis_thread.join()

# Example usage
def example_usage():
    # Configuration
    config = {
        'history_size': 10000,
        'analysis_interval': 60,
        'window_size': 24,
        'prediction_horizons': [1, 6, 24],
        'anomaly_threshold': 3,
        'pattern_significance': 0.3,
        'seasonal_periods': [24, 24*7]  # Daily and weekly
    }

    # Initialize monitor
    monitor = WorkloadMonitor(config)

    # Generate sample metrics
    for i in range(100):
        metrics = Metrics(
            cpu_usage=0.5 + 0.2 * np.sin(i/12),  # Sinusoidal pattern
            memory_usage=0.6 + 0.1 * np.random.randn(),
            network_io=50 + 10 * np.random.randn(),
            disk_io=40 + 5 * np.random.randn(),
            request_count=100 + 20 * np.sin(i/24) + 10 * np.random.randn(),
            response_time=150 + 30 * np.random.randn(),
            error_rate=0.02 + 0.005 * np.random.randn()
        )
        monitor.add_metrics(metrics)
        time.sleep(0.1)  # Simulate time passing

    # Get analysis results
    analysis = monitor.get_current_analysis()
    print("Patterns:", analysis['patterns'])
    print("Anomalies:", len(analysis['anomalies']))
    print("Predictions available for:", list(analysis['predictions'].keys()))
    print("Statistics:", analysis['statistics'])

    # Cleanup
    monitor.cleanup()

if __name__ == "__main__":
    example_usage()