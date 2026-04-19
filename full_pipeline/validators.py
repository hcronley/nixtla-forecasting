"""
validators.py
Input validation and data quality checks for time series forecasting.

This module provides comprehensive validation functions to ensure data
quality before running forecasting models. It handles CSV validation,
time series structure checks, and provides clear error messages.

Features:
- CSV file format validation
- Time series structure verification
- Data length validation
- Missing value detection and handling
- Frequency consistency checking
- Data type validation
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any
from datetime import datetime


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class DataValidator:
    """
    Validates data for time series forecasting.

    Provides comprehensive checks to ensure data quality and structure
    before passing to forecasting models.
    """

    # Minimum observations required for forecasting
    MIN_OBSERVATIONS = 20
    # Minimum observations needed for train/test split
    MIN_TRAIN_TEST = 40
    # Maximum allowed missing value percentage
    MAX_MISSING_PCT = 10

    @staticmethod
    def validate_csv_structure(
        df: pd.DataFrame,
        expected_columns: Optional[list] = None
    ) -> Tuple[bool, str]:
        """
        Validate CSV structure and column existence.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to validate
        expected_columns : list, optional
            List of required column names

        Returns
        -------
        tuple
            (is_valid, error_message)
        """
        try:
            # Check if empty
            if df.empty:
                return False, "❌ CSV file is empty"

            # Check column count
            if len(df.columns) < 2:
                return False, "❌ CSV must have at least 2 columns (date and value)"

            # Check for expected columns
            if expected_columns:
                missing = [col for col in expected_columns if col not in df.columns]
                if missing:
                    return False, f"❌ Missing required columns: {', '.join(missing)}"

            return True, "✅ CSV structure is valid"

        except Exception as e:
            return False, f"❌ Error validating CSV: {str(e)}"

    @staticmethod
    def validate_date_column(
        df: pd.DataFrame,
        date_col: str
    ) -> Tuple[bool, str]:
        """
        Validate and convert date column to datetime.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the date column
        date_col : str
            Name of the date column

        Returns
        -------
        tuple
            (is_valid, error_message)
        """
        try:
            if date_col not in df.columns:
                return False, f"❌ Date column '{date_col}' not found"

            # Try to convert to datetime
            test_convert = pd.to_datetime(df[date_col], errors='coerce')

            # Check for conversion failures
            failed_conversions = test_convert.isna().sum()
            if failed_conversions > 0:
                pct = (failed_conversions / len(df)) * 100
                return False, f"❌ {failed_conversions} dates could not be parsed ({pct:.1f}%)"

            # Check if all dates are unique
            if test_convert.nunique() < len(test_convert):
                return False, "❌ Date column contains duplicate dates"

            # Check if dates are sorted
            if not test_convert.is_monotonic_increasing:
                return False, "❌ Dates are not in ascending order"

            return True, "✅ Date column is valid"

        except Exception as e:
            return False, f"❌ Error validating date column: {str(e)}"

    @staticmethod
    def validate_value_column(
        df: pd.DataFrame,
        value_col: str
    ) -> Tuple[bool, str]:
        """
        Validate and convert value column to numeric.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the value column
        value_col : str
            Name of the value column

        Returns
        -------
        tuple
            (is_valid, error_message)
        """
        try:
            if value_col not in df.columns:
                return False, f"❌ Value column '{value_col}' not found"

            # Try to convert to numeric
            test_convert = pd.to_numeric(df[value_col], errors='coerce')

            # Check for conversion failures
            failed_conversions = test_convert.isna().sum()
            if failed_conversions > 0:
                pct = (failed_conversions / len(df)) * 100
                if pct > DataValidator.MAX_MISSING_PCT:
                    return False, f"❌ {failed_conversions} values are non-numeric ({pct:.1f}%)"

            # Check for all NaN
            if test_convert.isna().all():
                return False, "❌ Value column contains only missing values"

            # Check for infinite values
            if np.isinf(test_convert).any():
                return False, "❌ Value column contains infinite values"

            return True, "✅ Value column is valid"

        except Exception as e:
            return False, f"❌ Error validating value column: {str(e)}"

    @staticmethod
    def validate_data_length(
        df: pd.DataFrame,
        min_observations: int = MIN_OBSERVATIONS
    ) -> Tuple[bool, str]:
        """
        Validate that data has sufficient observations.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to validate
        min_observations : int
            Minimum required observations

        Returns
        -------
        tuple
            (is_valid, error_message)
        """
        try:
            n_obs = len(df)

            if n_obs < min_observations:
                return False, (
                    f"❌ Insufficient data: {n_obs} observations, "
                    f"but need at least {min_observations}"
                )

            if n_obs < DataValidator.MIN_TRAIN_TEST:
                return False, (
                    f"⚠️  Limited data: {n_obs} observations. "
                    f"Recommend at least {DataValidator.MIN_TRAIN_TEST} for train/test split"
                )

            return True, f"✅ Data length is valid ({n_obs} observations)"

        except Exception as e:
            return False, f"❌ Error validating data length: {str(e)}"

    @staticmethod
    def validate_missing_values(
        df: pd.DataFrame,
        value_col: str,
        max_missing_pct: float = MAX_MISSING_PCT
    ) -> Tuple[bool, str]:
        """
        Check for missing values in value column.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to validate
        value_col : str
            Name of the value column
        max_missing_pct : float
            Maximum allowed percentage of missing values

        Returns
        -------
        tuple
            (is_valid, error_message)
        """
        try:
            if value_col not in df.columns:
                return False, f"❌ Column '{value_col}' not found"

            n_missing = df[value_col].isna().sum()
            pct_missing = (n_missing / len(df)) * 100

            if pct_missing > max_missing_pct:
                return False, (
                    f"❌ Too many missing values: {n_missing} ({pct_missing:.1f}%) "
                    f"exceeds {max_missing_pct}% threshold"
                )

            if n_missing > 0:
                return True, (
                    f"⚠️  {n_missing} missing values found ({pct_missing:.1f}%) "
                    f"- will be removed during preprocessing"
                )

            return True, "✅ No missing values"

        except Exception as e:
            return False, f"❌ Error checking missing values: {str(e)}"

    @staticmethod
    def validate_time_series_structure(
        df: pd.DataFrame,
        date_col: str,
        value_col: str
    ) -> Tuple[bool, str]:
        """
        Comprehensive validation of time series structure.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to validate
        date_col : str
            Name of the date column
        value_col : str
            Name of the value column

        Returns
        -------
        tuple
            (is_valid, error_message)
        """
        checks = [
            DataValidator.validate_csv_structure(df),
            DataValidator.validate_date_column(df, date_col),
            DataValidator.validate_value_column(df, value_col),
            DataValidator.validate_data_length(df),
            DataValidator.validate_missing_values(df, value_col)
        ]

        # Collect all messages
        messages = [msg for _, msg in checks]
        errors = [msg for is_valid, msg in checks if not is_valid]

        # Return based on whether there are critical errors
        has_critical_error = any(msg.startswith('❌') for msg in messages)
        status = not has_critical_error
        combined_msg = '\n'.join(messages)

        return status, combined_msg

    @staticmethod
    def detect_frequency(
        dates: pd.Series
    ) -> Tuple[str, Optional[str]]:
        """
        Attempt to detect the frequency of a time series.

        Parameters
        ----------
        dates : pd.Series
            Series of datetime values

        Returns
        -------
        tuple
            (detected_frequency, frequency_name)
            Examples: ('D', 'Daily'), ('W', 'Weekly'), ('MS', 'Monthly')
        """
        try:
            dates = pd.to_datetime(dates).sort_values()

            if len(dates) < 3:
                return None, "Insufficient data to detect frequency"

            # Calculate differences between consecutive dates
            diffs = dates.diff().dropna()

            # Get most common difference
            most_common_diff = diffs.mode()[0]

            # Map to frequency
            frequency_map = {
                'D': ('Daily', 'D'),
                'H': ('Hourly', 'H'),
                'W': ('Weekly', 'W'),
                'MS': ('Monthly (Start)', 'MS'),
                'M': ('Monthly (End)', 'M'),
                'QS': ('Quarterly (Start)', 'QS'),
                'Q': ('Quarterly (End)', 'Q'),
                'AS': ('Yearly (Start)', 'YS'),
                'Y': ('Yearly (End)', 'Y')
            }

            # Try to infer from difference
            days = most_common_diff.days

            if days == 1:
                return 'D', 'Daily'
            elif days == 7:
                return 'W', 'Weekly'
            elif 28 <= days <= 31:
                return 'MS', 'Monthly (Start)'
            elif 89 <= days <= 92:
                return 'Q', 'Quarterly (End)'
            elif 364 <= days <= 366:
                return 'Y', 'Yearly (End)'
            else:
                return None, f"Uncommon frequency (interval: {days} days)"

        except Exception as e:
            return None, f"Could not detect frequency: {str(e)}"

    @staticmethod
    def validate_forecast_parameters(
        train_size: int,
        test_size: int,
        horizon: int,
        input_size: int = 12
    ) -> Tuple[bool, str]:
        """
        Validate forecast parameters for consistency.

        Parameters
        ----------
        train_size : int
            Training set size
        test_size : int
            Test set size
        horizon : int
            Forecast horizon
        input_size : int
            Input window size for neural models

        Returns
        -------
        tuple
            (is_valid, error_message)
        """
        try:
            messages = []

            # Check horizon vs test size
            if horizon > test_size:
                messages.append(
                    f"⚠️  Horizon ({horizon}) > test set size ({test_size})"
                )

            # Check input size vs train size
            if input_size > train_size:
                messages.append(
                    f"⚠️  Input size ({input_size}) > training set size ({train_size})"
                )

            # Check minimum training data
            if train_size < 20:
                messages.append(
                    "❌ Training set too small (< 20 observations)"
                )
                return False, '\n'.join(messages)

            # Check minimum test data
            if test_size < 5:
                messages.append(
                    "⚠️  Test set very small (< 5 observations) - results may be unreliable"
                )

            if messages:
                return True, '\n'.join(messages)
            else:
                return True, "✅ Forecast parameters are valid"

        except Exception as e:
            return False, f"❌ Error validating parameters: {str(e)}"

    @staticmethod
    def check_data_quality(
        df: pd.DataFrame,
        value_col: str
    ) -> Dict[str, Any]:
        """
        Generate comprehensive data quality report.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to assess
        value_col : str
            Name of the value column

        Returns
        -------
        dict
            Data quality metrics
        """
        try:
            values = df[value_col].dropna()

            report = {
                'total_rows': len(df),
                'valid_values': len(values),
                'missing_count': df[value_col].isna().sum(),
                'missing_pct': (df[value_col].isna().sum() / len(df) * 100),
                'min_value': float(values.min()),
                'max_value': float(values.max()),
                'mean_value': float(values.mean()),
                'std_value': float(values.std()),
                'has_outliers': bool((np.abs(values - values.mean()) > 3 * values.std()).any()),
                'is_stationary': None,  # Would require additional testing
            }

            return report

        except Exception as e:
            return {'error': str(e)}


def validate_upload_file(uploaded_file) -> Tuple[bool, str, Optional[pd.DataFrame]]:
    """
    Validate uploaded CSV file and return parsed data.

    Parameters
    ----------
    uploaded_file
        Streamlit uploaded file object

    Returns
    -------
    tuple
        (is_valid, message, dataframe)
    """
    try:
        # Try to read CSV
        df = pd.read_csv(uploaded_file)

        # Validate structure
        is_valid, msg = DataValidator.validate_csv_structure(df)
        if not is_valid:
            return False, msg, None

        return True, "✅ File loaded successfully", df

    except pd.errors.ParserError:
        return False, "❌ Invalid CSV format - parsing failed", None
    except Exception as e:
        return False, f"❌ Error reading file: {str(e)}", None


def validate_and_prepare_data(
    df: pd.DataFrame,
    date_col: str,
    value_col: str
) -> Tuple[bool, str, Optional[pd.DataFrame]]:
    """
    Validate and prepare data for forecasting.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    date_col : str
        Name of date column
    value_col : str
        Name of value column

    Returns
    -------
    tuple
        (is_valid, message, prepared_dataframe)
    """
    try:
        # Validate time series structure
        is_valid, validation_msg = DataValidator.validate_time_series_structure(
            df, date_col, value_col
        )

        if not is_valid and any(msg.startswith('❌') for msg in validation_msg.split('\n')):
            return False, validation_msg, None

        # Prepare data
        prepared_df = pd.DataFrame({
            'ds': pd.to_datetime(df[date_col]),
            'y': pd.to_numeric(df[value_col])
        })

        # Remove rows with missing values
        n_before = len(prepared_df)
        prepared_df = prepared_df.dropna()
        n_after = len(prepared_df)

        if n_before > n_after:
            msg = validation_msg + f"\n⚠️  Removed {n_before - n_after} rows with missing values"
        else:
            msg = validation_msg

        # Sort by date
        prepared_df = prepared_df.sort_values('ds').reset_index(drop=True)

        return True, msg, prepared_df

    except Exception as e:
        return False, f"❌ Error preparing data: {str(e)}", None


def get_validation_warnings(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    train_size: int,
    test_size: int,
    horizon: int
) -> list:
    """
    Get list of validation warnings (non-critical issues).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate
    date_col : str
        Date column name
    value_col : str
        Value column name
    train_size : int
        Training set size
    test_size : int
        Test set size
    horizon : int
        Forecast horizon

    Returns
    -------
    list
        List of warning messages
    """
    warnings = []

    # Check data quality
    quality = DataValidator.check_data_quality(df, value_col)
    if quality.get('has_outliers'):
        warnings.append("⚠️  Data contains outliers (>3 std dev from mean)")

    # Check frequency detectability
    freq, freq_name = DataValidator.detect_frequency(df[date_col])
    if freq is None:
        warnings.append(f"⚠️  Could not auto-detect frequency: {freq_name}")

    # Check parameter consistency
    _, param_msg = DataValidator.validate_forecast_parameters(
        train_size, test_size, horizon
    )
    if param_msg and '⚠️' in param_msg:
        warnings.extend([w.strip() for w in param_msg.split('\n') if w.startswith('⚠️')])

    return warnings
