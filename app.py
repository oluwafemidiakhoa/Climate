# ========================================
# Climate Data Analysis and Scenario Generation
# ========================================

# -------------------------------
# Environment Setup
# -------------------------------
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import csv
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, Lambda, BatchNormalization
from tensorflow.keras.models import Model
from scipy.stats import ks_2samp, wasserstein_distance
from scipy.spatial.distance import jensenshannon
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s',
                    handlers=[logging.StreamHandler()])

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -------------------------------
# Configuration Dictionary
# -------------------------------

CONFIG = {
    'data_path': '/content/GlobalLandTemperatures_GlobalLandTemperaturesByCountry.csv',  # Update with your path
    'vae': {
        'latent_dim': 5,
        'epochs': 50,
        'batch_size': 16
    },
    'gan': {
        'latent_dim_gan': 100,
        'epochs': 500,
        'batch_size': 32
    },
    'arima': {
        'order': (1, 1, 1),
        'forecast_steps': 50
    },
    'clustering': {
        'num_clusters': 5
    },
    'scenarios': {
        'num_scenarios': 5
    }
}

# -------------------------------
# Helper Functions
# -------------------------------

def detect_delimiter(file_path, num_lines=5):
    """
    Detects the delimiter used in a CSV file using csv.Sniffer.

    Parameters:
        file_path (str): Path to the CSV file.
        num_lines (int): Number of lines to read for detection.

    Returns:
        str: Detected delimiter.
    """
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        sample = ''.join([csvfile.readline() for _ in range(num_lines)])
        sniffer = csv.Sniffer()
        try:
            dialect = sniffer.sniff(sample, delimiters=[',', ';', '\t', '|'])
            delimiter = dialect.delimiter
            logging.info(f"Detected delimiter: '{delimiter}'")
            return delimiter
        except csv.Error:
            logging.warning("Could not detect delimiter, defaulting to comma ','")
            return ','

def load_data(file_path):
    """
    Loads the climate dataset from the specified CSV file with delimiter detection.

    Parameters:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    delimiter = detect_delimiter(file_path)
    try:
        data = pd.read_csv(
            file_path,
            sep=delimiter,
            encoding='utf-8',
            quotechar='"',
            engine='python',
            on_bad_lines='skip',  # Skips lines with too many fields
            usecols=['dt', 'AverageTemperature', 'AverageTemperatureUncertainty', 'Country']
        )
        logging.info(f"Dataset loaded successfully with delimiter '{delimiter}'.")
        return data
    except pd.errors.ParserError as e:
        logging.error(f"ParserError: {e}")
        raise
    except FileNotFoundError:
        logging.error(f"File not found. Please check the file path: {file_path}")
        raise
    except Exception as e:
        logging.error(f"An error occurred while loading the data: {e}")
        raise

def identify_malformed_rows(file_path, delimiter=','):
    """
    Identifies rows with an unexpected number of delimiters.

    Parameters:
        file_path (str): Path to the CSV file.
        delimiter (str): Delimiter used in the CSV file.

    Returns:
        list: List of line numbers that are malformed.
    """
    malformed_lines = []
    expected_num_delimiters = 3  # Since there are 4 columns

    with open(file_path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file, start=1):
            delimiter_count = line.count(delimiter)
            if delimiter_count != expected_num_delimiters:
                malformed_lines.append(i)

    if malformed_lines:
        logging.warning(f"Found {len(malformed_lines)} malformed lines: {malformed_lines[:5]}{'...' if len(malformed_lines) > 5 else ''}")
    else:
        logging.info("No malformed lines detected.")

    return malformed_lines

def inspect_data(data, num_rows=5):
    """
    Inspects the first few rows and data types of the DataFrame.

    Parameters:
        data (pd.DataFrame): DataFrame to inspect.
        num_rows (int): Number of rows to display.

    Returns:
        None
    """
    logging.info(f"DataFrame Head (First {num_rows} Rows):\n{data.head(num_rows)}")
    logging.info(f"DataFrame Dtypes:\n{data.dtypes}")

def clean_country_column(data, max_length=50):
    """
    Cleans the 'Country' column by removing rows where 'Country' contains concatenated names.

    Parameters:
        data (pd.DataFrame): DataFrame with 'Country' column.
        max_length (int): Maximum allowed length for country names.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    before_count = data.shape[0]
    data = data[data['Country'].str.len() <= max_length]
    after_count = data.shape[0]
    removed_rows = before_count - after_count
    logging.info(f"Removed {removed_rows} rows with excessively long 'Country' names.")
    return data

def verify_country_column(data, max_length=50):
    """
    Verifies that the 'Country' column contains individual country names.

    Parameters:
        data (pd.DataFrame): Loaded dataset.
        max_length (int): Maximum allowed length for country names.

    Raises:
        ValueError: If any country name exceeds the maximum length.

    Returns:
        None
    """
    unique_countries = data['Country'].unique()
    num_unique = len(unique_countries)
    logging.info(f"Number of unique countries: {num_unique}")
    logging.info(f"Sample of unique countries: {unique_countries[:10]}")

    # Check for excessively long country names
    long_countries = [country for country in unique_countries if isinstance(country, str) and len(country) > max_length]
    if long_countries:
        logging.error(f"Found {len(long_countries)} countries with names longer than {max_length} characters.")
        raise ValueError("Invalid 'Country' column: Contains concatenated country names.")
    else:
        logging.info("The 'Country' column appears to be correctly parsed.")

def preprocess_data(data):
    """
    Handles missing values, cleans 'Country' column, extracts date features, and performs feature engineering.

    Parameters:
        data (pd.DataFrame): Raw dataset.

    Returns:
        pd.DataFrame: Preprocessed dataset.
    """
    # Ensure critical columns exist
    required_columns = ['dt', 'AverageTemperature', 'AverageTemperatureUncertainty', 'Country']
    for col in required_columns:
        if col not in data.columns:
            logging.error(f"Missing required column: '{col}'")
            raise ValueError(f"Missing required column: '{col}'")

    # Handle Missing Values
    data['AverageTemperature'] = pd.to_numeric(data['AverageTemperature'], errors='coerce')
    data['AverageTemperatureUncertainty'] = pd.to_numeric(data['AverageTemperatureUncertainty'], errors='coerce')

    # Interpolate missing values
    data['AverageTemperature'] = data['AverageTemperature'].interpolate(method='linear', limit_direction='forward')
    data['AverageTemperatureUncertainty'] = data['AverageTemperatureUncertainty'].interpolate(method='linear', limit_direction='forward')
    logging.info("Missing values handled through interpolation.")

    # Verify missing values are handled
    missing_values = data.isnull().sum()
    if missing_values.any():
        logging.warning("There are still missing values in the dataset after interpolation.")
    else:
        logging.info("No missing values detected after interpolation.")

    # Clean 'Country' column to remove concatenated names
    data = clean_country_column(data, max_length=50)

    # Extract Date Features
    data['dt'] = pd.to_datetime(data['dt'], errors='coerce')  # Handle invalid dates
    data['Year'] = data['dt'].dt.year
    data['Month'] = data['dt'].dt.month
    data['Season'] = data['Month'].apply(get_season)
    logging.info("Date features extracted successfully.")

    # Feature Engineering
    data['RollingAvg_12M'] = data.groupby('Country')['AverageTemperature'].transform(
        lambda x: x.rolling(window=12, min_periods=1).mean()
    )
    data['YearlyAvgTemp'] = data.groupby(['Country', 'Year'])['AverageTemperature'].transform('mean')
    logging.info("Feature engineering completed: Rolling and Yearly Averages.")

    # Temperature Anomaly Calculation
    baseline_period = (1850, 1900)
    baseline_avg = calculate_baseline(data, baseline_period)
    data = data.merge(baseline_avg, on='Country', how='left')
    data['TempAnomaly'] = data['AverageTemperature'] - data['BaselineTemp']
    logging.info("Temperature anomaly calculated.")

    # Flag Sparse Data
    data = flag_sparse_data(data, threshold=50)
    logging.info("Sparse data flagged based on record count per country.")

    # Debugging: Print data head and dtypes
    inspect_data(data)
    logging.info(f"Total records after preprocessing: {data.shape[0]}")
    logging.info(f"Total columns after preprocessing: {data.shape[1]}")

    return data

def get_season(month):
    """
    Determines the season based on the month.

    Parameters:
        month (int): Month as an integer.

    Returns:
        str: Season name.
    """
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Autumn'

def calculate_baseline(data, period):
    """
    Calculates the baseline average temperature for each country over a specified period.

    Parameters:
        data (pd.DataFrame): Preprocessed dataset.
        period (tuple): Start and end year for the baseline period.

    Returns:
        pd.DataFrame: Baseline average temperatures per country.
    """
    baseline_data = data[(data['Year'] >= period[0]) & (data['Year'] <= period[1])]
    baseline_avg = baseline_data.groupby('Country')['AverageTemperature'].mean().reset_index()
    baseline_avg.rename(columns={'AverageTemperature': 'BaselineTemp'}, inplace=True)
    return baseline_avg

def flag_sparse_data(data, threshold=50):
    """
    Flags countries with a number of records below the specified threshold.

    Parameters:
        data (pd.DataFrame): Preprocessed dataset.
        threshold (int): Minimum number of records required.

    Returns:
        pd.DataFrame: Dataset with an additional 'SparseDataFlag' column.
    """
    country_counts = data['Country'].value_counts()
    sparse_countries = country_counts[country_counts < threshold].index
    data['SparseDataFlag'] = data['Country'].isin(sparse_countries)
    return data

def prepare_model_data(pivot_data):
    """
    Prepares data for modeling by excluding non-numeric columns.

    Parameters:
        pivot_data (pd.DataFrame): Pivoted temperature data with 'Country' and 'Cluster'.

    Returns:
        pd.DataFrame: DataFrame containing only numeric features.
    """
    # Exclude 'Cluster' and 'Country' columns
    if 'Country' in pivot_data.columns:
        numeric_data = pivot_data.drop(['Cluster', 'Country'], axis=1, errors='ignore')
    else:
        numeric_data = pivot_data.drop(['Cluster'], axis=1, errors='ignore')
    # Ensure all remaining columns are numeric
    numeric_data = numeric_data.apply(pd.to_numeric, errors='coerce').fillna(0)
    return numeric_data

def ensure_columns_are_strings(df):
    """
    Ensures that all column names in the DataFrame are strings.

    Parameters:
        df (pd.DataFrame): DataFrame to process.

    Returns:
        pd.DataFrame: DataFrame with string column names.
    """
    df.columns = df.columns.astype(str)
    return df

def check_column_types(df):
    """
    Checks if all column names in the DataFrame are of type string.

    Parameters:
        df (pd.DataFrame): DataFrame to check.

    Raises:
        TypeError: If any column name is not a string.
    """
    if not all(isinstance(col, str) for col in df.columns):
        df.columns = df.columns.astype(str)
        logging.warning("Converted all column names to strings to ensure compatibility with scikit-learn models.")

def validate_data(df, expected_columns):
    """
    Validates that the DataFrame contains the expected columns with appropriate data types.

    Parameters:
        df (pd.DataFrame): DataFrame to validate.
        expected_columns (dict): Dictionary with column names as keys and expected dtypes as values or lists of dtypes.

    Raises:
        ValueError: If any column is missing or has an incorrect dtype.
    """
    for col, dtype in expected_columns.items():
        if col not in df.columns:
            raise ValueError(f"Missing expected column: '{col}'")
        if isinstance(dtype, list):
            if df[col].dtype not in dtype:
                raise ValueError(f"Column '{col}' has dtype {df[col].dtype}, expected one of {dtype}")
        else:
            if not pd.api.types.is_dtype_equal(df[col].dtype, dtype):
                raise ValueError(f"Column '{col}' has dtype {df[col].dtype}, expected {dtype}")
    logging.info("Data validation passed.")

def perform_kmeans_clustering(data, num_clusters):
    """
    Performs K-Means clustering on the provided data.

    Parameters:
        data (pd.DataFrame): Data to cluster.
        num_clusters (int): Number of clusters.

    Returns:
        pd.DataFrame: Data with cluster labels.
        KMeans: Fitted KMeans model.
    """
    kmeans = KMeans(n_clusters=num_clusters, random_state=SEED)
    data['Cluster'] = kmeans.fit_predict(data)
    return data, kmeans

def plot_global_temperature_trends(global_trends):
    """
    Plots the global average temperature over years.

    Parameters:
        global_trends (pd.DataFrame): DataFrame with 'Year' and 'AverageTemperature'.

    Returns:
        None
    """
    plt.figure(figsize=(12, 6))
    plt.plot(global_trends['Year'], global_trends['AverageTemperature'], marker='o')
    plt.title('Global Average Temperature Over Years')
    plt.xlabel('Year')
    plt.ylabel('Average Temperature (°C)')
    plt.grid(True)
    plt.show()

def plot_country_wise_comparisons(data, top_n=5):
    """
    Plots average temperature comparisons among top countries.

    Parameters:
        data (pd.DataFrame): Dataset with 'Country', 'Year', and 'AverageTemperature'.
        top_n (int): Number of top countries to compare.

    Returns:
        None
    """
    top_countries = data['Country'].value_counts().head(top_n).index
    plt.figure(figsize=(12, 6))
    for country in top_countries:
        country_data = data[data['Country'] == country]
        plt.plot(country_data['Year'], country_data['AverageTemperature'], label=country)
    plt.title('Average Temperature Comparison Among Top Countries')
    plt.xlabel('Year')
    plt.ylabel('Average Temperature (°C)')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_correlation_heatmap(data):
    """
    Plots a correlation heatmap of the numerical features in the dataset.

    Parameters:
        data (pd.DataFrame): DataFrame containing numerical features.

    Returns:
        None
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(data.corr(), annot=False, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()

def plot_significant_warming_trends(data, recent_years=100, top_n=10):
    """
    Plots countries with the highest average temperature in recent years.

    Parameters:
        data (pd.DataFrame): Dataset with 'Country', 'Year', and 'AverageTemperature'.
        recent_years (int): Number of recent years to consider.
        top_n (int): Number of top countries to display.

    Returns:
        None
    """
    recent_data = data[data['Year'] >= (data['Year'].max() - recent_years)]
    country_avg_temp = recent_data.groupby('Country')['AverageTemperature'].mean().sort_values(ascending=False).head(top_n)
    plt.figure(figsize=(12, 6))
    country_avg_temp.plot(kind='bar')
    plt.title(f'Top {top_n} Countries with Highest Average Temperature in Last {recent_years} Years')
    plt.xlabel('Country')
    plt.ylabel('Average Temperature (°C)')
    plt.show()

def plot_drastic_temperature_changes(data):
    """
    Plots countries with the most drastic temperature changes.

    Parameters:
        data (pd.DataFrame): Dataset with 'Country' and 'AverageTemperature'.

    Returns:
        None
    """
    temp_changes = data.groupby('Country')['AverageTemperature'].agg(['min', 'max'])
    temp_changes['change'] = temp_changes['max'] - temp_changes['min']
    top_changes = temp_changes.sort_values('change', ascending=False).head(10)
    plt.figure(figsize=(12, 6))
    top_changes['change'].plot(kind='bar')
    plt.title('Countries with Most Drastic Temperature Changes')
    plt.xlabel('Country')
    plt.ylabel('Temperature Change (°C)')
    plt.show()

def arima_forecasting(series, order, forecast_steps):
    """
    Performs ARIMA forecasting on a time series.

    Parameters:
        series (pd.Series): Time series data.
        order (tuple): ARIMA order parameters.
        forecast_steps (int): Number of steps to forecast.

    Returns:
        np.ndarray: Forecasted values.
        pd.DataFrame: Confidence intervals.
    """
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    forecast_result = model_fit.get_forecast(steps=forecast_steps)
    forecast_values = forecast_result.predicted_mean
    forecast_ci = forecast_result.conf_int()
    return forecast_values, forecast_ci

def plot_arima_forecast(series, forecast_values, forecast_ci, forecast_index):
    """
    Plots the ARIMA forecast along with confidence intervals.

    Parameters:
        series (pd.Series): Historical time series data.
        forecast_values (np.ndarray): Forecasted values.
        forecast_ci (pd.DataFrame): Confidence intervals.
        forecast_index (range): Index for forecasted values.

    Returns:
        None
    """
    plt.figure(figsize=(12, 6))
    plt.plot(series.index, series, label='Historical')
    plt.plot(forecast_index, forecast_values, label='Forecast', color='red')
    plt.fill_between(forecast_index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='pink', alpha=0.3)
    plt.title('ARIMA Forecast')
    plt.xlabel('Year')
    plt.ylabel('Average Temperature (°C)')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_cluster_centers(cluster_summary):
    """
    Plots the cluster centers obtained from K-Means clustering.

    Parameters:
        cluster_summary (pd.DataFrame): DataFrame containing cluster centers.

    Returns:
        None
    """
    plt.figure(figsize=(12, 6))
    for index, row in cluster_summary.iterrows():
        plt.plot(row[1:], label=f'Cluster {int(row["Cluster"])}')
    plt.title('Cluster Centers')
    plt.xlabel('Year Index')
    plt.ylabel('Average Temperature (°C)')
    plt.legend()
    plt.grid(True)
    plt.show()

def build_vae(input_dim, latent_dim):
    """
    Builds the Variational Autoencoder (VAE) model using a custom Model subclass.

    Parameters:
        input_dim (int): Number of input features.
        latent_dim (int): Dimensionality of the latent space.

    Returns:
        tuple: (vae, encoder, decoder) Keras models.
    """
    # Encoder
    inputs = Input(shape=(input_dim,), name='encoder_input')
    x = Dense(128, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    # Sampling function
    def sampling(args):
        z_mean, z_log_var = args
        batch_size = tf.shape(z_mean)[0]
        epsilon = tf.random.normal(shape=(batch_size, latent_dim), mean=0., stddev=1.0)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    # Latent space
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # Build encoder model
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

    # Decoder
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(64, activation='relu')(latent_inputs)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    outputs = Dense(input_dim, activation='linear')(x)

    # Build decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')

    # Custom VAE model
    class VAE(Model):
        def __init__(self, encoder, decoder, **kwargs):
            super(VAE, self).__init__(**kwargs)
            self.encoder = encoder
            self.decoder = decoder

        def train_step(self, data):
            if isinstance(data, tuple):
                data = data[0]
            with tf.GradientTape() as tape:
                z_mean, z_log_var, z = self.encoder(data)
                reconstruction = self.decoder(z)
                # Compute reconstruction loss
                reconstruction_loss = tf.reduce_mean(tf.square(data - reconstruction))
                # Compute KL divergence
                kl_loss = -0.5 * tf.reduce_mean(
                    1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
                )
                total_loss = reconstruction_loss + kl_loss
            # Compute gradients
            grads = tape.gradient(total_loss, self.trainable_variables)
            # Update weights
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
            return {
                "loss": total_loss,
                "reconstruction_loss": reconstruction_loss,
                "kl_loss": kl_loss
            }

    # Instantiate and compile the VAE model
    vae = VAE(encoder, decoder)
    vae.compile(optimizer='adam')

    return vae, encoder, decoder

def build_gan_models(latent_dim_gan, input_dim_gan):
    """
    Builds and compiles the Generator, Discriminator, and GAN models.

    Parameters:
        latent_dim_gan (int): Dimensionality of the latent space for the GAN's generator.
        input_dim_gan (int): Number of features in the input data.

    Returns:
        tuple: (generator_gan, discriminator_gan, gan_model)
    """
    # Build the Generator
    generator_gan = build_generator_gan(latent_dim_gan, input_dim_gan)

    # Build the Discriminator
    discriminator_gan = build_discriminator_gan(input_dim_gan)

    # Compile the Discriminator
    discriminator_gan.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
                              loss='binary_crossentropy',
                              metrics=['accuracy'])

    # Build and compile the GAN
    discriminator_gan.trainable = False  # Freeze Discriminator during GAN training
    gan_input = Input(shape=(latent_dim_gan,), name='gan_input')
    gan_output = discriminator_gan(generator_gan(gan_input))
    gan_model = Model(gan_input, gan_output, name='GAN')
    gan_model.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
                      loss='binary_crossentropy')

    return generator_gan, discriminator_gan, gan_model


def build_generator_gan(latent_dim_gan, input_dim_gan):
    """
    Builds the Generator model for the GAN.

    Parameters:
        latent_dim_gan (int): Dimensionality of the latent space for the GAN's generator.
        input_dim_gan (int): Number of features in the input data.

    Returns:
        keras.Model: Generator model.
    """
    model = tf.keras.Sequential(name="Generator")
    model.add(Dense(256, input_dim=latent_dim_gan))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())
    model.add(Dense(input_dim_gan, activation='tanh'))  # Use 'tanh' activation for output layer
    return model

def build_discriminator_gan(input_dim_gan):
    """
    Builds the Discriminator model for the GAN.

    Parameters:
        input_dim_gan (int): Number of features in the input data.

    Returns:
        keras.Model: Discriminator model.
    """
    model = tf.keras.Sequential(name="Discriminator")
    model.add(Dense(512, input_dim=input_dim_gan))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))  # Output layer with sigmoid activation
    return model




def train_gan(generator, discriminator, gan_model, normalized_data, config):
    """
    Trains the Generative Adversarial Network (GAN).

    Parameters:
        generator (Model): GAN generator model.
        discriminator (Model): GAN discriminator model.
        gan_model (Model): Combined GAN model.
        normalized_data (np.ndarray): Normalized real data.
        config (dict): Configuration dictionary.

    Returns:
        None
    """
    epochs = config['gan']['epochs']
    batch_size = config['gan']['batch_size']
    latent_dim_gan = config['gan']['latent_dim_gan']
    steps_per_epoch = normalized_data.shape[0] // batch_size

    logging.info(f"Starting GAN training for {epochs} epochs with batch size {batch_size}.")

    for epoch in range(1, epochs + 1):
        d_loss_total = []
        d_acc_total = []
        g_loss_total = []

        for step in range(steps_per_epoch):
            # Train Discriminator
            idx = np.random.randint(0, normalized_data.shape[0], batch_size)
            real_samples = normalized_data[idx]

            noise = np.random.normal(0, 1, (batch_size, latent_dim_gan)).astype(np.float32)
            fake_samples = generator.predict(noise)

            real_labels = np.ones((batch_size, 1))
            fake_labels = np.zeros((batch_size, 1))

            d_loss_real, d_acc_real = discriminator.train_on_batch(real_samples, real_labels)
            d_loss_fake, d_acc_fake = discriminator.train_on_batch(fake_samples, fake_labels)

            d_loss = 0.5 * (d_loss_real + d_loss_fake)
            d_acc = 0.5 * (d_acc_real + d_acc_fake)

            d_loss_total.append(d_loss)
            d_acc_total.append(d_acc)

            # Train Generator
            noise = np.random.normal(0, 1, (batch_size, latent_dim_gan)).astype(np.float32)
            valid_y = np.ones((batch_size, 1))
            g_loss = gan_model.train_on_batch(noise, valid_y)
            g_loss_total.append(g_loss)

        # Log the metrics
        if epoch % 10 == 0 or epoch == 1:
            d_loss_avg = np.mean(d_loss_total)
            d_acc_avg = np.mean(d_acc_total)
            g_loss_avg = np.mean(g_loss_total)
            logging.info(f"Epoch {epoch}/{epochs} [D loss: {d_loss_avg:.4f}, acc.: {100 * d_acc_avg:.2f}%] [G loss: {g_loss_avg:.4f}]")

def plot_vae_generated_scenarios(decoder, scaler_vae, latent_dim, input_dim, num_scenarios=10):
    """
    Generates and plots climate scenarios using the trained VAE decoder.

    Parameters:
        decoder (Model): Trained VAE decoder model.
        scaler_vae (MinMaxScaler): Fitted scaler for inverse transformation.
        latent_dim (int): Dimensionality of the latent space.
        input_dim (int): Number of features (years) in the data.
        num_scenarios (int): Number of scenarios to generate.

    Returns:
        None
    """
    logging.info("Generating VAE scenarios...")
    latent_samples = np.random.normal(size=(num_scenarios, latent_dim)).astype(np.float32)
    generated_trends = decoder.predict(latent_samples)
    generated_trends_rescaled = scaler_vae.inverse_transform(generated_trends)

    # Define the years based on input_dim
    years = range(1, input_dim + 1)

    plt.figure(figsize=(14, 7))
    for i, trend in enumerate(generated_trends_rescaled):
        plt.plot(years, trend, label=f'Generated Scenario {i+1}')
    plt.title('VAE Generated Climate Scenarios', fontsize=18)
    plt.xlabel('Year Index', fontsize=14)
    plt.ylabel('Average Temperature (°C)', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    logging.info("VAE scenarios generated and plotted successfully.")

def plot_gan_generated_trends(generator_gan, scaler_vae, latent_dim_gan, input_dim, num_trends=10):
    """
    Generates and plots climate trends using the trained GAN generator.

    Parameters:
        generator_gan (Model): Trained GAN generator model.
        scaler_vae (MinMaxScaler): Fitted scaler for inverse transformation.
        latent_dim_gan (int): Dimensionality of the GAN's latent space.
        input_dim (int): Number of features (years) in the data.
        num_trends (int): Number of trends to generate.

    Returns:
        None
    """
    logging.info("Generating GAN trends...")
    noise = np.random.normal(0, 1, (num_trends, latent_dim_gan)).astype(np.float32)
    generated_trends = generator_gan.predict(noise)
    generated_trends_rescaled = scaler_vae.inverse_transform(generated_trends)

    # Define the years based on input_dim
    years = range(1, input_dim + 1)

    plt.figure(figsize=(14, 7))
    for i, trend in enumerate(generated_trends_rescaled):
        plt.plot(years, trend, label=f'GAN Generated Trend {i+1}')
    plt.title('GAN Generated Climate Trends', fontsize=18)
    plt.xlabel('Year Index', fontsize=14)
    plt.ylabel('Average Temperature (°C)', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    logging.info("GAN trends generated and plotted successfully.")

def perform_statistical_tests(real_sample, generated_sample):
    """
    Performs statistical tests between real and generated samples.

    Parameters:
        real_sample (array-like): Real data samples.
        generated_sample (array-like): Generated data samples.

    Returns:
        pd.DataFrame: Results of statistical tests.
    """
    logging.info("Performing statistical tests between real and generated samples...")
    ks_stat, ks_p = ks_2samp(real_sample, generated_sample)
    wd = wasserstein_distance(real_sample, generated_sample)
    jsd = jensenshannon(real_sample, generated_sample)

    results = pd.DataFrame({
        'Test': ['Kolmogorov-Smirnov', 'Wasserstein Distance', 'Jensen-Shannon Divergence'],
        'Statistic': [ks_stat, wd, jsd]
    })

    logging.info(f"Statistical Test Results:\n{results}")
    return results

def plot_statistical_comparison(real_sample, generated_sample):
    """
    Plots histograms to compare real and generated samples.

    Parameters:
        real_sample (array-like): Real data samples.
        generated_sample (array-like): Generated data samples.

    Returns:
        None
    """
    logging.info("Plotting statistical comparison between real and generated samples...")
    plt.figure(figsize=(14, 7))
    sns.histplot(real_sample, color='blue', label='Real Data', kde=True, stat='density', linewidth=0)
    sns.histplot(generated_sample, color='red', label='Generated Data', kde=True, stat='density', linewidth=0, alpha=0.6)
    plt.title('Statistical Comparison of Real vs Generated Data', fontsize=18)
    plt.xlabel('Average Temperature (°C)', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    logging.info("Statistical comparison plot generated successfully.")

def generate_future_scenarios(decoder, scaler_vae, latent_dim, input_dim, num_scenarios=10):
    """
    Generates future climate scenarios using the VAE decoder.

    Parameters:
        decoder (Model): Trained VAE decoder model.
        scaler_vae (MinMaxScaler): Fitted scaler for inverse transformation.
        latent_dim (int): Dimensionality of the latent space.
        input_dim (int): Number of features (years) in the data.
        num_scenarios (int): Number of future scenarios to generate.

    Returns:
        pd.DataFrame: Future scenarios data.
    """
    logging.info("Generating future climate scenarios...")
    latent_samples = np.random.normal(size=(num_scenarios, latent_dim)).astype(np.float32)
    generated_scenarios = decoder.predict(latent_samples)
    generated_scenarios_rescaled = scaler_vae.inverse_transform(generated_scenarios)

    # Create a DataFrame for scenarios
    scenario_columns = [f'Year_{i}' for i in range(1, input_dim + 1)]
    scenarios_df = pd.DataFrame(generated_scenarios_rescaled, columns=scenario_columns)

    logging.info(f"Future scenarios generated with shape {scenarios_df.shape}.")
    return scenarios_df

def plot_future_scenarios(scenarios_df, future_years):
    """
    Plots the future climate scenarios.

    Parameters:
        scenarios_df (pd.DataFrame): Future scenarios data.
        future_years (range): Range of future years.

    Returns:
        None
    """
    logging.info("Plotting future climate scenarios...")
    plt.figure(figsize=(14, 7))
    for index, row in scenarios_df.iterrows():
        plt.plot(future_years, row.values, label=f'Scenario {index + 1}')
    plt.title('Future Climate Scenarios', fontsize=18)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Average Temperature (°C)', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    logging.info("Future climate scenarios plotted successfully.")

def plot_time_series_comparison(real_series, generated_series):
    """
    Plots a comparison between real and generated time series data.

    Parameters:
        real_series (array-like): Real time series data.
        generated_series (array-like): Generated time series data.

    Returns:
        None
    """
    logging.info("Plotting time series comparison between real and generated data...")
    plt.figure(figsize=(14, 7))
    plt.plot(real_series, label='Real Data', color='blue')
    plt.plot(generated_series, label='Generated Data', color='red', alpha=0.7)
    plt.title('Time Series Comparison: Real vs Generated Data', fontsize=18)
    plt.xlabel('Time Index', fontsize=14)
    plt.ylabel('Average Temperature (°C)', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    logging.info("Time series comparison plot generated successfully.")

def save_models(encoder, decoder, generator_gan):
    """
    Saves the trained models to disk.

    Parameters:
        encoder (Model): Trained VAE encoder model.
        decoder (Model): Trained VAE decoder model.
        generator_gan (Model): Trained GAN generator model.

    Returns:
        None
    """
    logging.info("Saving trained models...")
    try:
        encoder.save('vae_encoder.h5')
        decoder.save('vae_decoder.h5')
        generator_gan.save('gan_generator.h5')
        logging.info("All models saved successfully.")
    except Exception as e:
        logging.error(f"Failed to save models: {e}")

def save_clustering_results(clustered_countries, filename='clustered_countries.csv'):
    """
    Saves the clustering results to a CSV file.

    Parameters:
        clustered_countries (pd.DataFrame): DataFrame containing clustering results.
        filename (str): Name of the CSV file to save.

    Returns:
        None
    """
    logging.info(f"Saving clustering results to {filename}...")
    try:
        clustered_countries.to_csv(filename, index=False)
        logging.info("Clustering results saved successfully.")
    except Exception as e:
        logging.error(f"Failed to save clustering results: {e}")

# -------------------------------
# Main Execution Flow
# -------------------------------

def main():
    # Step 1: Identify Malformed Rows
    file_path = CONFIG['data_path']
    delimiter = detect_delimiter(file_path)
    malformed_lines = identify_malformed_rows(file_path, delimiter=delimiter)

    # Step 2: Load Data Robustly
    data = load_data(file_path)

    # Step 3: Inspect Loaded Data
    inspect_data(data, num_rows=10)

    # Step 4: Verify 'Country' Column Integrity
    try:
        verify_country_column(data, max_length=50)
    except ValueError as ve:
        logging.error(f"Country column verification failed: {ve}")
        # Proceed after cleaning
        data = clean_country_column(data, max_length=50)
        try:
            verify_country_column(data, max_length=50)
        except ValueError as ve_inner:
            logging.error(f"Country column verification failed after cleaning: {ve_inner}")
            raise

    # Step 5: Data Preprocessing
    data = preprocess_data(data)

    # Step 6: Visualization - Global Trends
    global_trends = data.groupby('Year')['AverageTemperature'].mean().reset_index()
    plot_global_temperature_trends(global_trends)

    # Step 7: Visualization - Country-Wise Comparisons
    plot_country_wise_comparisons(data, top_n=5)

    # Step 8: Visualization - Correlation Heatmap
    numeric_data = data.select_dtypes(include=[np.number])
    plot_correlation_heatmap(numeric_data)

    # Step 9: Visualization - Significant Warming Trends
    plot_significant_warming_trends(data, recent_years=100, top_n=10)

    # Step 10: Visualization - Drastic Temperature Changes
    plot_drastic_temperature_changes(data)

    # Step 11: ARIMA Forecasting
    arima_order = CONFIG['arima']['order']
    forecast_steps = CONFIG['arima']['forecast_steps']
    try:
        series = global_trends.set_index('Year')['AverageTemperature']
        forecast_values, forecast_ci = arima_forecasting(series, arima_order, forecast_steps)
        forecast_index = range(series.index.max() + 1, series.index.max() + forecast_steps + 1)
        plot_arima_forecast(series, forecast_values, forecast_ci, forecast_index)
    except Exception as e:
        logging.error(f"ARIMA Forecasting could not be completed: {e}")

    # Step 12: Clustering - K-Means
    clustering_data = data.groupby(['Country', 'Year'])['AverageTemperature'].mean().reset_index()

    # Convert 'Year' column to int64 for consistency
    clustering_data['Year'] = pd.to_numeric(clustering_data['Year'], errors='coerce').astype('Int64')
    logging.info("Converted 'Year' column to Int64.")

    # Drop rows where 'Year' is NaN after conversion
    clustering_data = clustering_data.dropna(subset=['Year'])
    clustering_data['Year'] = clustering_data['Year'].astype(int)

    # Validate 'Year' and 'Country' columns
    expected_columns = {'Country': 'object', 'Year': ['int32', 'int64']}
    try:
        validate_data(clustering_data, expected_columns)
    except ValueError as ve:
        logging.error(f"Data validation error: {ve}")
        raise

    # Pivot data for clustering: Countries as rows, Years as columns
    pivot_data = clustering_data.pivot(index='Country', columns='Year', values='AverageTemperature').fillna(0)

    # Perform K-Means Clustering
    pivot_data, kmeans_model = perform_kmeans_clustering(pivot_data, CONFIG['clustering']['num_clusters'])

    # Ensure 'Cluster' column is numeric
    pivot_data['Cluster'] = pd.to_numeric(pivot_data['Cluster'], errors='coerce')

    # Drop rows with NaN in 'Cluster' if any
    pivot_data = pivot_data.dropna(subset=['Cluster'])

    # Convert 'Cluster' to integer
    pivot_data['Cluster'] = pivot_data['Cluster'].astype(int)

    # Generate Cluster Summary (only numeric columns)
    numeric_cols = pivot_data.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols.remove('Cluster')  # Exclude 'Cluster' from features if present
    cluster_summary = pivot_data.groupby('Cluster')[numeric_cols].mean().reset_index()

    # Plot Cluster Centers
    plot_cluster_centers(cluster_summary)

    # Create 'clustered_countries' to store the clustering results
    if 'Country' in pivot_data.index.names:
        pivot_data.reset_index(inplace=True)
    clustered_countries = pivot_data[['Country', 'Cluster']].copy()

    # Print Clustering Results
    print("Clustering Results: Country Groups")
    print(clustered_countries.head())

    # Rename columns for Plotly compatibility
    clustered_countries_map = clustered_countries.rename(columns={'Country': 'country', 'Cluster': 'cluster'})

    # Plot geographic visualization using Plotly
    try:
        fig = px.choropleth(
            clustered_countries_map,
            locations="country",
            locationmode="country names",
            color="cluster",
            title="Geographic Visualization of Temperature Clusters",
            color_continuous_scale="Viridis",
            labels={'cluster': 'Cluster'},
        )

        fig.update_layout(
            title_font_size=20,
            geo=dict(showframe=False, showcoastlines=True, projection_type='equirectangular'),
            coloraxis_colorbar=dict(title="Cluster"),
        )

        fig.show()
    except Exception as e:
        logging.error(f"Failed to create choropleth map: {e}")
        logging.error("Ensure that country names in the 'country' column are correctly spelled and recognized by Plotly.")
        raise

    # -------------------------------
    # Step 13: Variational Autoencoder (VAE)
    # -------------------------------

        # Prepare numeric data for VAE
    vae_features = prepare_model_data(pivot_data)

    # Log the number of features
    input_dim_vae = vae_features.shape[1]
    logging.info(f"Number of features for VAE: {input_dim_vae}")

    # Build the VAE with the custom Model subclass
    vae, encoder, decoder = build_vae(
        input_dim=input_dim_vae,
        latent_dim=CONFIG['vae']['latent_dim']
    )

    # Initialize the scaler and fit-transform
    scaler_vae = MinMaxScaler()
    normalized_vae_data = scaler_vae.fit_transform(vae_features).astype(np.float32)

    # Train the VAE
    vae.fit(
        normalized_vae_data,
        epochs=CONFIG['vae']['epochs'],
        batch_size=CONFIG['vae']['batch_size'],
        shuffle=True
    )

    # Generate and plot VAE scenarios
    plot_vae_generated_scenarios(
        decoder, scaler_vae, CONFIG['vae']['latent_dim'],
        input_dim=input_dim_vae, num_scenarios=10
    )
        # Step 14: Generative Adversarial Network (GAN)
    # Build GAN models
    generator_gan, discriminator_gan, gan_model = build_gan_models(
        latent_dim_gan=CONFIG['gan']['latent_dim_gan'],
        input_dim_gan=input_dim_vae
    )

    # Train GAN
    train_gan(generator_gan, discriminator_gan, gan_model, normalized_vae_data, CONFIG)

    # Step 15: Statistical Analysis of Generated Data
    # -------------------------------

    # Example: Compare ARIMA forecast with GAN generated data
    if 'forecast_values' in locals():
        real_sample = forecast_values.values.flatten()
        # Generate fake data using GAN
        generated_sample = scaler_vae.inverse_transform(
            generator_gan.predict(
                np.random.normal(0, 1, (len(real_sample), CONFIG['gan']['latent_dim_gan'])).astype(np.float32)
            )
        )[:len(real_sample)].flatten()

        # Perform statistical tests
        stats_results = perform_statistical_tests(real_sample, generated_sample)
        logging.info(f"Statistical Test Results:\n{stats_results}")

        # Plot statistical comparison
        plot_statistical_comparison(real_sample, generated_sample)
    else:
        logging.warning("ARIMA forecast data not available for statistical analysis.")

    # -------------------------------
    # Step 16: Scenario Generation and Visualization
    # -------------------------------

    future_years = range(global_trends['Year'].max() + 1, global_trends['Year'].max() + CONFIG['arima']['forecast_steps'] + 1)
    scenarios_df = generate_future_scenarios(decoder, scaler_vae, CONFIG['vae']['latent_dim'], input_dim=input_dim_vae, num_scenarios=CONFIG['scenarios']['num_scenarios'])
    plot_future_scenarios(scenarios_df, future_years)

    # -------------------------------
    # Step 17: Time-Series Comparison
    # -------------------------------

    # Example: Compare ARIMA forecast with a sample generated trend
    if 'forecast_values' in locals():
        sample_generated_trend = generator_gan.predict(
            np.random.normal(0, 1, (len(forecast_values), CONFIG['gan']['latent_dim_gan'])).astype(np.float32)
        )
        sample_generated_trend_rescaled = scaler_vae.inverse_transform(sample_generated_trend)[:len(forecast_values)].flatten()
        plot_time_series_comparison(forecast_values.values.flatten(), sample_generated_trend_rescaled)
    else:
        logging.warning("ARIMA forecast data not available for time-series comparison.")

    # -------------------------------
    # Step 18: Save Models and Results
    # -------------------------------

    save_models(encoder, decoder, generator_gan)
    save_clustering_results(clustered_countries, filename='clustered_countries.csv')

    logging.info("Climate data analysis and scenario generation completed successfully.")

# -------------------------------
# Execute Main Function
# -------------------------------

if __name__ == "__main__":
    main()
