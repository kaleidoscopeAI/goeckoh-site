class SuperNode:
    """
    Represents an autonomous, learning AI agent designed to mimic and master a specific real-world domain.
    """
    def __init__(self, node_id, domain_description):
        if not isinstance(node_id, str) or not node_id:
            raise ValueError("SuperNode: node_id must be a non-empty string.")
        if not isinstance(domain_description, str) or not domain_description:
            raise ValueError("SuperNode: domain_description must be a non-empty string.")

        self.node_id = node_id
        self.domain_description = domain_description
        self.historical_data = {} # Stores time series data for different aspects of the domain
        self.generative_models = {} # Models to 'mimic' or generate domain behavior
        self.prophet_models = {}  # Prophet models
        print(f"--- Super Node '{self.node_id}' Initialized: Forging reality's mimicry for {domain_description}. ---")

    def ingest_historical_data(self, data_series_name, data_points):
        """
        Ingest initial historical data for mimicry and causal analysis.
        Robustness: Validate input types and data points.
        """
        if not isinstance(data_series_name, str) or not data_series_name:
            print(f"  [ERROR] Super Node '{self.node_id}': data_series_name must be a non-empty string.")
            return
        if not isinstance(data_points, (list, np.ndarray)) or not data_points:
            print(f"  [ERROR] Super Node '{self.node_id}': data_points must be a non-empty list or numpy array.")
            return
        try:
            dates = pd.date_range(start='2020-01-01', periods=len(data_points), freq='D')
            self.historical_data[data_series_name] = pd.DataFrame({'ds': dates, 'y': data_points})
            print(f"  Super Node '{self.node_id}': Ingested historical data for '{data_series_name}'.")
        except Exception as e:
            print(f"  [ERROR] Super Node '{self.node_id}': Failed to ingest data for '{data_series_name}': {e}")

    def crawl_and_ingest_data(self, url, data_series_name, selector):
        """
        Crawls a web page and ingests data using BeautifulSoup.
        Selector is the CSS selector for the data elements.
        """
        if not isinstance(url, str) or not url:
            print(f"  [ERROR] Super Node '{self.node_id}': url must be a non-empty string.")
            return
        if not isinstance(selector, str) or not selector:
            print(f"  [ERROR] Super Node '{self.node_id}': selector must be a non-empty string.")
            return
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            elements = soup.select(selector)
            data_points = [float(el.text.strip()) for el in elements if el.text.strip().replace('.', '', 1).isdigit()]
            if not data_points:
                print(f"  [WARNING] Super Node '{self.node_id}': No data found using selector '{selector}'.")
                return
            self.ingest_historical_data(data_series_name, data_points)
        except requests.RequestException as e:
            print(f"  [ERROR] Super Node '{self.node_id}': Failed to crawl '{url}': {e}")
        except ValueError as e:
            print(f"  [ERROR] Super Node '{self.node_id}': Data parsing error: {e}")

    def optimize_arima_parameters(self, data, p_range=range(0, 6), d_range=range(0, 3), q_range=range(0, 6), P_range=range(0, 3), D_range=range(0, 3), Q_range=range(0, 3), s=12):
        """
        Advanced optimization of SARIMA parameters using parallel grid search based on AIC.
        """
        def fit_sarima(order):
            p, d, q, P, D, Q = order
            try:
                model = SARIMAX(data, order=(p, d, q), seasonal_order=(P, D, Q, s))
                results = model.fit(disp=False)
                return (p, d, q, P, D, Q), results.aic
            except Exception:
                return order, np.inf

        orders = list(itertools.product(p_range, d_range, q_range, P_range, D_range, Q_range))
        with Pool(processes=mp.cpu_count()) as pool:
            results = pool.map(fit_sarima, orders)

        best_order, best_aic = min(results, key=lambda x: x[1])
        if best_aic == np.inf:
            return None, np.inf
        return best_order, best_aic

    def train_generative_models(self, series_names):
        """
        Trains simple AutoRegressive models to 'generatively mimic' domain behavior.
        This is a highly simplified generative simulation.
        Robustness: Validate series_names and handle training errors.
        """
        if not isinstance(series_names, list) or not series_names:
            print(f"  [ERROR] Super Node '{self.node_id}': series_names must be a non-empty list of strings.")
            return

        print(f"  Super Node '{self.node_id}': Training generative models for {', '.join(series_names)}...")
        for name in series_names:
            if name not in self.historical_data or self.historical_data[name].empty:
                print(f"    [WARNING] No historical data for '{name}'. Skipping model training.")
                self.generative_models[name] = None
                continue
            if len(self.historical_data[name]) < 10:
                print(f"    [WARNING] Not enough historical data for '{name}' to train generative model (min 10 points needed). Skipping.")
                self.generative_models[name] = None
                continue
            try:
                data = self.historical_data[name]['y']
                best_order, best_aic = self.optimize_arima_parameters(data)
                if best_order is None:
                    print(f"    [WARNING] Could not find optimal parameters for '{name}'. Skipping.")
                    continue
                p, d, q, P, D, Q = best_order
                print(f"    Optimal SARIMA order for '{name}': (({p},{d},{q}), ({P},{D},{Q},12)) with AIC {best_aic:.2f}")
                self.generative_models[name] = SARIMAX(data, order=(p, d, q), seasonal_order=(P, D, Q, 12)).fit(disp=False)
                print(f"    Model for '{name}' trained successfully.")
            except Exception as e:
                print(f"    [ERROR] Super Node '{self.node_id}': Error training model for '{name}': {e}")
                self.generative_models[name] = None

    def train_prophet_model(self, series_name):
        """
        Trains a Prophet model for forecasting.
        """
        if series_name not in self.historical_data or self.historical_data[series_name].empty:
            print(f"    [WARNING] No historical data for '{series_name}'. Skipping Prophet training.")
            return
        try:
            df = self.historical_data[series_name]
            model = Prophet()
            model.fit(df)
            self.prophet_models[series_name] = model
            print(f"    Prophet model for '{series_name}' trained successfully.")
        except Exception as e:
            print(f"    [ERROR] Super Node '{self.node_id}': Error training Prophet model for '{series_name}': {e}")

    def perform_prophet_forecast(self, series_name, steps=1):
        """
        Performs forecasting using the Prophet model.
        """
        if series_name not in self.prophet_models:
            print(f"  [WARNING] Super Node '{self.node_id}': Prophet model for '{series_name}' not trained.")
            return None
        model = self.prophet_models[series_name]
        future = model.make_future_dataframe(periods=steps)
        forecast = model.predict(future)
        return forecast['yhat'][-steps:].values

    def perform_sarimax_forecast(self, series_name, steps=1):
        """
        Performs forecasting using the SARIMAX model.
        """
        if series_name not in self.generative_models or self.generative_models[series_name] is None:
            print(f"  [WARNING] Super Node '{self.node_id}': SARIMAX model for '{series_name}' not trained.")
            return None
        model = self.generative_models[series_name]
        forecast = model.forecast(steps=steps)
        return forecast.values

    def perform_generative_simulation(self, series_name, steps=1, use_hybrid=False):
        """
        Simulates future 'behavior' using the generative models or Prophet or hybrid.
        This represents a basic 'multi-fidelity, generative simulation'.
        Robustness: Validate series_name, steps, and handle prediction errors.
        """
        if not isinstance(series_name, str) or not series_name:
            print(f"  [ERROR] Super Node '{self.node_id}': series_name must be a non-empty string.")
            return
        if not isinstance(steps, int) or steps <= 0:
            print(f"  [ERROR] Super Node '{self.node_id}': steps must be a positive integer.")
            return

        if use_hybrid:
            prophet_forecast = self.perform_prophet_forecast(series_name, steps)
            sarimax_forecast = self.perform_sarimax_forecast(series_name, steps)
            if prophet_forecast is not None and sarimax_forecast is not None:
                hybrid_forecast = (prophet_forecast + sarimax_forecast) / 2
                print(f"    Hybrid Simulated '{series_name}': {np.round(hybrid_forecast, 2)}")
                return hybrid_forecast
            else:
                print(f"  [WARNING] Super Node '{self.node_id}': Hybrid forecast failed due to missing models.")
                return None

        prophet_forecast = self.perform_prophet_forecast(series_name, steps)
        if prophet_forecast is not None:
            return prophet_forecast

        sarimax_forecast = self.perform_sarimax_forecast(series_name, steps)
        if sarimax_forecast is not None:
            return sarimax_forecast

        print(f"  [WARNING] Super Node '{self.node_id}': No model available for '{series_name}'.")
        return None

    def infer_causal_links(self, series_x, series_y, max_lags=3):
        """
        Attempts to infer causal links between two time series using Granger Causality.
        This represents a basic form of 'causal inference network'.
        Robustness: Validate inputs, data availability, and handle statistical test errors.
        """
        if not isinstance(series_x, str) or not series_x or not isinstance(series_y, str) or not series_y:
            print(f"  [ERROR] Super Node '{self.node_id}': series_x and series_y must be non-empty strings.")
            return None, None
        if not isinstance(max_lags, int) or max_lags <= 0:
            print(f"  [ERROR] Super Node '{self.node_id}': max_lags must be a positive integer.")
            return None, None

        if (series_x not in self.historical_data or self.historical_data[series_x].empty or
            series_y not in self.historical_data or self.historical_data[series_y].empty):
            print(f"  [WARNING] Super Node '{self.node_id}': Missing historical data for '{series_x}' or '{series_y}'.")
            return None, None
        if len(self.historical_data[series_x]) <= max_lags or len(self.historical_data[series_y]) <= max_lags:
            print(f"  [WARNING] Super Node '{self.node_id}': Not enough data for causal inference (min {max_lags+1} points needed).")
            return None, None
        if len(self.historical_data[series_x]) != len(self.historical_data[series_y]):
            print(f"  [WARNING] Super Node '{self.node_id}': Time series '{series_x}' and '{series_y}' have different lengths. Granger causality may be unreliable.")

        data = pd.DataFrame({series_x: self.historical_data[series_x]['y'], series_y: self.historical_data[series_y]['y']})
        print(f"  Super Node '{self.node_id}': Inferring causal links: '{series_x}' vs '{series_y}' (max_lags={max_lags})...")
        try:
            # Does X Granger-cause Y?
            test_result = grangercausality(data[[series_y, series_x]], max_lags, verbose=False)
            p_value = test_result[max_lags][0]['ssr_ftest'][1]
            causal_direction = f"'{series_x}' Granger-causes '{series_y}'" if p_value < 0.05 else f"'{series_x}' does NOT Granger-cause '{series_y}'"
            print(f"    Result: {causal_direction} (p-value: {p_value:.4f})")
            return causal_direction, p_value
        except Exception as e:
            print(f"  [ERROR] Super Node '{self.node_id}': Error during Granger Causality for {series_x} vs {series_y}: {e}")
            return None, None

    def provide_proactive_insight(self):
        """
        Conceptual function to generate a proactive insight and an activity score
        for the Conscious Cube based on its domain's state.
        Robustness: Handle cases where models are not trained.
        """
        print(f"  Super Node '{self.node_id}': Generating proactive insight for Conscious Cube...")
        if 'inventory' in self.generative_models and self.generative_models['inventory']:
            predicted_inventory = self.perform_generative_simulation('inventory', steps=1)
            if predicted_inventory is not None:
                if predicted_inventory[0] < 95: # Hypothetical critical threshold
                    insight_text = f"Predicted critical low inventory ({predicted_inventory[0]:.2f}) for {self.domain_description}. Potential supply chain disruption imminent."
                    keywords = ['inventory', 'supply chain', 'risk', 'disruption']
                    activity_score = 0.9 # High activity for critical insight
                else:
                    insight_text = f"Inventory levels for {self.domain_description} are stable and within expected range ({predicted_inventory[0]:.2f})."
                    keywords = ['inventory', 'stable']
                    activity_score = 0.4 # Normal activity
                print(f"    Insight: {insight_text[:70]}...")
                return insight_text, keywords, activity_score
            else:
                print(f"  [WARNING] Super Node '{self.node_id}': Could not perform inventory simulation for insight. Defaulting.")
        return "No specific insight generated due to missing model or data.", [], 0.1 # Default low activity


