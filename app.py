from flask import Flask, request, render_template, redirect, url_for
import pickle
import numpy as np
import os

app = Flask(__name__)

# --- Configuration ---
MODEL_PATH = 'model.pkl'

# Min-Max values for manual scaling
# Provided by the user:
# Age: max=85, min=7
# Flight Distance: max=3739.0, min=31.0
# Arrival Delay: max=32.5, min=0.0

MANUAL_SCALING_PARAMS = {
    'Age': {'min': 7, 'max': 85},
    'Flight Distance': {'min': 31.0, 'max': 3739.0},
    'Arrival Delay': {'min': 0.0, 'max': 32.5} # Assuming delay cannot be negative based on min:0.0
}

# --- Load Model ---
try:
    model = pickle.load(open(MODEL_PATH, 'rb'))
except FileNotFoundError:
    print(f"Error: Ensure '{MODEL_PATH}' is in the same directory as app.py.")
    model = None
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Define the order of features exactly as your model expects them
EXPECTED_FEATURE_ORDER = [
    'Age', 'Flight Distance', 'Arrival Delay',
    'Departure and Arrival Time Convenience', 'Ease of Online Booking',
    'Check-in Service', 'Online Boarding', 'Gate Location',
    'On-board Service', 'Seat Comfort', 'Leg Room Service', 'Cleanliness',
    'Food and Drink', 'In-flight Service', 'In-flight Wifi Service',
    'In-flight Entertainment', 'Baggage Handling',
    'Gender', 'Customer Type', 'Type of Travel', 'Class'
]

# Columns that will be manually scaled
MANUALLY_SCALED_COLUMNS = ['Age', 'Flight Distance', 'Arrival Delay']

def min_max_scale(value, min_val, max_val):
    """Applies Min-Max scaling to a single value."""
    if max_val == min_val: # Avoid division by zero if min and max are the same
        return 0.5 # Or 0, or raise an error, depending on desired behavior
    return (value - min_val) / (max_val - min_val)

@app.route('/')
def index():
    if model is None:
        return "Error: Model not loaded. Check console for details.", 500
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "Error: Model not loaded. Cannot predict.", 500

    if request.method == 'POST':
        try:
            # --- Collect and Process Form Data ---
            form_data = request.form.to_dict()
            input_features = {}

            # --- Numeric features (some to be manually scaled) ---
            # Get raw values first
            raw_age = float(form_data['age'])
            raw_flight_distance = float(form_data['flight_distance'])
            raw_arrival_delay = float(form_data['arrival_delay'])

            # Apply manual Min-Max scaling
            input_features['Age'] = min_max_scale(
                raw_age,
                MANUAL_SCALING_PARAMS['Age']['min'],
                MANUAL_SCALING_PARAMS['Age']['max']
            )
            input_features['Flight Distance'] = min_max_scale(
                raw_flight_distance,
                MANUAL_SCALING_PARAMS['Flight Distance']['min'],
                MANUAL_SCALING_PARAMS['Flight Distance']['max']
            )
            input_features['Arrival Delay'] = min_max_scale(
                raw_arrival_delay,
                MANUAL_SCALING_PARAMS['Arrival Delay']['min'],
                MANUAL_SCALING_PARAMS['Arrival Delay']['max']
            )
            
            # --- Rating features (0-5) ---
            rating_cols_map = {
                'Departure and Arrival Time Convenience': 'departure_arrival_time_convenience',
                'Ease of Online Booking': 'ease_of_online_booking',
                'Check-in Service': 'check_in_service',
                'Online Boarding': 'online_boarding',
                'Gate Location': 'gate_location',
                'On-board Service': 'on_board_service',
                'Seat Comfort': 'seat_comfort',
                'Leg Room Service': 'leg_room_service',
                'Cleanliness': 'cleanliness',
                'Food and Drink': 'food_and_drink',
                'In-flight Service': 'in_flight_service',
                'In-flight Wifi Service': 'in_flight_wifi_service',
                'In-flight Entertainment': 'in_flight_entertainment',
                'Baggage Handling': 'baggage_handling'
            }

            for model_col, form_col in rating_cols_map.items():
                input_features[model_col] = int(form_data[form_col])


            # --- Categorical features (apply custom encoding) ---
            input_features['Gender'] = int(form_data['gender'])
            input_features['Customer Type'] = int(form_data['customer_type'])
            input_features['Type of Travel'] = int(form_data['type_of_travel'])
            input_features['Class'] = int(form_data['class_type'])

            # --- Assemble features in the correct order for the model ---
            final_features_list = []
            for feature_name in EXPECTED_FEATURE_ORDER:
                if feature_name not in input_features:
                    raise ValueError(f"Missing feature in input_features: {feature_name}")
                final_features_list.append(input_features[feature_name])

            # Convert to NumPy array and reshape for prediction (model expects 2D array)
            final_features_array = np.array(final_features_list).reshape(1, -1)

            # --- Make Prediction ---
            prediction_value = model.predict(final_features_array)
            output = round(prediction_value[0], 2) # Assuming regression, get the first element

            return render_template('result.html', prediction=output)

        except ValueError as ve:
            print(f"ValueError: {ve}")
            return f"Error in input data: {ve}. Please check your inputs. <a href='{url_for('index')}'>Go back</a>", 400
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return f"An error occurred during prediction: {e}. <a href='{url_for('index')}'>Go back</a>", 500

    return redirect(url_for('index'))


if __name__ == '__main__':
    if not os.path.exists(MODEL_PATH):
        print("---")
        print(f"CRITICAL ERROR: Model file '{MODEL_PATH}' not found.")
        print("Please ensure this file is in the same directory as app.py before running.")
        print("The application will likely fail if it proceeds.")
        print("---")
    app.run(debug=True)