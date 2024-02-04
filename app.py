from flask import Flask, request, render_template
import pickle
import pandas as pd

# Create the Flask app
app = Flask(__name__)

# Load the model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Function to ensure the input DataFrame matches the training DataFrame's columns
def align_input_to_training_columns(input_df, training_columns):
    missing_cols = set(training_columns) - set(input_df.columns)
    for c in missing_cols:
        input_df[c] = 0
    input_df = input_df[training_columns]
    return input_df

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect form data
    data = {
        'rent_numeric': [float(request.form['rent_numeric'])],
        'beds': [int(request.form['beds'])],
        'baths': [int(request.form['baths'])],
        'parking': [int(request.form['parking'])],
        'days_listed': [int(request.form['days_listed'])],
        'suburb': [request.form['suburb']],
        'post_code': [request.form['post_code']]
    }

    # Create a DataFrame from the form data
    input_data = pd.DataFrame(data)

    # One-hot encode 'suburb' and 'post_code'
    input_data_encoded = pd.get_dummies(input_data, columns=['suburb', 'post_code'], drop_first=True)
    
    # Align input DataFrame to the training DataFrame's columns
    training_columns = pickle.load(open('training_columns.pkl', 'rb'))  # Load the list of columns from training
    input_data_aligned = align_input_to_training_columns(input_data_encoded, training_columns)

    # Make a prediction
    prediction = model.predict(input_data_aligned)

    # Return the result
    output = prediction[0]
    return render_template('index.html', prediction_text='Rental price prediction result: {}'.format('Reasonable' if output == 1 else 'Unreasonable'))

if __name__ == "__main__":
    app.run(debug=True)
