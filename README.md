# Student Performance Analysis and Prediction System

This project is a small educational web application that demonstrates how to:

- Collect student performance data from a web form
- Store data in a local SQLite database using SQLAlchemy
- Perform basic analysis and visualization using Pandas and Matplotlib
- Train a simple Linear Regression model with scikit-learn
- Predict exam scores from study hours and attendance

## Project Structure

- `app.py` – Flask entry point and routes
- `database.py` – Database initialization (`SQLAlchemy`)
- `models.py` – `Student` model definition
- `analysis.py` – Data analysis and visualization helpers
- `ml_model.py` – Linear Regression model training and prediction
- `requirements.txt` – Python dependencies
- `data/students.csv` – Example CSV structure for student data
- `templates/index.html` – Main UI
- `static/style.css` – Basic styling

## Setup

1. Create and activate a virtual environment (recommended).
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Flask development server:

   ```bash
   python app.py
   ```

4. Open `http://127.0.0.1:5000` in your browser.

## Usage

- Use the **Add Student Record** form to enter new data.
- The table at the bottom shows all stored records.
- Summary statistics and plots update automatically when data is available.
- Once at least 5 records are saved, the **Predict Exam Score** form becomes fully useful and predictions will be based on the trained Linear Regression model.

This project is designed for academic and demonstration purposes and intentionally keeps the architecture simple and easy to read.

