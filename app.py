import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from flask import Flask, render_template, request, redirect, url_for, flash

from database import init_db, db
from models import Student
from analysis import (
    get_summary_statistics,
    generate_all_plots,
)
from ml_model import train_model, predict_score


BASE_DIR = Path(__file__).resolve().parent
LOG_FILE = BASE_DIR / "app.log"


def create_app():
    app = Flask(__name__)
    app.config["SECRET_KEY"] = "change-this-in-production"
    app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{BASE_DIR / 'students.db'}"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    init_db(app)
    configure_logging(app)
    ensure_folders()

    @app.route("/", methods=["GET", "POST"])
    def index():
        if request.method == "POST":
            name = request.form.get("name", "").strip()
            age = request.form.get("age", "").strip()
            study_hours = request.form.get("study_hours", "").strip()
            attendance = request.form.get("attendance", "").strip()
            exam_score = request.form.get("exam_score", "").strip()

            valid, message = validate_student_input(
                name=name,
                age=age,
                study_hours=study_hours,
                attendance=attendance,
                exam_score=exam_score,
            )

            if not valid:
                flash(message, "error")
            else:
                try:
                    student = Student(
                        name=name,
                        age=int(age),
                        study_hours=float(study_hours),
                        attendance=float(attendance),
                        exam_score=float(exam_score),
                    )
                    db.session.add(student)
                    db.session.commit()
                    flash("Student record added successfully.", "success")
                except Exception as exc:  # pragma: no cover - defensive logging
                    app.logger.exception("Failed to add student record: %s", exc)
                    db.session.rollback()
                    flash("An error occurred while saving the record.", "error")

            return redirect(url_for("index"))

        students = Student.query.order_by(Student.id.desc()).all()
        stats = get_summary_statistics()
        # Generate or refresh plots
        plot_paths = generate_all_plots()
        trained = train_model()

        return render_template(
            "index.html",
            students=students,
            stats=stats,
            plot_paths=plot_paths,
            model_trained=trained,
            predicted_score=None,
        )

    @app.route("/predict", methods=["POST"])
    def predict():
        study_hours = request.form.get("predict_study_hours", "").strip()
        attendance = request.form.get("predict_attendance", "").strip()

        valid, message = validate_prediction_input(
            study_hours=study_hours,
            attendance=attendance,
        )
        if not valid:
            flash(message, "error")
            return redirect(url_for("index"))

        if not train_model():
            flash("Not enough data to train the model. Add more student records.", "error")
            return redirect(url_for("index"))

        try:
            predicted = predict_score(
                study_hours=float(study_hours),
                attendance=float(attendance),
            )
            flash(f"Predicted exam score: {predicted:.2f}", "info")
        except Exception as exc:  # pragma: no cover - defensive logging
            logging.getLogger(__name__).exception("Prediction failed: %s", exc)
            flash("An error occurred while predicting the score.", "error")

        return redirect(url_for("index"))

    return app


def ensure_folders():
    """Ensure that data and static/plots folders exist."""
    data_dir = BASE_DIR / "data"
    plots_dir = BASE_DIR / "static" / "plots"
    data_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)


def configure_logging(app: Flask):
    """Configure file-based logging for the application."""
    LOG_FILE.parent.mkdir(exist_ok=True)

    handler = RotatingFileHandler(LOG_FILE, maxBytes=1_000_000, backupCount=3)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)

    app.logger.setLevel(logging.INFO)
    app.logger.addHandler(handler)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)


def validate_student_input(name, age, study_hours, attendance, exam_score):
    """Validate form data for creating a student record."""
    if not name:
        return False, "Name is required."

    try:
        age_val = int(age)
        if age_val <= 0:
            return False, "Age must be a positive integer."
    except ValueError:
        return False, "Age must be an integer."

    for field_name, value, allow_zero in [
        ("Study hours", study_hours, False),
        ("Attendance", attendance, True),
        ("Exam score", exam_score, True),
    ]:
        try:
            val = float(value)
        except ValueError:
            return False, f"{field_name} must be a number."

        if not allow_zero and val <= 0:
            return False, f"{field_name} must be greater than 0."

        if field_name == "Attendance" and not (0 <= val <= 100):
            return False, "Attendance must be between 0 and 100."
        if field_name == "Exam score" and not (0 <= val <= 100):
            return False, "Exam score must be between 0 and 100."

    return True, ""


def validate_prediction_input(study_hours, attendance):
    """Validate form data for prediction."""
    if not study_hours or not attendance:
        return False, "Both study hours and attendance are required for prediction."

    try:
        sh = float(study_hours)
        att = float(attendance)
    except ValueError:
        return False, "Study hours and attendance must be numeric."

    if sh <= 0:
        return False, "Study hours must be greater than 0."
    if not (0 <= att <= 100):
        return False, "Attendance must be between 0 and 100."

    return True, ""


if __name__ == "__main__":
    application = create_app()
    application.run(debug=True)

