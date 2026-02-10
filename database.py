from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


def init_db(app):
    """Initialize the database and create all tables."""
    db.init_app(app)
    with app.app_context():
        from models import Student  # noqa: F401  # ensure models are registered

        db.create_all()

