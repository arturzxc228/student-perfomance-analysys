from database import db


class Student(db.Model):
    """Database model representing a student's academic record."""

    __tablename__ = "students"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    study_hours = db.Column(db.Float, nullable=False)
    attendance = db.Column(db.Float, nullable=False)  # percentage (0-100)
    exam_score = db.Column(db.Float, nullable=False)  # percentage (0-100)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Student {self.id}: {self.name}>"

