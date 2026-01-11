from app import app, db, User

with app.app_context():
    admin = User.query.filter_by(email="admin@example.com").first()
    if not admin:
        admin = User(
            username="Admin",
            email="admin@example.com",
            role="Admin",
            is_verified=True
        )
        admin.set_password("admin123") 
        db.session.add(admin)
        db.session.commit()
        print("Admin created successfully")
    else:
        print("Admin already exists")
