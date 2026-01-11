import os
import numpy as np
from flask import Flask, render_template, request, redirect, session, url_for, flash, jsonify
import keras
from keras.models import load_model, Model
from keras.layers import Flatten, Dropout, Dense, Input
from keras.applications import VGG16
from keras.preprocessing.image import load_img, img_to_array
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from authlib.integrations.flask_client import OAuth
from api_key import CLIENT_ID, CLIENT_SECRET
import h5py
import zipfile
import tempfile
import shutil
import traceback
from PIL import Image

app = Flask(__name__)
app.secret_key = "your_secret_key"

# Configure SQL Alchemy
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

oauth = OAuth(app)

google = oauth.register(
    name='google',
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"}
)

# Database models
from datetime import datetime

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(120), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(150), nullable=True)
    role = db.Column(db.String(20), default='Patient')
    is_verified = db.Column(db.Boolean, default=True)
    phone = db.Column(db.String(20), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    doctor_profile = db.relationship('DoctorProfile', backref='user', uselist=False, cascade="all, delete-orphan")
    appointments_as_patient = db.relationship('Appointment', foreign_keys='Appointment.patient_id', backref='patient', lazy=True)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
        
    @property
    def full_name(self):
        return self.username

    def __repr__(self):
        return f"<User {self.username}, Email: {self.email}, Role: {self.role}>"

class DoctorProfile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    specialization = db.Column(db.String(100), nullable=True)
    qualification = db.Column(db.String(100), nullable=True)
    experience_years = db.Column(db.Integer, nullable=True)
    description = db.Column(db.Text, nullable=True)
    phone = db.Column(db.String(20), nullable=True)
    image_file = db.Column(db.String(120), nullable=True, default='default_doctor.jpg')
    
    @property
    def full_name(self):
        return self.user.full_name

class Appointment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    doctor_id = db.Column(db.Integer, db.ForeignKey('doctor_profile.id'), nullable=False)
    appointment_date = db.Column(db.DateTime, nullable=False)
    status = db.Column(db.String(20), default='pending')
    reason = db.Column(db.String(255), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    doctor = db.relationship('DoctorProfile', backref='appointments')

class ScanResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    scan_type = db.Column(db.String(50), nullable=False, default='Brain Tumor')
    result = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.String(20), nullable=False)
    image_path = db.Column(db.String(255), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    user = db.relationship('User', backref=db.backref('scan_results', lazy=True))



def load_class_order(model_type='tumor'):
    """Load class names from class_order.txt in correct order"""
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    if model_type == 'tumor':
        class_file = os.path.join(BASE_DIR, "models", "class_order.txt")
        fallback_classes = sorted(['glioma', 'meningioma', 'notumor', 'pituitary'])
    else:  # alzheimer
        class_file = os.path.join(BASE_DIR, "models", "alzheimer_class_order.txt")
        fallback_classes = sorted(['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented'])
    
    if os.path.exists(class_file):
        with open(class_file, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        print(f"✓ {model_type.capitalize()} class order loaded from file: {classes}")
        return classes
    else:
        print(f"⚠️  {class_file} not found. Using sorted fallback: {fallback_classes}")
        return fallback_classes

# Load class labels
tumor_class_labels = load_class_order('tumor')
alzheimer_class_labels = load_class_order('alzheimer')

UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def get_model():
    """Load the trained brain tumor .keras model with fallback for reconstruction"""
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    KERAS_MODEL_PATH = os.path.join(BASE_DIR, "models", "brain_tumor_model.keras")
    
    if not os.path.exists(KERAS_MODEL_PATH):
        print(f"⚠️  CRITICAL: Model file not found at {KERAS_MODEL_PATH}")
        return None

    print("\n" + "=" * 70)
    print("DEBUG: ATTEMPTING ROBUST MODEL LOAD (v2.1)")
    print("=" * 70)
    
    # Step 1: Try standard Keras loading
    try:
        print("Step 1: Attempting native load_model...")
        model = load_model(KERAS_MODEL_PATH, compile=False)
        print("✓ SUCCESS: Model loaded natively.")
        return model
    except Exception as e:
        print(f"Native load failed: {str(e)[:100]}... Attempting reconstruction...")

    # Step 2: Functional Reconstruction
    try:
        # Build the exact architecture
        vgg_base = VGG16(weights=None, include_top=False, input_shape=(128, 128, 3))
        x = vgg_base.output
        x = Flatten(name="flatten")(x)
        x = Dropout(0.3, name="dropout")(x)
        x = Dense(128, activation='relu', name="dense")(x)
        x = Dropout(0.2, name="dropout_1")(x)
        outputs = Dense(4, activation='softmax', name="dense_1")(x)
        model_final = Model(inputs=vgg_base.input, outputs=outputs, name="reconstructed_model")
        
        temp_dir = tempfile.mkdtemp()
        try:
            weights_path = os.path.join(temp_dir, "model.weights.h5")
            # The .keras file is a zip archive. Extract the weights.
            with zipfile.ZipFile(KERAS_MODEL_PATH, 'r') as zf:
                if 'model.weights.h5' in zf.namelist():
                    zf.extract('model.weights.h5', temp_dir)
                else:
                    print("✗ Error: model.weights.h5 not found inside .keras file")
                    return None

            # Map weights manually
            with h5py.File(weights_path, 'r') as f:
                weight_data = {}
                def collect_weights(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        weight_data[name] = obj[()] if obj.shape == () else obj[:]
                f.visititems(collect_weights)
                
                def apply_weights_to_layer(layer, weight_key_path):
                    k_val, b_val = None, None
                    # Use exact matching to avoid 'conv2d_1' matching 'conv2d_10'
                    prefix = f"{weight_key_path}/vars/"
                    
                    if f"{prefix}0" in weight_data:
                        k_val = weight_data[f"{prefix}0"]
                    if f"{prefix}1" in weight_data:
                        b_val = weight_data[f"{prefix}1"]
                    
                    if k_val is not None:
                        try:
                            layer.set_weights([k_val, b_val] if b_val is not None else [k_val])
                            return True
                        except Exception as ex:
                            print(f"      Error setting weights for {weight_key_path}: {ex}")
                            return False
                    return False

                # 1. Map VGG Layers (13 Conv2D layers)
                vgg_conv_layers = [l for l in vgg_base.layers if "conv" in l.name]
                vgg_count = 0
                for i, layer in enumerate(vgg_conv_layers):
                    # Keras naming convention for extracted weights
                    suffix = "" if i == 0 else f"_{i}"
                    weight_path = f"layers/functional/layers/conv2d{suffix}"
                    if apply_weights_to_layer(layer, weight_path):
                        vgg_count += 1
                
                # 2. Map Head Layers
                head_count = 0
                for i, layer_name in enumerate(["dense", "dense_1"]):
                    weight_path = f"layers/{layer_name}"
                    try:
                        layer = model_final.get_layer(layer_name)
                        if apply_weights_to_layer(layer, weight_path):
                            head_count += 1
                    except Exception:
                        pass
                
                print(f"Mapped weights: {vgg_count} VGG layers, {head_count} head layers.")

            print("✓ SUCCESS: Model reconstructed surgically via Functional API mapping.")
            return model_final
            
        finally:
            shutil.rmtree(temp_dir)
            
    except Exception as e:
        print(f"✗ All restoration attempts failed: {e}")
        traceback.print_exc()
        return None


def get_alzheimer_model():
    """Load the trained Alzheimer's .keras model"""
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    KERAS_MODEL_PATH = os.path.join(BASE_DIR, "models", "cnn_best_weights_v2.keras")
    
    if not os.path.exists(KERAS_MODEL_PATH):
        print(f"⚠️  CRITICAL: Alzheimer model file not found at {KERAS_MODEL_PATH}")
        return None

    print("\n" + "=" * 70)
    print("DEBUG: LOADING ALZHEIMER'S MODEL")
    print("=" * 70)
    
    try:
        print("Attempting native load_model...")
        model = load_model(KERAS_MODEL_PATH, compile=False)
        print("✓ SUCCESS: Alzheimer's model loaded.")
        return model
    except Exception as e:
        print(f"✗ Alzheimer's model load failed: {e}")
        traceback.print_exc()
        return None


# Initialize models at startup
print("=" * 70)
print("INITIALIZING MEDICAL DIAGNOSIS SYSTEM")
print("=" * 70)

tumor_model = get_model()
if tumor_model:
    print(f"✓ Brain Tumor Model loaded")
    print(f"✓ Tumor Classes: {tumor_class_labels}")
else:
    print("✗ Brain Tumor Model failed to load")

alzheimer_model = get_alzheimer_model()
if alzheimer_model:
    print(f"✓ Alzheimer's Model loaded")
    print(f"✓ Alzheimer Classes: {alzheimer_class_labels}")
else:
    print("✗ Alzheimer's Model failed to load")

print("=" * 70)


# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def make_prediction(image_path):
    """
    Make prediction on uploaded MRI image for Brain Tumor detection
    
    Args:
        image_path: Path to uploaded image file
        
    Returns:
        result: String describing the prediction
        confidence: Confidence score as percentage
    """
    global tumor_model
    
    if tumor_model is None:
        print("⚠️  Tumor model not loaded, attempting to reload...")
        tumor_model = get_model()
             
    if tumor_model is None:
        raise RuntimeError("Tumor model is not loaded. Please ensure models/brain_tumor_model.keras exists.")
    
    IMAGE_SIZE = 128
    
    try:
        # Load and preprocess image
        img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
        img_array = img_to_array(img) / 255.0  # Normalize to [0, 1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make prediction
        predictions = tumor_model.predict(img_array, verbose=0)
        probs = predictions[0]
        
        # Get predicted class
        predicted_class_index = np.argmax(probs)
        confidence_score = float(probs[predicted_class_index]) * 100
        predicted_class = tumor_class_labels[predicted_class_index]
        
        # Log prediction details
        print(f"\n{'='*60}")
        print(f"BRAIN TUMOR PREDICTION RESULTS:")
        print(f"{'='*60}")
        print(f"Predicted Class: {predicted_class}")
        print(f"Confidence: {confidence_score:.2f}%")
        print(f"\nAll Probabilities:")
        for i, cls in enumerate(tumor_class_labels):
            print(f"  {cls:12s}: {probs[i]*100:6.2f}%")
        print(f"{'='*60}\n")

        # Format result message
        if predicted_class == 'notumor':
            result = "No Tumor Detected"
        else:
            result = f"Tumor Detected: {predicted_class.capitalize()}"

        return result, round(confidence_score, 2)
        
    except Exception as e:
        print(f"✗ Error during brain tumor prediction: {e}")
        traceback.print_exc()
        raise


def make_alzheimer_prediction(image_path):
    """
    Make prediction on uploaded MRI image for Alzheimer's detection
    
    Args:
        image_path: Path to uploaded image file
        
    Returns:
        result: String describing the prediction
        confidence: Confidence score as percentage
        all_probabilities: Dictionary with all class probabilities
    """
    global alzheimer_model
    
    if alzheimer_model is None:
        print("⚠️  Alzheimer model not loaded, attempting to reload...")
        alzheimer_model = get_alzheimer_model()
             
    if alzheimer_model is None:
        raise RuntimeError("Alzheimer model is not loaded. Please ensure models/cnn_best_weights_v2.keras exists.")
    
    IMAGE_SIZE = 128
    
    try:
        # Load and preprocess image
        img = Image.open(image_path)
        
        # Convert to RGB (in case it's grayscale)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize to match model input
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # CRITICAL: Keep values in 0-255 range!
        # The model's Rescaling layer will handle normalization
        img_array = img_array.astype('float32')  # DO NOT divide by 255!
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        predictions = alzheimer_model.predict(img_array, verbose=0)
        probs = predictions[0]
        
        # Get predicted class
        predicted_class_index = np.argmax(probs)
        confidence_score = float(probs[predicted_class_index]) * 100
        predicted_class = alzheimer_class_labels[predicted_class_index]
        
        # Create probabilities dictionary
        all_probs = {}
        for i, cls in enumerate(alzheimer_class_labels):
            all_probs[cls] = float(probs[i] * 100)
        
        # Log prediction details
        print(f"\n{'='*60}")
        print(f"ALZHEIMER'S PREDICTION RESULTS:")
        print(f"{'='*60}")
        print(f"Predicted Class: {predicted_class}")
        print(f"Confidence: {confidence_score:.2f}%")
        print(f"\nAll Probabilities:")
        for cls, prob in all_probs.items():
            print(f"  {cls:20s}: {prob:6.2f}%")
        print(f"{'='*60}\n")

        # Format result message
        if predicted_class == 'NonDemented':
            result = "No Dementia Detected - Normal"
        elif predicted_class == 'VeryMildDemented':
            result = "Very Mild Dementia Detected"
        elif predicted_class == 'MildDemented':
            result = "Mild Dementia Detected"
        else:  # ModerateDemented
            result = "Moderate Dementia Detected"

        return result, round(confidence_score, 2), all_probs
        
    except Exception as e:
        print(f"✗ Error during Alzheimer's prediction: {e}")
        traceback.print_exc()
        raise



#Routes

@app.route("/", methods=["GET"])
def home():
    return render_template("login.html")

@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    from flask import send_from_directory
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route("/login", methods=["POST"])
def login():
    email = request.form['email']
    password = request.form['password']

    user = User.query.filter_by(email=email).first()
    if not user:
        flash("Account not found with this email. Please register first.", "error")
        return redirect(url_for("home"))

    if user.check_password(password):
        if user.role == "Doctor" and not user.is_verified:
            flash("Doctor account pending admin approval.", "error")
            return redirect(url_for("home"))

        session['username'] = user.username
        session['email'] = user.email
        session['role'] = user.role
        session['is_verified'] = user.is_verified
        session['user_id'] = user.id

        if user.role == "Admin":
            return redirect(url_for("admin_dashboard"))
        elif user.role == "Doctor":
            return redirect(url_for("doctor_dashboard"))
        else:
            return redirect(url_for("patient_dashboard"))

    flash("Incorrect password.", "error")
    return redirect(url_for("home"))


@app.route("/register", methods=["POST"])
def register():
    username = request.form['username']
    email = request.form['email']
    password = request.form['password']
    role = request.form.get('role', 'Patient')

    if User.query.filter_by(email=email).first():
        flash("Email already registered. Please login.", "error")
        return redirect(url_for("home"))
    
    if User.query.filter_by(username=username).first():
        flash("Username already taken. Please choose another.", "error")
        return redirect(url_for("home"))

    is_verified = True if role == "Patient" else False

    new_user = User(username=username, email=email, role=role, is_verified=is_verified)
    new_user.set_password(password)
    db.session.add(new_user)
    db.session.commit()

    if role == "Doctor":
        try:
            specialization = request.form.get('specialization')
            qualification = request.form.get('qualification')
            experience_years = request.form.get('experience_years')
            description = request.form.get('description')
            phone = request.form.get('phone')
            
            # Phone validation: exactly 10 digits
            if not phone or not phone.isdigit() or len(phone) != 10:
                db.session.delete(new_user)
                db.session.commit()
                flash("Phone number must be exactly 10 digits.", "error")
                return redirect(url_for("home"))
            
            image_file = 'default_doctor.jpg'
            if 'image' in request.files:
                file = request.files['image']
                if file and file.filename != '':
                    filename = f"doctor_{new_user.id}_{file.filename}"
                    profile_pics_path = os.path.join(app.root_path, 'static/images/profile_pics')
                    os.makedirs(profile_pics_path, exist_ok=True)
                    file.save(os.path.join(profile_pics_path, filename))
                    image_file = filename

            experience_years = int(experience_years) if experience_years else 0
            if experience_years < 0:
                db.session.delete(new_user)
                db.session.commit()
                flash("Experience years cannot be negative.", "error")
                return redirect(url_for("home"))
                
            doctor_profile = DoctorProfile(
                user_id=new_user.id,
                specialization=specialization,
                qualification=qualification,
                experience_years=experience_years,
                description=description,
                phone=phone,
                image_file=image_file
            )
            db.session.add(doctor_profile)
            db.session.commit()
            flash("Registration successful! Waiting for admin approval.", "success")
        except Exception as e:
            db.session.delete(new_user)
            db.session.commit()
            flash(f"Error creating doctor profile: {e}", "error")
            return redirect(url_for("home"))
    
    elif role == "Admin":
        session['username'] = new_user.username
        session['role'] = new_user.role
        session['is_verified'] = new_user.is_verified
        return redirect(url_for("admin_dashboard"))
    else:
        session['username'] = new_user.username
        session['role'] = new_user.role
        session['is_verified'] = new_user.is_verified
        return redirect(url_for("patient_dashboard"))
    
    return redirect(url_for("home"))

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))


@app.route('/login/google')
def login_google():
    redirect_uri = url_for('authorize_google', _external=True)
    return google.authorize_redirect(redirect_uri)

@app.route('/authorize/google')
def authorize_google():
    try:
        token = google.authorize_access_token()
        user_info = google.get('https://www.googleapis.com/oauth2/v3/userinfo').json()

        email = user_info.get('email')
        if not email:
            return "Google login failed: email not found", 400

        user = User.query.filter_by(email=email).first()

        if not user:
            user = User(
                username=user_info.get('name', email.split('@')[0]),
                email=email,
                role="Patient",
                is_verified=True
            )
            db.session.add(user)
            db.session.commit()

        if user.role == "Doctor" and not user.is_verified:
            flash("Doctor account pending admin approval.", "error")
            return redirect(url_for("home"))

        session['username'] = user.username
        session['email'] = user.email
        session['role'] = user.role
        session['is_verified'] = user.is_verified
        session['user_id'] = user.id

        if user.role == "Admin":
            return redirect(url_for("admin_dashboard"))
        elif user.role == "Doctor":
            return redirect(url_for("doctor_dashboard"))
        else:
            return redirect(url_for("patient_dashboard"))

    except Exception as e:
        app.logger.error(f"Google OAuth Error: {e}")
        return "Authentication failed", 500


@app.route("/patient_dashboard", methods=["GET"])
def patient_dashboard():
    if "username" in session:
        user = User.query.filter_by(username=session['username']).first()
        doctors = DoctorProfile.query.join(User).filter(User.is_verified == True).all()
        my_appointments = Appointment.query.filter_by(patient_id=user.id).all()
        
        return render_template("patient_dashboard.html", 
                               username=session['username'], 
                               doctors=doctors, 
                               appointments=my_appointments)
    return redirect(url_for('home'))

@app.route("/appointment", methods=["GET"])
def appointment_page():
    if "username" in session:
        doctors = DoctorProfile.query.join(User).filter(User.is_verified == True).all()
        return render_template("book_appointment.html", 
                               username=session['username'], 
                               doctors=doctors)
    return redirect(url_for('home'))


@app.route("/history", methods=["GET"])
def history():
    if "username" in session:
        user = User.query.filter_by(username=session['username']).first()
        my_appointments = Appointment.query.filter_by(patient_id=user.id).all()
        my_scans = ScanResult.query.filter_by(user_id=user.id).order_by(ScanResult.created_at.desc()).all()
        return render_template("history.html", 
                               username=session['username'], 
                               appointments=my_appointments,
                               scans=my_scans)
    return redirect(url_for('home'))


@app.route("/save_scan", methods=["POST"])
def save_scan():
    if "username" not in session:
        return jsonify({'status': 'error', 'message': 'Please login first.'}), 401
    
    user = User.query.filter_by(username=session['username']).first()
    if not user:
        return jsonify({'status': 'error', 'message': 'User not found.'}), 404
    
    scan_type = request.form.get('scan_type', 'Brain Tumor')
    result = request.form.get('result')
    confidence = request.form.get('confidence')
    image_path = request.form.get('image_path')
    
    if not result or not confidence:
        return jsonify({'status': 'error', 'message': 'Missing result data.'}), 400
        
    try:
        new_scan = ScanResult(
            user_id=user.id,
            scan_type=scan_type,
            result=result,
            confidence=confidence,
            image_path=image_path
        )
        db.session.add(new_scan)
        db.session.commit()
        return jsonify({'status': 'success', 'message': 'Scan result saved to history!'})
    except Exception as e:
        db.session.rollback()
        print(f"ERROR in save_scan: {e}")
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route("/delete_scan/<int:scan_id>", methods=["POST"])
def delete_scan(scan_id):
    if "username" not in session:
        return redirect(url_for('home'))
    
    user = User.query.filter_by(username=session['username']).first()
    scan = ScanResult.query.get(scan_id)
    
    if scan and scan.user_id == user.id:
        db.session.delete(scan)
        db.session.commit()
        flash("Scan record deleted.", "success")
    else:
        flash("Scan record not found or unauthorized.", "error")
        
    return redirect(url_for('history'))


@app.route("/doctor_dashboard", methods=["GET"])
def doctor_dashboard():
    if "username" in session and session.get('role') == 'Doctor':
        user = User.query.filter_by(username=session['username']).first()
        if not user or not user.is_verified:
             return redirect(url_for('home'))
             
        doctor_profile = user.doctor_profile
        pending_count = 0
        approved_count = 0
        total_count = 0
        
        if doctor_profile:
             appointments = Appointment.query.filter_by(doctor_id=doctor_profile.id).all()
             total_count = len(appointments)
             pending_count = len([a for a in appointments if a.status == 'pending'])
             approved_count = len([a for a in appointments if a.status == 'approved'])
             
        return render_template("doctor_dashboard.html", 
                               username=session['username'], 
                               pending_count=pending_count,
                               approved_count=approved_count,
                               total_count=total_count)
    return redirect(url_for('home'))


@app.route("/doctor/profile", methods=["GET", "POST"])
def doctor_profile():
    if "username" in session and session.get('role') == 'Doctor':
        user = User.query.filter_by(username=session['username']).first()
        if not user:
            return redirect(url_for('home'))
        
        doctor = user.doctor_profile
        
        if request.method == "POST":
            doctor.specialization = request.form.get('specialization')
            doctor.qualification = request.form.get('qualification')
            try:
                exp_val = int(request.form.get('experience_years') or 0)
                if exp_val < 0:
                    flash("Experience years cannot be negative.", "error")
                    return redirect(url_for('doctor_profile'))
                doctor.experience_years = exp_val
            except ValueError:
                flash("Invalid experience years format.", "error")
                return redirect(url_for('doctor_profile'))
            phone = request.form.get('phone')
            if not phone or not phone.isdigit() or len(phone) != 10:
                flash("Phone number must be exactly 10 digits.", "error")
                return redirect(url_for('doctor_profile'))
            doctor.phone = phone
            doctor.description = request.form.get('description')
            
            if 'image' in request.files:
                file = request.files['image']
                if file and file.filename != '':
                    filename = f"doctor_{user.id}_{file.filename}"
                    profile_pics_path = os.path.join(app.root_path, 'static/images/profile_pics')
                    os.makedirs(profile_pics_path, exist_ok=True)
                    file.save(os.path.join(profile_pics_path, filename))
                    doctor.image_file = filename
            
            db.session.commit()
            flash("Profile updated successfully!", "success")
            return redirect(url_for('doctor_profile'))

        return render_template("doctor_profile.html", username=session['username'], doctor=doctor)
    return redirect(url_for('home'))

@app.route("/doctor/requests", methods=["GET"])
def doctor_requests():
    if "username" in session and session.get('role') == 'Doctor':
        user = User.query.filter_by(username=session['username']).first()
        if not user: return redirect(url_for('home'))
        
        doctor_profile = user.doctor_profile
        if doctor_profile:
            appointments = Appointment.query.filter_by(doctor_id=doctor_profile.id).all()
        else:
            appointments = []
            
        return render_template("doctor_requests.html", username=session['username'], appointments=appointments)
    return redirect(url_for('home'))


@app.route("/doctor/calendar", methods=["GET"])
def doctor_calendar():
    if "username" in session and session.get('role') == 'Doctor':
        user = User.query.filter_by(username=session['username']).first()
        if not user: return redirect(url_for('home'))
        
        doctor_profile = user.doctor_profile
        events = []
        if doctor_profile:
            appointments = Appointment.query.filter_by(doctor_id=doctor_profile.id).all()
            for appt in appointments:
                color = '#467BBD'
                if appt.status == 'approved': color = '#28a745'
                elif appt.status == 'rejected': color = '#dc3545'
                elif appt.status == 'pending': color = '#ffc107'
                
                events.append({
                    'title': f"{appt.patient.full_name} ({appt.status})",
                    'start': appt.appointment_date.isoformat(),
                    'color': color
                })
        
        import json
        return render_template("doctor_calendar.html", username=session['username'], events=json.dumps(events))
    return redirect(url_for('home'))



@app.route("/admin_dashboard", methods=["GET"])
def admin_dashboard():
    if "username" in session and session.get('role') == 'Admin':
        pending_doctors = DoctorProfile.query.join(User).filter(User.role == 'Doctor', User.is_verified == False).all()
        approved_count = DoctorProfile.query.join(User).filter(User.role == 'Doctor', User.is_verified == True).count()
        patients_count = User.query.filter_by(role='Patient').count()
        
        return render_template("admin_dashboard.html", 
                               username=session['username'],
                               pending_doctors=pending_doctors,
                               approved_count=approved_count,
                               patients_count=patients_count)
    return redirect(url_for('home'))


@app.route("/admin/approved_doctors", methods=["GET"])
def admin_approved_doctors():
    if "username" in session and session.get('role') == 'Admin':
        approved_doctors = DoctorProfile.query.join(User).filter(User.role == 'Doctor', User.is_verified == True).all()
        return render_template("admin_approved_doctors.html", username=session['username'], approved_doctors=approved_doctors)
    return redirect(url_for('home'))


@app.route("/admin/patients", methods=["GET"])
def admin_patients():
    if "username" in session and session.get('role') == 'Admin':
        patients = User.query.filter_by(role='Patient').all()
        return render_template("admin_patients.html", username=session['username'], patients=patients)
    return redirect(url_for('home'))



from datetime import timedelta

@app.route("/book_appointment/<int:doctor_id>", methods=["POST"])
def book_appointment(doctor_id):
    is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'
    
    if "username" not in session:
        if is_ajax:
             return jsonify({'status': 'error', 'message': 'Please login first.'}), 401
        return redirect(url_for('home'))
    
    user = User.query.filter_by(username=session['username']).first()
    date_str = request.form.get('date')
    reason = request.form.get('reason')
    
    try:
        appt_date = datetime.strptime(date_str, '%Y-%m-%dT%H:%M')
    except ValueError:
        msg = "Invalid date format"
        if is_ajax: return jsonify({'status': 'error', 'message': msg})
        flash(msg, "error")
        return redirect(url_for('patient_dashboard'))
        
    now = datetime.now()
    if appt_date < now:
        msg = "Cannot book appointments in the past."
        if is_ajax: return jsonify({'status': 'error', 'message': msg})
        flash(msg, "error")
        return redirect(url_for('patient_dashboard'))

    min_advance_date = now + timedelta(days=2)
    if appt_date < min_advance_date:
        msg = "Appointments must be booked at least 2 days in advance."
        if is_ajax: return jsonify({'status': 'error', 'message': msg})
        flash(msg, "error")
        return redirect(url_for('patient_dashboard'))

    if not (9 <= appt_date.hour < 17):
        msg = "Appointments only available between 9 AM and 5 PM."
        if is_ajax: return jsonify({'status': 'error', 'message': msg})
        flash(msg, "error")
        return redirect(url_for('patient_dashboard'))

    if appt_date.weekday() == 5:
        msg = "Appointments are not available on Saturdays."
        if is_ajax: return jsonify({'status': 'error', 'message': msg})
        flash(msg, "error")
        return redirect(url_for('patient_dashboard'))

    conflicting_appt = Appointment.query.filter(
        Appointment.doctor_id == doctor_id,
        Appointment.status.notin_(['rejected', 'cancelled']),
        Appointment.appointment_date > (appt_date - timedelta(minutes=30)),
        Appointment.appointment_date < (appt_date + timedelta(minutes=30))
    ).first()

    if conflicting_appt:
        msg = "This time slot is taken. Please choose a time at least 30 minutes away from existing bookings."
        if is_ajax: return jsonify({'status': 'error', 'message': msg})
        flash(msg, "error")
        return redirect(url_for('patient_dashboard'))

    patient_conflict = Appointment.query.filter(
        Appointment.patient_id == user.id,
        Appointment.status.notin_(['rejected', 'cancelled']),
        Appointment.appointment_date > (appt_date - timedelta(minutes=30)),
        Appointment.appointment_date < (appt_date + timedelta(minutes=30))
    ).first()

    if patient_conflict:
        msg = "You already have an appointment scheduled at this time with another doctor."
        if is_ajax: return jsonify({'status': 'error', 'message': msg})
        flash(msg, "error")
        return redirect(url_for('patient_dashboard'))
        
    new_appt = Appointment(
        patient_id=user.id,
        doctor_id=doctor_id,
        appointment_date=appt_date,
        reason=reason
    )
    db.session.add(new_appt)
    db.session.commit()
    
    if is_ajax: return jsonify({'status': 'success', 'message': 'Appointment requested successfully!'})
    flash("Appointment requested successfully!", "success")
    return redirect(url_for('patient_dashboard'))

@app.route("/cancel_appointment/<int:appointment_id>", methods=["POST"])
def cancel_appointment(appointment_id):
    if "username" not in session:
        return redirect(url_for('home'))
    
    user = User.query.filter_by(username=session['username']).first()
    appt = Appointment.query.get(appointment_id)
    
    if appt and appt.patient_id == user.id:
        if appt.status != 'cancelled':
            appt.status = 'cancelled'
            db.session.commit()
            flash("Appointment cancelled successfully.", "success")
        else:
             flash("Appointment is already cancelled.", "info")
    else:
        flash("Appointment not found or unauthorized.", "error")
        
    return redirect(url_for('history'))

@app.route("/approve_appointment/<int:appointment_id>", methods=["POST"])
def approve_appointment(appointment_id):
    if "username" not in session or session.get('role') != 'Doctor':
        return redirect(url_for('home'))
        
    appt = Appointment.query.get(appointment_id)
    if appt:
        appt.status = 'approved'
        db.session.commit()
        flash("Appointment approved", "success")
    return redirect(url_for('doctor_dashboard'))

@app.route("/reject_appointment/<int:appointment_id>", methods=["POST"])
def reject_appointment(appointment_id):
    if "username" not in session or session.get('role') != 'Doctor':
        return redirect(url_for('home'))
        
    appt = Appointment.query.get(appointment_id)
    if appt:
        appt.status = 'rejected'
        db.session.commit()
        flash("Appointment rejected", "success")
    return redirect(url_for('doctor_dashboard'))

@app.route("/approve_doctor/<int:doctor_id>", methods=["POST"])
def approve_doctor(doctor_id):
    if "username" not in session or session.get('role') != 'Admin':
         return redirect(url_for('home'))
         
    doc_profile = DoctorProfile.query.get(doctor_id)
    if doc_profile and doc_profile.user:
        doc_profile.user.is_verified = True
        db.session.commit()
        flash(f"Doctor {doc_profile.user.username} approved.", "success")
    return redirect(url_for('admin_dashboard'))

@app.route("/reject_doctor/<int:doctor_id>", methods=["POST"])
def reject_doctor(doctor_id):
    if "username" not in session or session.get('role') != 'Admin':
         return redirect(url_for('home'))

    doc_profile = DoctorProfile.query.get(doctor_id)
    if doc_profile and doc_profile.user:
        db.session.delete(doc_profile.user)
        db.session.commit()
        flash("Doctor registration rejected and removed.", "success")
    return redirect(url_for('admin_dashboard'))



@app.route("/detect_tumor", methods=["GET", "POST"])
def detect_tumor_route():
    if request.method == 'POST':
        try:
            # Check if file is present
            if 'file' not in request.files:
                flash("No file uploaded", "error")
                return render_template('detect_tumor.html', result=None)
            
            file = request.files['file']
            
            if file.filename == '':
                flash("No file selected", "error")
                return render_template('detect_tumor.html', result=None)
            
            # Validate file type
            allowed_extensions = {'png', 'jpg', 'jpeg'}
            if not ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
                flash("Invalid file type. Please upload PNG, JPG, or JPEG images.", "error")
                return render_template('detect_tumor.html', result=None)
            
            # Save the file
            filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
            file_location = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_location)

            print(f"\n{'='*60}")
            print(f"NEW BRAIN TUMOR PREDICTION REQUEST")
            print(f"{'='*60}")
            print(f"File: {filename}")
            print(f"Path: {file_location}")

            # Make prediction
            result, confidence = make_prediction(file_location)

            # Return result with image path
            return render_template('detect_tumor.html', 
                                   result=result, 
                                   confidence=f"{confidence}%", 
                                   file_path=f'/uploads/{filename}')

        except Exception as e:
            print(f"✗ Error in detect_tumor route: {e}")
            import traceback
            traceback.print_exc()
            flash(f"Error processing image: {str(e)}", "error")
            return render_template('detect_tumor.html', result=None)

    return render_template('detect_tumor.html', result=None)


@app.route("/detect_alzheimer", methods=["GET", "POST"])
def detect_alzheimer_route():
    if request.method == 'POST':
        try:
            # Check if file is present
            if 'file' not in request.files:
                flash("No file uploaded", "error")
                return render_template('detect_alzheimer.html', result=None)
            
            file = request.files['file']
            
            if file.filename == '':
                flash("No file selected", "error")
                return render_template('detect_alzheimer.html', result=None)
            
            # Validate file type
            allowed_extensions = {'png', 'jpg', 'jpeg'}
            if not ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
                flash("Invalid file type. Please upload PNG, JPG, or JPEG images.", "error")
                return render_template('detect_alzheimer.html', result=None)
            
            # Save the file
            filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
            file_location = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_location)

            print(f"\n{'='*60}")
            print(f"NEW ALZHEIMER'S DETECTION REQUEST")
            print(f"{'='*60}")
            print(f"File: {filename}")
            print(f"Path: {file_location}")

            # Make prediction using Alzheimer's model
            result, confidence, all_probs = make_alzheimer_prediction(file_location)

            # Return result with image path and all probabilities
            return render_template('detect_alzheimer.html', 
                                   result=result, 
                                   confidence=f"{confidence}%", 
                                   file_path=f'/uploads/{filename}',
                                   all_probabilities=all_probs)

        except Exception as e:
            print(f"✗ Error in detect_alzheimer route: {e}")
            import traceback
            traceback.print_exc()
            flash(f"Error processing image: {str(e)}", "error")
            return render_template('detect_alzheimer.html', result=None)

    return render_template('detect_alzheimer.html', result=None)


@app.route("/detect_parkinson", methods=["GET", "POST"])
def detect_parkinson_route():
    if request.method == 'POST':
        flash("Parkinson's detection model is coming soon! Stay tuned.", "info")
        return render_template('detect_parkinson.html', result=None)

    return render_template('detect_parkinson.html', result=None)





@app.route("/debug_db")
def debug_db():
    from sqlalchemy import inspect
    inspector = inspect(db.engine)
    tables = inspector.get_table_names()
    return jsonify({
        'tables': tables,
        'has_scan_result': 'scan_result' in tables,
        'database_uri': app.config['SQLALCHEMY_DATABASE_URI']
    })

@app.route("/health")
def health_check():
    """API endpoint to check if models are loaded"""
    return jsonify({
        'status': 'healthy' if (tumor_model is not None and alzheimer_model is not None) else 'partial',
        'tumor_model_loaded': tumor_model is not None,
        'alzheimer_model_loaded': alzheimer_model is not None,
        'tumor_classes': tumor_class_labels,
        'alzheimer_classes': alzheimer_class_labels
    })


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True, use_reloader=True)