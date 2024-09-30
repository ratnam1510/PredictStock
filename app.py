import logging
from flask import Flask, render_template, request, redirect, url_for, jsonify, session, flash
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.src.models import Sequential
from keras.src.layers import LSTM, Dense, Dropout
from keras.src.optimizers import Adam
from keras.src.callbacks import EarlyStopping
import yfinance as yf
import re
from datetime import datetime, timedelta
import pytz
from flask_mail import Mail, Message
import pyotp
import qrcode
from io import BytesIO
import base64
import random
import string
from concurrent.futures import ThreadPoolExecutor
import time

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'  # Use a strong, randomly generated key in production
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'your_email@gmail.com'
app.config['MAIL_PASSWORD'] = 'your_app_password'  # Use app-specific password
db = SQLAlchemy(app)
migrate = Migrate(app, db)
mail = Mail(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    is_verified = db.Column(db.Boolean, default=False)
    two_factor_secret = db.Column(db.String(32))


with app.app_context():
    db.create_all()


def generate_otp():
    return ''.join(random.choices(string.digits, k=6))

def send_verification_email(email, otp):
    msg = Message('Verify Your Email', sender=app.config['MAIL_USERNAME'], recipients=[email])
    msg.body = f'Your verification code is: {otp}'
    mail.send(msg)

def create_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def prepare_data(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

def validate_stock_symbol(symbol):
    pattern = r'^[A-Z]{1,5}(\.[A-Z]{1,2})?$'
    return re.match(pattern, symbol) is not None

@app.route('/')
def home():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        if User.query.filter_by(username=username).first() or User.query.filter_by(email=email).first():
            flash('Username or email already exists')
            return redirect(url_for('signup'))

        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(username=username, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        otp = generate_otp()
        send_verification_email(email, otp)
        session['email'] = email
        session['otp'] = otp

        return redirect(url_for('verify_email'))

    return render_template('signup.html')


@app.route('/verify-email', methods=['GET', 'POST'])
def verify_email():
    if 'email' not in session:
        return redirect(url_for('signup'))

    if request.method == 'POST':
        user_otp = request.form['otp']
        if user_otp == session['otp']:
            user = User.query.filter_by(email=session['email']).first()
            user.is_verified = True
            db.session.commit()
            session.pop('otp', None)
            return redirect(url_for('setup_2fa'))
        else:
            flash('Invalid OTP. Please try again.')

    return render_template('email.html')


@app.route('/setup-2fa', methods=['GET', 'POST'])
def setup_2fa():
    if 'email' not in session:
        return redirect(url_for('login'))

    user = User.query.filter_by(email=session['email']).first()

    if request.method == 'POST':
        otp = request.form['otp']
        if pyotp.TOTP(user.two_factor_secret).verify(otp):
            session.pop('email', None)
            session['user_id'] = user.id
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid OTP. Please try again.')

    if not user.two_factor_secret:
        user.two_factor_secret = pyotp.random_base32()
        db.session.commit()

    totp = pyotp.TOTP(user.two_factor_secret)
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(totp.provisioning_uri(user.email, issuer_name="YourApp"))
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buffered = BytesIO()
    img.save(buffered, "PNG")
    qr_code = base64.b64encode(buffered.getvalue()).decode()

    return render_template('twofa.html', qr_code=qr_code, secret=user.two_factor_secret)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        otp = request.form.get('otp')

        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            if not user.is_verified:
                flash('Please verify your email first.')
                return redirect(url_for('login'))

            if pyotp.TOTP(user.two_factor_secret).verify(otp):
                session['user_id'] = user.id
                return redirect(url_for('dashboard'))
            else:
                flash('Invalid 2FA code. Please try again.')
        else:
            flash('Invalid username or password')

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('login'))


@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html')


stock_cache = {}
executor = ThreadPoolExecutor(max_workers=5)

def fetch_stock_data(symbol):
    try:
        stock = yf.Ticker(symbol)
        ny_time = datetime.now(pytz.timezone('America/New_York'))
        is_market_open = 9 <= ny_time.hour < 16 and ny_time.weekday() < 5

        if is_market_open:
            current_price = stock.info['regularMarketPrice']
        else:
            last_trading_day = ny_time.date()
            if ny_time.weekday() >= 5:
                last_trading_day -= timedelta(days=ny_time.weekday() - 4)
            hist = stock.history(start=last_trading_day)
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
            else:
                return None

        return {
            "symbol": symbol,
            "current_price": current_price,
            "market_status": "open" if is_market_open else "closed",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logging.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

def update_stock_data():
    while True:
        futures = {executor.submit(fetch_stock_data, symbol): symbol for symbol in list(stock_cache.keys())}
        for future in futures:
            symbol = futures[future]
            try:
                data = future.result()
                if data:
                    stock_cache[symbol] = data
            except Exception as e:
                logging.error(f"Error updating {symbol}: {str(e)}")
        time.sleep(60)

@app.route('/get_current_price', methods=['GET'])
def get_current_price():
    symbol = request.args.get('symbol')

    if not symbol or not validate_stock_symbol(symbol):
        return jsonify({"error": "Invalid stock symbol"}), 400

    cached_data = stock_cache.get(symbol)
    if cached_data and (datetime.now() - datetime.fromisoformat(cached_data['timestamp'])).total_seconds() < 60:
        return jsonify(cached_data)

    data = fetch_stock_data(symbol)
    if data:
        stock_cache[symbol] = data
        return jsonify(data)
    else:
        return jsonify({"error": "Unable to fetch stock data"}), 500

@app.route('/get_price_and_predict', methods=['POST'])
def get_price_and_predict():
    try:
        symbol = request.form.get('symbol', '').upper()

        if not validate_stock_symbol(symbol):
            raise ValueError('Invalid stock symbol format')

        ticker = yf.Ticker(symbol)
        data = ticker.history(period="5y")

        if data.empty:
            raise ValueError(f'No data found for the symbol: {symbol}')

        now = datetime.now(pytz.timezone('US/Eastern'))
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        is_weekday = now.weekday() < 5

        if is_weekday and market_open <= now <= market_close:
            market_status = "open"
        else:
            market_status = "closed"

        current_price = data['Close'].iloc[-1]

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

        time_steps = 60
        X, y = prepare_data(scaled_data, time_steps)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = create_model((time_steps, 1))
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping], verbose=0)

        last_sequence = scaled_data[-time_steps:].reshape(1, -1, 1)
        predicted_price = model.predict(last_sequence)
        predicted_price = scaler.inverse_transform(predicted_price)[0][0]

        profit_percentage = ((predicted_price - current_price) / current_price) * 100

        if profit_percentage > 2:
            recommendation = "BUY"
        elif profit_percentage < -2:
            recommendation = "SELL"
        else:
            recommendation = "HOLD"

        return jsonify({
            'symbol': symbol,
            'current_price': float(current_price),
            'predicted_price': float(predicted_price),
            'profit_percentage': float(profit_percentage),
            'recommendation': recommendation,
            'market_status': market_status,
            'last_trading_time': data.index[-1].strftime('%Y-%m-%d %H:%M:%S %Z')
        })

    except Exception as e:
        logging.error(f"Error processing request: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    executor.submit(update_stock_data)
    app.run(debug=False)
