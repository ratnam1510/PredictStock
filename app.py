import logging
from flask import Flask, render_template, request, redirect, url_for, jsonify, session, flash
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import keras as keras
from keras.src.models import Sequential
from keras.src.layers import LSTM, Dense, Dropout
import yfinance as yf
import re
from datetime import datetime, timedelta
import pytz
import threading
import time
from flask_mail import Mail, Message
import pyotp
import qrcode
from io import BytesIO
import base64
import random
import string
