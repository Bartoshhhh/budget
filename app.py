import os
from datetime import datetime
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

app = Flask(__name__)

# Konfiguracja bazy danych SQLite
base_dir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(base_dir, 'budzet.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Model bazy danych - Tabela Transakcji
class Transaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    category = db.Column(db.String(50), nullable=False)
    type = db.Column(db.String(20), nullable=False)  # 'wydatek' lub 'przychod'
    date = db.Column(db.Date, nullable=False, default=datetime.utcnow)

    def __repr__(self):
        return f'<Transaction {self.title} - {self.amount} PLN>'

with app.app_context():
    db.create_all()

# Funkcja pomocnicza: Analiza szeregów czasowych i prognozowanie wydatków
def calculate_predictions(transactions):
    expenses = [t for t in transactions if t.type == 'wydatek']
    
    if not expenses:
        return None

    data = [{'date': t.date, 'amount': t.amount} for t in expenses]
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    
    df['month'] = df['date'].dt.to_period('M')
    monthly_expenses = df.groupby('month')['amount'].sum().reset_index()
    monthly_expenses['month_str'] = monthly_expenses['month'].astype(str)
    
    months_count = len(monthly_expenses)
    if months_count < 2:
        return {
            'status': 'insufficient_data',
            'message': 'Wymagane są transakcje z co najmniej 2 różnych miesięcy, aby wygenerować prognozę.'
        }

    X = np.array(range(months_count)).reshape(-1, 1)
    y = monthly_expenses['amount'].values

    model = LinearRegression()
    model.fit(X, y)

    next_month_index = np.array([[months_count]])
    predicted_next_month = float(model.predict(next_month_index)[0])
    predicted_next_month = max(0, predicted_next_month)

    moving_average_pred = float(np.mean(y[-3:]))

    y_pred_history = model.predict(X)
    mae_score = float(mean_absolute_error(y, y_pred_history))

    last_period = monthly_expenses['month'].iloc[-1]
    next_period_str = str(last_period + 1)

    labels = monthly_expenses['month_str'].tolist() + [f"{next_period_str} (PROGNOZA)"]
    historical_values = monthly_expenses['amount'].tolist() + [None]
    regression_trend = y_pred_history.tolist() + [predicted_next_month]

    return {
        'status': 'success',
        'predicted_value': round(predicted_next_month, 2),
        'moving_average_value': round(moving_average_pred, 2),
        'mae_error': round(mae_score, 2),
        'labels': labels,
        'historical_values': historical_values,
        'regression_trend': [round(v, 2) for v in regression_trend],
        'months_count': months_count
    }

# Strona główna
@app.route('/', methods=['GET'])
def index():
    transactions = Transaction.query.order_by(Transaction.date.asc(), Transaction.id.asc()).all()
    
    total_income = sum(t.amount for t in transactions if t.type == 'przychod')
    total_expenses = sum(t.amount for t in transactions if t.type == 'wydatek')
    balance = total_income - total_expenses

    expenses_by_category = {}
    for t in transactions:
        if t.type == 'wydatek':
            expenses_by_category[t.category] = expenses_by_category.get(t.category, 0.0) + t.amount

    category_labels = list(expenses_by_category.keys())
    category_data = list(expenses_by_category.values())

    prediction_results = calculate_predictions(transactions)

    transactions_reversed = list(reversed(transactions))

    return render_template('index.html', 
                           transactions=transactions_reversed, 
                           total_income=total_income, 
                           total_expenses=total_expenses, 
                           balance=balance,
                           category_labels=category_labels,
                           category_data=category_data,
                           prediction=prediction_results)

# Dodawanie transakcji
@app.route('/add', methods=['POST'])
def add_transaction():
    title = request.form.get('title')
    amount = float(request.form.get('amount'))
    category = request.form.get('category')
    trans_type = request.form.get('type')
    date_str = request.form.get('date')

    trans_date = datetime.strptime(date_str, '%Y-%m-%d').date() if date_str else datetime.utcnow().date()

    new_transaction = Transaction(
        title=title,
        amount=amount,
        category=category,
        type=trans_type,
        date=trans_date
    )

    db.session.add(new_transaction)
    db.session.commit()

    return redirect(url_for('index'))

# Usuwanie transakcji
@app.route('/delete/<int:id>', methods=['POST'])
def delete_transaction(id):
    transaction = Transaction.query.get_or_404(id)
    db.session.delete(transaction)
    db.session.commit()
    return redirect(url_for('index'))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
