import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

# Load and prepare data
df = pd.read_csv('Task 3 and 4_Loan_Data.csv')
X = df.drop(['customer_id', 'default'], axis=1)
y = df['default']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features (for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
# Best performing model: Logistic Regression
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)

# Alternative models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
rf_model.fit(X_train, y_train)

gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5)
gb_model.fit(X_train, y_train)

dt_model = DecisionTreeClassifier(random_state=42, max_depth=10)
dt_model.fit(X_train, y_train)


# MAIN FUNCTION: Predict Expected Loss
def predict_expected_loss(loan_features, recovery_rate=0.10, model='logistic'):
    """
    Predict the expected loss for a loan given borrower characteristics.
    
    Parameters:
    -----------
    loan_features : dict or pandas DataFrame
        Dictionary or DataFrame containing loan features:
        - credit_lines_outstanding: Number of outstanding credit lines
        - loan_amt_outstanding: Current loan amount outstanding
        - total_debt_outstanding: Total debt across all sources
        - income: Annual income
        - years_employed: Years in current employment
        - fico_score: FICO credit score
    
    recovery_rate : float, default=0.10
        Expected recovery rate on defaulted loans (10% = 0.10)
    
    model : str, default='logistic'
        Model to use: 'logistic', 'random_forest', 'gradient_boosting', or 'decision_tree'
    
    Returns:
    --------
    dict containing:
        - probability_of_default: Predicted probability of default (PD)
        - loan_amount: The outstanding loan amount
        - loss_given_default: Loss amount if default occurs (LGD)
        - expected_loss: Expected loss (PD × LGD)
    """
    
    # Convert input to DataFrame if it's a dictionary
    if isinstance(loan_features, dict):
        loan_df = pd.DataFrame([loan_features])
    else:
        loan_df = loan_features.copy()
    
    # Ensure correct column order
    feature_order = [
        'credit_lines_outstanding', 
        'loan_amt_outstanding', 
        'total_debt_outstanding',
        'income', 
        'years_employed', 
        'fico_score'
    ]
    loan_df = loan_df[feature_order]
    
    # Get loan amount
    loan_amount = loan_df['loan_amt_outstanding'].values[0]
    
    # Select and apply model
    if model == 'logistic':
        loan_scaled = scaler.transform(loan_df)
        prob_default = lr_model.predict_proba(loan_scaled)[0, 1]
    elif model == 'random_forest':
        prob_default = rf_model.predict_proba(loan_df)[0, 1]
    elif model == 'gradient_boosting':
        prob_default = gb_model.predict_proba(loan_df)[0, 1]
    elif model == 'decision_tree':
        prob_default = dt_model.predict_proba(loan_df)[0, 1]
    else:
        raise ValueError("Model must be 'logistic', 'random_forest', 'gradient_boosting', or 'decision_tree'")
    
    # Calculate Loss Given Default (LGD)
    # LGD = Loan Amount × (1 - Recovery Rate)
    loss_given_default = loan_amount * (1 - recovery_rate)
    
    # Calculate Expected Loss
    # EL = PD × LGD
    expected_loss = prob_default * loss_given_default
    
    return {
        'probability_of_default': prob_default,
        'loan_amount': loan_amount,
        'loss_given_default': loss_given_default,
        'expected_loss': expected_loss
    }


# BATCH PREDICTION FUNCTION
def predict_portfolio_loss(loan_portfolio_df, recovery_rate=0.10, model='logistic'):
    """
    Predict expected loss for an entire portfolio of loans.
    
    Parameters:
    -----------
    loan_portfolio_df : pandas DataFrame
        DataFrame containing multiple loans with required features
    
    recovery_rate : float, default=0.10
        Expected recovery rate on defaulted loans
    
    model : str, default='logistic'
        Model to use for predictions
    
    Returns:
    --------
    pandas DataFrame with original features plus:
        - probability_of_default
        - loss_given_default
        - expected_loss
    """
    
    results = []
    
    for idx, row in loan_portfolio_df.iterrows():
        loan_dict = row.to_dict()
        prediction = predict_expected_loss(loan_dict, recovery_rate, model)
        results.append(prediction)
    
    results_df = pd.DataFrame(results)
    final_df = pd.concat([loan_portfolio_df.reset_index(drop=True), results_df], axis=1)
    
    return final_df


# EXAMPLE USAGE
print("\n" + "="*70)
print("EXAMPLE 1: Single Loan Prediction")
print("="*70)

# Example borrower with high-risk characteristics
high_risk_borrower = {
    'credit_lines_outstanding': 5,
    'loan_amt_outstanding': 5000.0,
    'total_debt_outstanding': 25000.0,
    'income': 45000.0,
    'years_employed': 1,
    'fico_score': 550
}

result = predict_expected_loss(high_risk_borrower, recovery_rate=0.10, model='logistic')
print(f"\nHigh-Risk Borrower:")
print(f"  Probability of Default: {result['probability_of_default']:.4f} ({result['probability_of_default']*100:.2f}%)")
print(f"  Loan Amount: ${result['loan_amount']:,.2f}")
print(f"  Loss Given Default: ${result['loss_given_default']:,.2f}")
print(f"  Expected Loss: ${result['expected_loss']:,.2f}")

# Example borrower with low-risk characteristics
low_risk_borrower = {
    'credit_lines_outstanding': 1,
    'loan_amt_outstanding': 5000.0,
    'total_debt_outstanding': 8000.0,
    'income': 90000.0,
    'years_employed': 7,
    'fico_score': 750
}

result = predict_expected_loss(low_risk_borrower, recovery_rate=0.10, model='logistic')
print(f"\nLow-Risk Borrower:")
print(f"  Probability of Default: {result['probability_of_default']:.4f} ({result['probability_of_default']*100:.2f}%)")
print(f"  Loan Amount: ${result['loan_amount']:,.2f}")
print(f"  Loss Given Default: ${result['loss_given_default']:,.2f}")
print(f"  Expected Loss: ${result['expected_loss']:,.2f}")

print("\n" + "="*70)
print("EXAMPLE 2: Portfolio-Level Analysis")
print("="*70)

# Predict expected loss for test set
portfolio_predictions = predict_portfolio_loss(X_test, recovery_rate=0.10, model='logistic')

# Calculate portfolio statistics
total_exposure = portfolio_predictions['loan_amount'].sum()
total_expected_loss = portfolio_predictions['expected_loss'].sum()
avg_pd = portfolio_predictions['probability_of_default'].mean()

print(f"\nPortfolio Statistics (Test Set - 2,000 loans):")
print(f"  Total Loan Exposure: ${total_exposure:,.2f}")
print(f"  Total Expected Loss: ${total_expected_loss:,.2f}")
print(f"  Expected Loss Rate: {(total_expected_loss/total_exposure)*100:.4f}%")
print(f"  Average Probability of Default: {avg_pd:.4f} ({avg_pd*100:.2f}%)")
print(f"  Capital Required (at 10% recovery): ${total_expected_loss:,.2f}")

# Save predictions to CSV
portfolio_predictions.to_csv('loan_portfolio_predictions.csv', index=False)
print(f"\n✓ Portfolio predictions saved to 'loan_portfolio_predictions.csv'")
