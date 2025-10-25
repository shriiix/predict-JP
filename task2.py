import pandas as pd
import numpy as np
from datetime import datetime

def price_storage_contract(injection_dates, withdrawal_dates, prices_dict, 
                           injection_rate, withdrawal_rate, max_storage_volume, 
                           storage_cost_per_unit):
    """
    Price a natural gas storage contract.
    
    Parameters:
    -----------
    injection_dates : list of str
        Dates when gas is injected (format: 'MM/DD/YY' or datetime)
    withdrawal_dates : list of str
        Dates when gas is withdrawn (format: 'MM/DD/YY' or datetime)
    prices_dict : dict
        Dictionary mapping dates to gas prices {date: price}
    injection_rate : float
        Volume of gas injected per injection date (same units as max_storage_volume)
    withdrawal_rate : float
        Volume of gas withdrawn per withdrawal date (same units as max_storage_volume)
    max_storage_volume : float
        Maximum storage capacity
    storage_cost_per_unit : float
        Cost per unit of gas stored (applied for each period gas is in storage)
    
    Returns:
    --------
    dict : Contains contract value and detailed breakdown of cash flows
    """
    
   
    current_volume = 0
    total_injection_cost = 0
    total_withdrawal_revenue = 0
    total_storage_cost = 0
    cash_flows = []
    
  
    all_transactions = []
    
    for date in injection_dates:
        all_transactions.append({
            'date': pd.to_datetime(date),
            'type': 'injection',
            'volume': injection_rate
        })
    
    for date in withdrawal_dates:
        all_transactions.append({
            'date': pd.to_datetime(date),
            'type': 'withdrawal',
            'volume': withdrawal_rate
        })
    

    all_transactions.sort(key=lambda x: x['date'])
    
  
    for i, transaction in enumerate(all_transactions):
        date = transaction['date']
        trans_type = transaction['type']
        volume = transaction['volume']
        

        date_str = date.strftime('%-m/%-d/%y') 
        if date_str not in prices_dict:
            raise ValueError(f"Price not available for date {date_str}")
        
        price = prices_dict[date_str]
        
        if trans_type == 'injection':
            
            if current_volume + volume > max_storage_volume:
                raise ValueError(f"Storage capacity exceeded on {date_str}. "
                               f"Current: {current_volume}, Injecting: {volume}, "
                               f"Max: {max_storage_volume}")
            
         
            injection_cost = volume * price
            total_injection_cost += injection_cost
            current_volume += volume
            
            cash_flows.append({
                'date': date_str,
                'type': 'injection',
                'volume': volume,
                'price': price,
                'cash_flow': -injection_cost,
                'storage_level': current_volume
            })
            # Withdrawal: sell gas (cash inflow)
        else: 
            
            if current_volume < volume:
                raise ValueError(f"Insufficient volume to withdraw on {date_str}. "
                               f"Current: {current_volume}, Withdrawing: {volume}")
            
       
            withdrawal_revenue = volume * price
            total_withdrawal_revenue += withdrawal_revenue
            current_volume -= volume
            
            cash_flows.append({
                'date': date_str,
                'type': 'withdrawal',
                'volume': volume,
                'price': price,
                'cash_flow': withdrawal_revenue,
                'storage_level': current_volume
            })
    

    for i in range(len(all_transactions) - 1):
        volume_stored = cash_flows[i]['storage_level']
        if volume_stored > 0:
          
            days = (all_transactions[i+1]['date'] - all_transactions[i]['date']).days
           
            period_storage_cost = volume_stored * storage_cost_per_unit * (days / 30)  
            total_storage_cost += period_storage_cost
    

    contract_value = total_withdrawal_revenue - total_injection_cost - total_storage_cost
    

    result = {
        'contract_value': contract_value,
        'total_withdrawal_revenue': total_withdrawal_revenue,
        'total_injection_cost': total_injection_cost,
        'total_storage_cost': total_storage_cost,
        'net_before_storage': total_withdrawal_revenue - total_injection_cost,
        'final_storage_level': current_volume,
        'cash_flows': pd.DataFrame(cash_flows)
    }
    
    return result



def load_price_data(filepath='Nat_Gas.csv'):
    """Load price data from CSV file."""
    df = pd.read_csv(filepath)
    
    
    df['Prices'] = df['Prices'].astype(float)
    
  
    prices_dict = {}
    for _, row in df.iterrows():
        date = pd.to_datetime(row['Dates'])
        date_key = date.strftime('%-m/%-d/%y')  
        prices_dict[date_key] = row['Prices']
    
    return prices_dict



if __name__ == "__main__":
  
    prices = load_price_data('Nat_Gas.csv')
    
    print("=" * 70)
    print("GAS STORAGE CONTRACT PRICING MODEL - TEST CASES")
    print("=" * 70)
    
  
    print("\n" + "=" * 70)
    print("TEST CASE 1: Simple Arbitrage Strategy")
    print("=" * 70)
    print("Strategy: Inject in summer (low prices), withdraw in winter (high prices)")
    
    result1 = price_storage_contract(
        injection_dates=['5/31/21', '6/30/21'],      
        withdrawal_dates=['12/31/21', '1/31/22'],    
        prices_dict=prices,
        injection_rate=50000,                     
        withdrawal_rate=50000,                       
        max_storage_volume=100000,                   
        storage_cost_per_unit=0.02                    
    )
    
    print(f"\nContract Value: ${result1['contract_value']:,.2f}")
    print(f"Total Revenue (Withdrawals): ${result1['total_withdrawal_revenue']:,.2f}")
    print(f"Total Cost (Injections): ${result1['total_injection_cost']:,.2f}")
    print(f"Storage Costs: ${result1['total_storage_cost']:,.2f}")
    print(f"Net Before Storage: ${result1['net_before_storage']:,.2f}")
    print(f"Final Storage Level: {result1['final_storage_level']:,.0f} units")
    print("\nDetailed Cash Flows:")
    print(result1['cash_flows'].to_string(index=False))
    
    
    print("\n" + "=" * 70)
    print("TEST CASE 2: Multiple Trading Cycles")
    print("=" * 70)
    print("Strategy: Multiple injection/withdrawal cycles over 2 years")
    
    result2 = price_storage_contract(
        injection_dates=['4/30/21', '5/31/21', '6/30/21'],  
        withdrawal_dates=['12/31/21', '1/31/22', '2/28/22'], 
        prices_dict=prices,
        injection_rate=30000,
        withdrawal_rate=30000,
        max_storage_volume=100000,
        storage_cost_per_unit=0.015
    )
    
    print(f"\nContract Value: ${result2['contract_value']:,.2f}")
    print(f"Total Revenue (Withdrawals): ${result2['total_withdrawal_revenue']:,.2f}")
    print(f"Total Cost (Injections): ${result2['total_injection_cost']:,.2f}")
    print(f"Storage Costs: ${result2['total_storage_cost']:,.2f}")
    print(f"Net Before Storage: ${result2['net_before_storage']:,.2f}")
    print("\nDetailed Cash Flows:")
    print(result2['cash_flows'].to_string(index=False))
    
 
    print("\n" + "=" * 70)
    print("TEST CASE 3: Long-term Storage Strategy")
    print("=" * 70)
    print("Strategy: Inject during price dip in 2021, hold until price spike in 2023")
    
    result3 = price_storage_contract(
        injection_dates=['5/31/21'],                  
        withdrawal_dates=['12/31/23'],                
        prices_dict=prices,
        injection_rate=80000,
        withdrawal_rate=80000,
        max_storage_volume=100000,
        storage_cost_per_unit=0.025
    )
    
    print(f"\nContract Value: ${result3['contract_value']:,.2f}")
    print(f"Total Revenue (Withdrawals): ${result3['total_withdrawal_revenue']:,.2f}")
    print(f"Total Cost (Injections): ${result3['total_injection_cost']:,.2f}")
    print(f"Storage Costs: ${result3['total_storage_cost']:,.2f}")
    print(f"Net Before Storage: ${result3['net_before_storage']:,.2f}")
    print("\nDetailed Cash Flows:")
    print(result3['cash_flows'].to_string(index=False))
    
    print("\n" + "=" * 70)
    print("Analysis Complete")
    print("=" * 70)