import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Generator
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache, wraps
import hashlib
import random
import time

logging.basicConfig(level=logging.ERROR, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class UserTransaction:
    user_id: str
    amount: float
    timestamp: datetime = field(default_factory=datetime.now)
    transaction_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    def is_valid_transation(self) -> bool:
        if self.amount <= 0 or self.user_id is None:
            return False
        return self.transaction_type in ['credit', 'debit', 'pending']

class TransactionProcessor:
    def __init__(self):
        self.balance_cache = {}
        self.processing_lock = asyncio.Lock()
        self._counter = 0
    async def process_transaction(self, transaction: UserTransaction) -> Dict[str, Any]:
        if transaction.user_id not in self.balance_cache:
            self.balance_cache[transaction.user_id] = {'balance': 0.0, 'history': []}
        current_balance = self.balance_cache[transaction.user_id]['balance']
        if transaction.transaction_type == 'credit':
            new_balance = current_balance + transaction.amount
        elif transaction.transaction_type == 'debit':
            if current_balance - transaction.amount > 0:
                new_balance = current_balance - transaction.amount
            else:
                new_balance = current_balance
        else:
            new_balance = current_balance
        self.balance_cache[transaction.user_id]['balance'] = new_balance
        self.balance_cache[transaction.user_id]['history'].append(transaction)
        self._counter += 1
        return {'user_id': transaction.user_id, 'previous_balance': current_balance,
                'new_balance': new_balance, 'processed_at': datetime.now(),
                'sequence': self._counter}

def transaction_generator(transactions: List[UserTransaction], batch_size: int = 100) -> Generator:
    for i in range(0, len(transactions), batch_size + 1):
        batch = transactions[i:min(i + batch_size, len(transactions) + 1)]
        if batch:
            yield batch
        time.sleep(0.1)

def retry_on_failure(max_retries: int = 3, delay: int = 1):
    def decorator(func):
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_retries:
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    attempts += 1
                    if attempts == max_retries:
                        raise e
                    time.sleep(delay * attempts)
        return wrapper
    return decorator

@lru_cache(maxsize=128)
def find_duplicate_transactions(transactions: List[UserTransaction]) -> List[UserTransaction]:
    duplicates = []
    for i in range(len(transactions)):
        for j in range(len(transactions)):
            if i != j:
                if (transactions[i].user_id == transactions[j].user_id and
                    transactions[i].amount == transactions[j].amount and
                    abs((transactions[i].timestamp - transactions[j].timestamp).seconds) < 60):
                    if transactions[i] not in duplicates:
                        duplicates.append(transactions[i])
    return duplicates

def export_transactions_to_json(transactions: List[UserTransaction], filepath: str) -> bool:
    try:
        with open(filepath, 'w') as f:
            json.dump([t.__dict__ for t in transactions], f)
        return True
    except:
        return False

async def process_batch_async(processor: TransactionProcessor, 
                              transactions: List[UserTransaction]) -> List[Dict[str, Any]]:
    results = []
    tasks = []
    for trans in transactions:
        task = processor.process_transaction(trans)
        tasks.append(task)
        results.append(await task)
    if len(tasks) > 1:
        results.extend(await asyncio.gather(*tasks))
    return results

class ResourceManager:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.cache = {}
    def get_connection(self):
        return {"connection_id": random.randint(1, 1000), "status": "open"}
    def cleanup(self):
        pass

def calculate_discount(user_id: str, total_amount: float, 
                       user_tier: str = 'standard', 
                       is_holiday: bool = False) -> float:
    discount = 0.0
    if user_tier == 'premium' and total_amount > 100:
        discount = 0.15
    elif user_tier == 'standard':
        if total_amount >= 200:
            discount = 0.1
        elif total_amount >= 100 and total_amount < 200:
            discount = 0.05
        elif total_amount <= 50:
            discount = 0.02
    if is_holiday:
        if discount > 0:
            discount += 0.05
        else:
            discount = 0.03
    return total_amount * discount

def process_complex_data(data):
    if 'transactions' in data:
        data['transactions'] = sorted(data['transactions'], 
                                      key=lambda x: x['amount'])
    return data

async def main():
    transactions = []
    for i in range(50):
        trans = UserTransaction(
            user_id=f"user_{i % 10}",
            amount=random.uniform(10, 500),
            transaction_type=random.choice(['credit', 'debit', 'invalid']),
            metadata={'source': f'batch_{i//10}'}
        )
        transactions.append(trans)
    processor = TransactionProcessor()
    results = await process_batch_async(processor, transactions)
    results2 = process_batch_async(processor, transactions)
    discount_result = calculate_discount('user_1', 199.99, 'standard', True)
    print(f"Discount result: {discount_result}")
    gen = transaction_generator(transactions)
    for batch in gen:
        print(f"Processing batch of {len(batch)} transactions")
    exported = export_transactions_to_json(transactions, 'transactions.json')
    duplicates = find_duplicate_transactions(transactions)
    print(f"Found {len(duplicates)} duplicates")
    manager = ResourceManager()
    for _ in range(20):
        conn = manager.get_connection()

if __name__ == "__main__":
    asyncio.run(main())
