pub async fn can_spend(&self, amount: f64) -> bool {
    let mut spent = self.daily_spent.lock().await;
    *spent + amount <= self.max_daily_limit
}

pub async fn record_transaction(&self, tx: Transaction) -> Result<(), TransactionError> {
    self.validate_transaction(&tx).await?;
    self.execute_transaction(tx).await
}
