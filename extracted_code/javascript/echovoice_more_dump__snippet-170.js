    pub async fn request_approval(&self, action: Action) -> Result<ApprovalResult, ApprovalError> {
        let approval_id = self.store_action(action).await?;
        self.notify_human(approval_id).await?;
        self.wait_for_response(approval_id).await
    }
