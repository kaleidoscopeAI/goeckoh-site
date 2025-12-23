// Shared state wrapper to allow safe access from multiple threads
inner_state: Arc<Mutex<InnerState>>,
