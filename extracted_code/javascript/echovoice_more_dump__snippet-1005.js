// Initialize logging
tracing_subscriber::fmt::init();

let config = Config::load().await?;
let db_pool = DbPool::connect(&config.database_url).await?;

// Initialize AI services
let nlp_service = NlpService::new(config.bert_model_path);
let email_client = EmailClient::new(config.smtp_config);

// Create shared application state
let app_state = Arc::new(AppState {
    config,
    db_pool,
    nlp_service,
    email_client,
    opportunity_queue: Mutex::new(Vec::new()),
});

// Start background tasks
let state_clone = app_state.clone();
tokio::spawn(async move {
    OpportunityCrawler::new(state_clone)
        .run()
        .await
        .expect("Crawler failed");
});

// Start REST API
api::serve(app_state).await?;

Ok(())
