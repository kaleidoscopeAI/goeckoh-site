task::block_on(async {
    let mut crawler = WebCrawler::new("https://rust-lang.org", 2);
    crawler.crawl().await;
});
