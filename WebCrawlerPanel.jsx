import React from 'react';

const WebCrawlerPanel = ({ webCrawler }) => {
  return (
    <div className="absolute top-4 right-4 bg-gray-800/50 p-4 rounded-lg text-white font-mono text-sm">
      <h2 className="font-bold text-lg mb-2">Web Crawler</h2>
      <p>Status: {webCrawler.status}</p>
      <p>URLs in Queue: {webCrawler.queueSize}</p>
      <p>Last URL Crawled: {webCrawler.lastUrl}</p>
    </div>
  );
};

export default WebCrawlerPanel;
