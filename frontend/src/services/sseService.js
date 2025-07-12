class CrawlEventService {
  constructor(onEvent, onOpen, onError) {
    this.eventSource = null;
    this.onEvent = onEvent;
    this.onOpen = onOpen || (() => {});
    this.onError = onError || (() => {});
    this.processedIds = new Set();
  }

  connect() {
    if (this.eventSource) this.disconnect();
    this.eventSource = new EventSource(
      "http://localhost:8001/api/crawl-events"
    );
    this.eventSource.onopen = () => this.onOpen();
    this.eventSource.onerror = (err) => this.onError(err);

    const eventTypes = [
      "search_started",
      "urls_found",
      "scraping_url",
      "url_scraped",
      "scraping_complete",
    ];
    eventTypes.forEach((type) => {
      this.eventSource.addEventListener(type, (e) => {
        try {
          const eventData = JSON.parse(e.data);
          if (!eventData.id) return;
          if (!this.processedIds.has(eventData.id)) {
            this.processedIds.add(eventData.id);
            this.onEvent(eventData);
          }
        } catch (err) {
          console.error("Error parsing event data:", err);
        }
      });
    });
  }

  disconnect() {
    if (this.eventSource) {
      this.eventSource.close();
      this.eventSource = null;
      this.processedIds.clear();
    }
  }
}

export default CrawlEventService;
