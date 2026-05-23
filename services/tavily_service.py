import os
import re
import logging
import requests
from typing import Dict
from typing import List
from typing import Optional
from datetime import datetime
from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tavily_service")


class TavilyService:
    """Tavily integration for real-time university information with domain filtering."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY environment variable not set")

        self.base_url = "https://api.tavily.com"

        self.pattern_domains = [
            "strathmore.edu",
        ]

        self.exact_domains = []

        self.search_cache = {}
        self.cache_ttl = 300

        logger.info("TavilyService initialized")
        logger.info(f"Pattern domains: {self.pattern_domains}")
        logger.info(f"Exact domains: {self.exact_domains}")

    def _is_domain_allowed(self, domain: str) -> bool:
        domain = domain.lower().strip()

        if domain in self.exact_domains:
            return True

        for pattern in self.pattern_domains:
            if domain == pattern or domain.endswith(f".{pattern}"):
                return True

        return False

    def _get_tavily_domains(self) -> List[str]:
        tavily_domains = []
        tavily_domains.extend(self.exact_domains)
        tavily_domains.extend(
            [f"*.{domain}" for domain in self.pattern_domains]
        )
        return tavily_domains

    def search(
        self,
        query: str,
        max_results: int = 2,
        search_depth: str = "basic",
        include_answer: bool = True,
        include_raw_content: bool = False,
        topic: str = "general",
    ) -> Dict:
        cache_key = f"{query}:{max_results}:{topic}"
        if self._is_cache_valid(cache_key):
            logger.info(f"Returning cached results for: {query}")
            return self.search_cache[cache_key]["data"]

        enhanced_query = self._enhance_query(query, topic)

        try:
            response = self._make_request(
                {
                    "query": enhanced_query,
                    "max_results": max_results,
                    "search_depth": search_depth,
                    "include_answer": include_answer,
                    "include_raw_content": include_raw_content,
                    "include_domains": self._get_tavily_domains(),
                }
            )

            if response.get("results"):
                filtered_results = self._filter_and_process_results(
                    response["results"], query
                )
                response["results"] = filtered_results
                response["filtered"] = True
                response["original_result_count"] = len(
                    response.get("results", [])
                )

            self.search_cache[cache_key] = {
                "data": response,
                "timestamp": datetime.now(),
            }

            logger.info(
                f"Found {len(response.get('results', []))} relevant results for: {query}"
            )
            return response

        except Exception as e:
            logger.error(f"Tavily search failed: {e}")
            return {
                "error": str(e),
                "results": [],
                "answer": f"Unable to fetch real-time information for: {query}",
            }

    def _enhance_query(self, query: str, topic: str) -> str:
        query_lower = query.lower()

        if "strathmore" not in query_lower:
            query = f"Strathmore University {query}"

        topic_enhancements = {
            "fees": "tuition fees payment schedule",
            "admission": "admission requirements entry qualifications",
            "schedule": "class timetable semester calendar",
            "events": "university events announcements",
            "facilities": "campus facilities services",
            "policies": "university policies regulations",
            "academic": "academic programs courses degrees",
        }

        if topic in topic_enhancements and topic not in query_lower:
            query += f" {topic_enhancements[topic]}"

        current_year = datetime.now().year
        if (
            str(current_year) not in query
            and str(current_year - 1) not in query
        ):
            query += f" {current_year}"

        return query

    def _make_request(self, payload: Dict) -> Dict:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        response = requests.post(
            f"{self.base_url}/search",
            json=payload,
            headers=headers,
            timeout=30,
        )

        if response.status_code != 200:
            raise Exception(
                f"Tavily API error: {response.status_code} - {response.text}"
            )

        return response.json()

    def _filter_and_process_results(
        self, results: List[Dict], original_query: str
    ) -> List[Dict]:
        if not results:
            return []

        filtered_results = []
        query_lower = original_query.lower()

        for result in results:
            url = result.get("url", "")
            domain = urlparse(url).netloc.lower()

            if not self._is_domain_allowed(domain):
                continue

            content = result.get("content", "").lower()
            title = result.get("title", "").lower()

            relevance_score = self._calculate_relevance_score(
                query_lower, title, content, url
            )

            if relevance_score > 0.3:
                result["relevance_score"] = relevance_score
                result["processed"] = True
                result["content"] = self._clean_content(
                    result.get("content", "")
                )
                filtered_results.append(result)

        filtered_results.sort(
            key=lambda x: x.get("relevance_score", 0), reverse=True
        )

        return filtered_results

    def _calculate_relevance_score(
        self, query: str, title: str, content: str, url: str
    ) -> float:
        score = 0.0

        query_terms = set(query.split())
        title_terms = set(title.split())
        title_overlap = len(query_terms.intersection(title_terms)) / len(
            query_terms
        )
        score += 0.4 * title_overlap

        content_words = set(content.split())
        content_overlap = len(query_terms.intersection(content_words)) / len(
            query_terms
        )
        score += 0.3 * content_overlap

        university_terms = [
            "strathmore",
            "university",
            "student",
            "academic",
            "campus",
        ]
        university_count = sum(
            1 for term in university_terms if term in content
        )
        score += 0.2 * min(university_count / len(university_terms), 1.0)

        recency_bonus = self._calculate_recency_bonus(content, url)
        score += 0.1 * recency_bonus

        return min(score, 1.0)

    def _calculate_recency_bonus(self, content: str, url: str) -> float:
        current_year = datetime.now().year

        year_pattern = r"\b(20\d{2})\b"
        years_found = re.findall(year_pattern, content + " " + url)

        if not years_found:
            return 0.0

        latest_year = max(int(year) for year in years_found)

        if latest_year == current_year:
            return 1.0
        elif latest_year == current_year - 1:
            return 0.7
        elif latest_year >= current_year - 2:
            return 0.4
        else:
            return 0.0

    def _clean_content(self, content: str) -> str:
        if not content:
            return ""

        content = re.sub(r"\s+", " ", content.strip())

        max_length = 500
        if len(content) > max_length:
            content = content[:max_length] + "..."

        return content

    def _is_cache_valid(self, cache_key: str) -> bool:
        if cache_key not in self.search_cache:
            return False

        cached_time = self.search_cache[cache_key]["timestamp"]
        return (datetime.now() - cached_time).seconds < self.cache_ttl

    def search_specific_topics(
        self,
        query: str,
        topics: List[str] = None,
        max_results_per_topic: int = 1,
    ) -> Dict[str, List[Dict]]:
        if not topics:
            topics = ["academic", "fees", "admission", "events", "policies"]

        results = {}

        for topic in topics:
            topic_query = f"{query} {topic}"
            topic_results = self.search(
                topic_query, max_results=max_results_per_topic, topic=topic
            )
            results[topic] = topic_results.get("results", [])

        return results

    def get_latest_announcements(self, max_results: int = 2) -> Dict:
        queries = [
            "Strathmore University latest announcements news",
            "Strathmore University important notices updates",
        ]

        all_results = []

        for query in queries:
            results = self.search(
                query,
                max_results=max_results,
                topic="events",
                search_depth="basic",
            )
            all_results.extend(results.get("results", []))

        unique_results = {result["url"]: result for result in all_results}
        sorted_results = sorted(
            unique_results.values(),
            key=lambda x: x.get("relevance_score", 0),
            reverse=True,
        )

        return {
            "results": sorted_results[:max_results],
            "query": "latest announcements",
            "timestamp": datetime.now().isoformat(),
        }

    def search_with_fallback(
        self, query: str, fallback_domains: List[str] = None
    ) -> Dict:
        results = self.search(query, max_results=2)

        if results.get("results") and len(results["results"]) > 0:
            results["used_fallback"] = False
            return results

        if fallback_domains:
            logger.info(
                "Primary search yielded no results, trying fallback domains"
            )

            original_pattern = self.pattern_domains.copy()
            original_exact = self.exact_domains.copy()

            for domain in fallback_domains:
                if domain.startswith("*."):
                    self.pattern_domains.append(domain[2:])
                else:
                    self.exact_domains.append(domain)

            try:
                fallback_results = self.search(query, max_results=2)
                fallback_results["used_fallback"] = True
                fallback_results["fallback_domains"] = fallback_domains
                return fallback_results
            finally:
                self.pattern_domains = original_pattern
                self.exact_domains = original_exact

        return results

    def clear_cache(self):
        self.search_cache.clear()
        logger.info("Tavily search cache cleared")

    def get_cache_stats(self) -> Dict:
        now = datetime.now()
        valid_entries = 0

        for cache_data in self.search_cache.values():
            if (now - cache_data["timestamp"]).seconds < self.cache_ttl:
                valid_entries += 1

        return {
            "total_cached_queries": len(self.search_cache),
            "valid_cached_entries": valid_entries,
            "cache_ttl_seconds": self.cache_ttl,
            "pattern_domains": self.pattern_domains,
            "exact_domains": self.exact_domains,
        }
