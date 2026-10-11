"""
Keyword Extraction and Content Enrichment Module

Extracts keywords and enriches content before embedding to improve search accuracy.
"""

import re
import logging
from typing import List, Set, Tuple, Optional
from collections import Counter

logger = logging.getLogger(__name__)

try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False

_CJK_RE = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf]')

CHINESE_STOP_WORDS = {
    '的', '了', '是', '在', '我', '你', '他', '她', '它', '我们', '你们', '他们',
    '这', '那', '有', '和', '与', '就', '都', '也', '还', '吗', '呢', '吧', '啊',
    '去', '来', '什么', '怎么', '哪里', '哪儿', '哪', '哪个', '多少', '为何', '为什么'
}
SINGLE_CHAR_STOP_WORDS = {c for c in CHINESE_STOP_WORDS if len(c) == 1}


def contains_cjk(text: str) -> bool:
    """Return True if the text contains CJK characters."""
    return bool(text and _CJK_RE.search(text))


def tokenize_text(text: str, max_tokens: int = 128) -> List[str]:
    """Tokenize text for keyword indexing and query matching.

    Uses jieba when available. Otherwise falls back to ASCII words plus
    sliding CJK character bigrams so Chinese text still gets searchable keys.
    """
    text = (text or '').strip().lower()
    if not text:
        return []

    if JIEBA_AVAILABLE:
        tokens = [t.strip().lower() for t in jieba.cut(text) if t.strip()]
        chars = _CJK_RE.findall(text)
        if chars:
            for i in range(len(chars) - 1):
                bigram = chars[i] + chars[i + 1]
                if bigram in CHINESE_STOP_WORDS:
                    continue
                if chars[i] in SINGLE_CHAR_STOP_WORDS or chars[i + 1] in SINGLE_CHAR_STOP_WORDS:
                    continue
                tokens.append(bigram)
        return [t for t in tokens if t not in CHINESE_STOP_WORDS and t not in SINGLE_CHAR_STOP_WORDS][:max_tokens]

    tokens = []
    for raw in re.split(r'[^\w\u4e00-\u9fff]+', text):
        if not raw:
            continue
        if _CJK_RE.search(raw):
            chars = list(raw)
            tokens.extend(c for c in chars if c not in SINGLE_CHAR_STOP_WORDS)
            for i in range(len(chars) - 1):
                bigram = chars[i] + chars[i + 1]
                if bigram in CHINESE_STOP_WORDS:
                    continue
                if chars[i] in SINGLE_CHAR_STOP_WORDS or chars[i + 1] in SINGLE_CHAR_STOP_WORDS:
                    continue
                tokens.append(bigram)
        else:
            tokens.append(raw)
    return [t for t in tokens if t not in CHINESE_STOP_WORDS and t not in SINGLE_CHAR_STOP_WORDS][:max_tokens]


class KeywordEnricher:
    """Extract keywords and enrich content for better embeddings."""

    def __init__(self):
        """Initialize keyword enricher."""
        self.stop_words = {
            'the', 'a', 'an', 'is', 'was', 'are', 'were', 'been', 'be', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
            'can', 'shall', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
            'but', 'or', 'and', 'if', 'so', 'yet', 'it', 'this', 'that', 'these', 'those',
            'i', 'you', 'he', 'she', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
            'my', 'your', 'his', 'her', 'its', 'our', 'their', 'what', 'which', 'who',
            'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
            'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
            'than', 'too', 'very', 'just', 'about', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'between', 'under', 'again', 'further', 'then', 'once'
        }

    def extract_keywords(self, text: str, max_keywords: int = 15) -> List[str]:
        """
        Extract important keywords from text.

        Args:
            text: Input text
            max_keywords: Maximum number of keywords to extract

        Returns:
            List of extracted keywords
        """
        if not text:
            return []

        keywords = []
        text_lower = text.lower()

        names = re.findall(r'(?<![.!?]\s)\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        for name in names[:5]:
            name_lower = name.lower()
            if name_lower not in self.stop_words:
                keywords.append(name_lower)

        years = re.findall(r'\b(19|20)\d{2}\b', text)
        keywords.extend(years[:2])

        months = re.findall(r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\b', text, re.IGNORECASE)
        keywords.extend([m.lower() for m in months[:2]])

        dates = re.findall(r'\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b', text)
        keywords.extend(dates[:2])

        times = re.findall(r'\b\d{1,2}:\d{2}(?:\s*[apAP][mM])?\b', text)
        keywords.extend(times[:2])

        words = re.findall(r'\b[a-zA-Z]+\b', text_lower)

        content_words = [w for w in words if len(w) > 2 and w not in self.stop_words]

        word_freq = Counter(content_words)

        for word, _ in word_freq.most_common(20):
            if word not in keywords and len(keywords) < max_keywords:
                keywords.append(word)

        bigrams = self.extract_bigrams(text_lower)
        for bigram in bigrams[:3]:
            if len(keywords) < max_keywords:
                keywords.append(bigram)

        seen = set()
        unique_keywords = []
        for kw in keywords:
            kw_lower = kw.lower()
            if kw_lower not in seen:
                seen.add(kw_lower)
                unique_keywords.append(kw_lower)

        return unique_keywords[:max_keywords]

    def extract_bigrams(self, text: str, top_n: int = 5) -> List[str]:
        """
        Extract important two-word phrases.

        Args:
            text: Input text
            top_n: Number of bigrams to return

        Returns:
            List of bigrams (connected with underscore)
        """
        words = re.findall(r'\b[a-z]+\b', text.lower())
        bigrams = []

        for i in range(len(words) - 1):
            if (words[i] not in self.stop_words and
                words[i+1] not in self.stop_words and
                len(words[i]) > 2 and len(words[i+1]) > 2):
                bigrams.append(f"{words[i]}_{words[i+1]}")

        if not bigrams:
            return []

        bigram_freq = Counter(bigrams)
        return [bg for bg, _ in bigram_freq.most_common(top_n)]

    def enrich_content(self, content: str, metadata: Optional[dict] = None) -> str:
        """
        Enrich content with extracted keywords for better embedding.

        Args:
            content: Original content
            metadata: Optional metadata with pre-extracted information

        Returns:
            Enriched content string
        """
        if not content:
            return content

        keywords = self.extract_keywords(content)

        if metadata:
            if 'entities' in metadata and metadata['entities']:
                for entity in metadata.get('entities', [])[:5]:
                    entity_lower = str(entity).lower()
                    if entity_lower not in keywords and entity_lower not in self.stop_words:
                        keywords.append(entity_lower)

            if 'topic' in metadata and metadata['topic']:
                topic_words = str(metadata['topic']).lower().split()
                for word in topic_words:
                    if word not in self.stop_words and word not in keywords and len(word) > 2:
                        keywords.append(word)

            if 'speaker' in metadata and metadata['speaker']:
                speaker = str(metadata['speaker']).lower()
                if speaker not in keywords and speaker not in self.stop_words:
                    keywords.append(speaker)

            if 'semantic_facts' in metadata and metadata['semantic_facts']:
                for fact in metadata.get('semantic_facts', [])[:3]:
                    fact_words = str(fact).lower().split()
                    for word in fact_words[:2]:
                        if word not in self.stop_words and word not in keywords and len(word) > 2:
                            keywords.append(word)

        if keywords:
            keyword_str = ' '.join(keywords[:15])
            enriched = f"{content} [KEYWORDS: {keyword_str}]"
        else:
            enriched = content

        return enriched

    def enrich_query(self, query: str) -> str:
        """
        Enrich a query with extracted keywords for better search.

        Args:
            query: Original query

        Returns:
            Enriched query string
        """
        if not query:
            return query

        keywords = self.extract_keywords(query, max_keywords=10)

        query_lower = query.lower()
        question_type = None
        for qw in ['who', 'what', 'where', 'when', 'why', 'how', 'which']:
            if query_lower.startswith(qw):
                question_type = qw
                break

        entities = re.findall(r'\b[A-Z][a-z]+\b', query)
        entities = [e.lower() for e in entities if e.lower() not in self.stop_words]

        enriched_parts = [query]

        if keywords:
            important_keywords = []

            for entity in entities:
                if entity in keywords:
                    important_keywords.append(entity)

            for kw in keywords:
                if kw not in important_keywords:
                    important_keywords.append(kw)

            if important_keywords:
                enriched_parts.append(' '.join(important_keywords[:8]))

        return ' '.join(enriched_parts)

def test_enrichment():
    """Test the keyword enrichment functionality."""
    enricher = KeywordEnricher()

    print("="*60)
    print("Testing Keyword Enrichment")
    print("="*60)

    # Test content enrichment
    test_content = """
    Caroline mentioned that she's been researching adoption agencies
    because she and her husband are considering adoption. The process
    started in January 2023 and they visited three agencies in Boston.
    """

    print("\n1. Content Enrichment Test:")
    print("-" * 40)
    print("Original content:")
    print(test_content.strip())

    enriched = enricher.enrich_content(test_content)
    print("\nEnriched content:")
    print(enriched)

    # Extract just keywords
    keywords = enricher.extract_keywords(test_content)
    print(f"\nExtracted keywords: {keywords}")

    # Test with metadata
    test_metadata = {
        'speaker': 'Caroline',
        'entities': ['Caroline', 'Boston', 'adoption agencies'],
        'topic': 'adoption research',
        'semantic_facts': ['researching adoption', 'visiting agencies']
    }

    enriched_with_meta = enricher.enrich_content(test_content, test_metadata)
    print("\nEnriched with metadata:")
    print(enriched_with_meta)

    # Test query enrichment
    print("\n2. Query Enrichment Test:")
    print("-" * 40)

    test_queries = [
        "What did Caroline research about adoption?",
        "When did the charity race happen in October?",
        "Who is Melanie's best friend?",
        "How long did Jon stay in Boston?",
        "Where did they move from in 2023?"
    ]

    for query in test_queries:
        enriched_query = enricher.enrich_query(query)
        print(f"\nOriginal: {query}")
        print(f"Enriched: {enriched_query}")
        keywords = enricher.extract_keywords(query)
        print(f"Keywords: {keywords}")

if __name__ == "__main__":
    test_enrichment()
