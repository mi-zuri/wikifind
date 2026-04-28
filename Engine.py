import gc

import math
import ssl
import logging
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re
import time
import os
from urllib.parse import urljoin
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import networkx as nx
import plotly.graph_objects as go

from sklearn.manifold import TSNE

DATABASE_FILENAME_PREFIX = "wiki_db_"

np.random.seed(42)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

class Engine:
    def __init__(self):
        self.raw_data = pd.DataFrame(columns=['title', 'url', 'text', 'categories', 'cleaned_text'])
        self.vectorizer = None
        self.tfidf_matrix = None
        self.reduced_matrix = None
        self.lsa_model = None
        self.feature_names = None
        self.history = []
        self.embedding_2d = None

        self.fetch_attempts = 0
        self.successful_fetches = 0
        self.duplicate_fetches = 0

        self.stop_words = list(stopwords.words('english'))
        self.stop_words += ['wikipedia', 'edit', 'retrieved', 'archived', 'original', 'article', 'reference', 'isbn', 'external',
                            'links'] # chat proposed

        self.lemmatizer = WordNetLemmatizer()
        self.headers = {'User-Agent': 'WikiRecommenderBot/1.0 (Educational Project)'}
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def _scrape_single_page(self, url):
        self.fetch_attempts += 1

        try:
            res = self.session.get(url, timeout=8)
            if res.status_code != 200:
                logger.warning(f"Skipping: {url:<90} ||| Request failed with status code: {res.status_code}")
                return None

            soup = BeautifulSoup(res.text, 'html.parser')

            title_tag = soup.find(id="firstHeading")
            if not title_tag:
                logger.warning(f"Skipping: {url:<90} ||| TITLE TAG NOT FOUND (#firstHeading)")
                return None
            title = title_tag.text.strip()

            content_div = soup.select_one("div.mw-content-ltr.mw-parser-output")
            if not content_div:
                logger.warning(f"Skipping: {url:<90} ||| CONTENT DIV NOT FOUND (div.mw-content-ltr.mw-parser-output)")
                return None

            # Issue: could also be <ul>
            paragraphs = content_div.find_all('p')
            text_parts = []
            for p in paragraphs:
                text = p.get_text().strip()
                text = re.sub(r'\[\d+]|\[citation needed]', '', text, flags=re.IGNORECASE)
                if len(text) > 30:
                    text_parts.append(text)

            full_text = " ".join(text_parts)

            if len(full_text) < 200:
                logger.warning(f"Skipping: {url:<90} ||| NOT ENOUGH TEXT ({len(full_text)})")
                return None

            # Paragraph++ link search
            links = []
            # for a in content_div.find_all('a', href=True):
            #     href = a['href']
            #     if href.startswith('/wiki/') and ':' not in href and not href.startswith('/wiki/Special:'):
            #         links.append(urljoin("https://en.wikipedia.org/", href))

            for p in paragraphs:
                for a in p.find_all('a', href=True):
                    href = a['href']
                    if href.startswith('/wiki/') and ':' not in href and not href.startswith('/wiki/Special:'):
                        links.append(urljoin("https://en.wikipedia.org/", href))

            cat_div = soup.select_one("div.mw-normal-catlinks")
            cats = [link.get_text().strip() for link in cat_div.find_all('li') if link.get_text().strip()]

            logger.info(
                f"{title[:50]:<50} | Pars: {len(text_parts):>3} | Chars: {len(full_text):>6} | Links: {len(links):>4} | Cats: {len(cats):>2}")
            return {'title': title, 'url': url, 'text': full_text, 'categories': cats, 'links': links}

        except Exception as e:
            logger.warning(f"Failed to scrape {url}: {e}")
            return None

    def _scrape(self, seed_article, links_per_page, limit, progress_scrape):
        logger.info("")
        logger.info("="*60)
        seed_url = "https://en.wikipedia.org/wiki/" + seed_article
        logger.info(f"🚀 Starting Scrape: {seed_url}")
        start_time = time.time()

        NUM_OF_THREADS = 6  # Set to (Wikipedia connection pool limit - 4)
        BATCH_SIZE = max(links_per_page, 3*NUM_OF_THREADS)

        discovered_urls = set([seed_url])  # for avoiding duplicate links between pages (optimization)
        visited_urls = set()  # ultimate url check
        visited_titles = set()  # secondary check (in case of different urls pointing to the same article)
        queue = deque([seed_url])  # collected links for processing
        max_queue_size = 0  # metric
        data = []

        # Issue: Our queue can grow endlessly. It would be optimal to use Producer/Consumer model
        with ThreadPoolExecutor(max_workers=NUM_OF_THREADS) as executor:
            while queue and len(data) < limit:
                # PROCESS IN BATCHES of BATCH_SIZE
                batch_size = min(BATCH_SIZE, limit - len(data), len(queue))
                batch_urls = []

                for _ in range(batch_size):
                    # 1. Pop URL
                    url = queue.popleft()
                    # 2. Check if VISITED URL
                    if url not in visited_urls:
                        # 3. Add to VISITED URL and process
                        visited_urls.add(url)
                        batch_urls.append(url)

                if not batch_urls:
                    logger.info("Scraping complete | NO NEW URLS IN A BATCH")
                    break

                # BATCH BEING PROCESSED
                future_to_url = {
                    # Dictionary: task (future) -> url
                    executor.submit(self._scrape_single_page, url): url
                    for url in batch_urls
                }

                # Asynchronously solving tasks (futures) - scraping
                # Start loop in order of completion
                for future in as_completed(future_to_url):
                    scraped_page = future.result()

                    if scraped_page is None:
                        # Logging handled in _scrape_single_page
                        # logger.warning(f"Skipping: {future_to_url[future]:<90} ||| NO DATA")
                        continue

                    # 4. Check if VISITED TITLE (different urls)
                    if scraped_page['title'] not in visited_titles:
                        # 5. Add to VISITED TITLE
                        visited_titles.add(scraped_page['title'])
                        # 6. Get NEW URLs (not yet DISCOVERED)
                        new_links = [l for l in scraped_page['links'] if l not in discovered_urls][:links_per_page]
                        # 7. Add to DISCOVERED and process
                        discovered_urls.update(new_links)
                        queue.extend(new_links)
                        max_queue_size = max(max_queue_size, len(queue))

                        data.append({
                            'title': scraped_page['title'],
                            'url': scraped_page['url'],
                            'text': scraped_page['text'],
                            'categories': scraped_page['categories'],
                            'links': scraped_page['links']
                        })
                        self.successful_fetches += 1
                        progress_scrape(len(data), limit, scraped_page['title'])
                    else:
                        logger.warning(f"Skipping: {scraped_page['url']:<90} ||| ALREADY PROCESSED")
                        self.duplicate_fetches += 1

                    if len(data) >= limit:
                        logger.info("Scraping complete | REACHED LIMIT OF PAGES TO SCRAPE")
                        break

                    if not queue:
                        logger.info("Scraping complete | QUEUE EMPTY")

        total_time = time.time() - start_time

        # Estimate memory size: max * ~(avg url size + string overhead + pointer size) / KB
        queue_estimate = (max_queue_size * (70 + 50 + 8)) / 1000
        logger.info(f"Max queue items during scraping: {max_queue_size} (~{queue_estimate:.0f} KB)")

        original_fetches = self.fetch_attempts - self.duplicate_fetches
        success_rate = (self.successful_fetches / original_fetches) * 100 if original_fetches > 0 else 0
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"Total Time:                               {total_time:.2f} seconds")
        logger.info(f"Total URL Attempts:                       {self.fetch_attempts}")
        logger.info(f"Total URL Attempts (without duplicates):  {original_fetches}")
        logger.info(f"Successful Fetches:                       {self.successful_fetches} ({success_rate:.2f}%)")

        if not data:
            logger.error("No data was scraped. Check the seed URL and network connection.")
            return

        self.raw_data = pd.DataFrame(data).drop_duplicates(subset='title').reset_index(drop=True)

    def _plot_word_count_distribution(self):
        if self.raw_data.empty:
            return

        word_counts = self.raw_data['text'].apply(lambda x: len(str(x).split()))

        plt.figure(figsize=(10, 6))
        plt.hist(word_counts, bins=30, color='#9467bd', edgecolor='black', alpha=0.7)

        mean_val = word_counts.mean()
        median_val = word_counts.median()
        plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.0f}')
        plt.axvline(median_val, color='green', linestyle='dashdot', linewidth=2, label=f'Median: {median_val:.0f}')

        plt.title('Distribution of Article Word Counts', fontsize=14)
        plt.xlabel('Number of Words', fontsize=12)
        plt.ylabel('Number of Articles', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend()

        plot_filename = "plots/word_count_distribution.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"✅ Word-count distribution plot saved to {plot_filename}")

    def _plot_top_categories(self, n=15):
        if self.raw_data.empty:
            return

        all_cats = [cat for sublist in self.raw_data['categories'] for cat in sublist]

        if not all_cats:
            logger.warning("No categories found to plot.")
            return

        cat_counts = pd.Series(all_cats).value_counts().head(n)

        plt.figure(figsize=(12, 8))
        cat_counts.sort_values(ascending=True).plot(kind='barh', color='#ff7f0e', edgecolor='black', alpha=0.8)

        plt.title(f'Top {n} Most Frequent Categories', fontsize=14)
        plt.xlabel('Number of Articles', fontsize=12)
        plt.ylabel('Category Name', fontsize=12)
        plt.grid(axis='x', linestyle='--', alpha=0.6)

        for i, v in enumerate(cat_counts.sort_values(ascending=True)):
            plt.text(v + 0.5, i, str(v), color='black', va='center', fontweight='bold')

        plot_filename = "plots/top_categories.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"✅ Top categories plot saved to {plot_filename}")

    def _plot_category_diversity(self, data):
        cumulative_total = 0
        seen_categories = set()
        coefficients = []

        for entry in data:
            cats = entry.get('categories', [])
            cumulative_total += len(cats)
            seen_categories.update(cats)
            coeff = len(seen_categories) / cumulative_total if cumulative_total > 0 else 0
            coefficients.append(coeff)

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(data) + 1), coefficients, color='#2ca02c', linewidth=2)
        plt.title('Category Diversity Coefficient vs. Articles Scraped', fontsize=14)
        plt.xlabel('Number of Articles Scraped', fontsize=12)
        plt.ylabel('Coefficient (Unique / Total Tags)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.axhline(y=coefficients[-1], color='r', linestyle=':', label=f'Final: {coefficients[-1]:.4f}')
        plt.legend()
        plot_filename = "plots/category_diversity.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"✅ Category diversity plot saved to {plot_filename}")

    def _analyze_raw_data(self):
        # WORD COUNT DISTRIBUTION
        self._plot_word_count_distribution()

        # CATEGORY STATS
        data = self.raw_data.to_dict('records')
        all_cats = [cat for entry in data for cat in entry.get('categories', [])]
        total_cats = len(all_cats)
        unique_cats = len(set(all_cats))
        coefficient = unique_cats / total_cats if total_cats > 0 else 0

        logger.info("")
        logger.info("=" * 60)
        logger.info("CATEGORY STATISTICS")
        logger.info(f"Total Found:  {total_cats}")
        logger.info(f"Unique Found:       {unique_cats}")
        logger.info(f"Diversity Coefficient:   {coefficient:.4f} (Unique/Total)")

        self._plot_top_categories()
        self._plot_category_diversity(data)

    def _preprocess(self, progress_prep):
        # PREPROCESSING
        logger.info("")
        logger.info("=" * 60)
        logger.info("🛠 Starting Preprocessing...")

        processed = []
        total = len(self.raw_data)

        if total == 0:
            logger.error("DataFrame is empty after deduplication.")
            return

        for i, text in enumerate(self.raw_data['text']):
            clean_text = re.sub(r'[^a-ząćęłńóśźż\s]', ' ', text.lower())
            clean_text = re.sub(r'\[\d+]', '', clean_text) # remove citations

            tokens = word_tokenize(clean_text)
            tokens = [
                self.lemmatizer.lemmatize(w)
                for w in tokens
                if w not in self.stop_words and len(w) > 2
            ]
            processed.append(" ".join(tokens))
            progress_prep(i + 1, total)
            if i % 100 == 0:
                logger.info(f"Articles Processed: {i}/{total}")

        self.raw_data['cleaned_text'] = processed
        db_filename = "db/" + DATABASE_FILENAME_PREFIX + self.raw_data.iloc[0]['url'].rstrip('/').split('/')[-1].lower() + ".parquet"
        self.raw_data.to_parquet(db_filename)
        logger.info(f"✅ Pipeline Complete. {total} articles processed and saved to {db_filename}")

    def _analyze_processed_data(self):
        if 'cleaned_text' not in self.raw_data.columns:
            logger.warning("No processed text available. Run preprocessing first.")
            return

        random_idx = self.raw_data.sample(n=1).index[0]

        logger.info("")
        logger.info("=" * 50)
        logger.info(f"RANDOM ARTICLE ({self.raw_data.iloc[random_idx]['url']})")
        logger.info(f"[{', '.join(self.raw_data.iloc[random_idx]['categories'])}]")
        logger.info("=" * 50)
        logger.info(f"ORIGINAL FIRST SENTENCE:")
        logger.info(f"{self.raw_data['text'].iloc[random_idx][:150]}...")
        logger.info("-" * 50)
        logger.info(f"PROCESSED VERSION (Root words only):")
        logger.info(f"{self.raw_data['cleaned_text'].iloc[random_idx][:150]}...")
        logger.info("=" * 50)
        logger.info("")

    def _plot_zipf_law(self):
        if 'cleaned_text' not in self.raw_data.columns:
            logger.warning("No processed text available. Run preprocessing first.")
            return

        all_words = []
        for text in self.raw_data['cleaned_text']:
            all_words.extend(text.split())

        word_counts = pd.Series(all_words).value_counts()
        frequencies = word_counts.values
        ranks = range(1, len(frequencies) + 1)

        plt.figure(figsize=(10, 6))
        plt.loglog(ranks, frequencies, color='#1f77b4', linewidth=2, marker='o', markersize=3, alpha=0.7)
        plt.title("Zipf's Law: Word Frequency Distribution", fontsize=14)
        plt.xlabel('Rank (log scale)', fontsize=12)
        plt.ylabel('Frequency (log scale)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7, which='both')

        ideal_zipf = frequencies[0] / np.array(ranks)
        plt.loglog(ranks, ideal_zipf, 'r--', alpha=0.5, label="Ideal Zipf's Law (f ∝ 1/r)")
        plt.legend()

        plot_filename = "plots/zipf_law.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"✅ Zipf's Law plot saved to {plot_filename}")

    def _plot_heaps_law(self):
        if 'cleaned_text' not in self.raw_data.columns:
            logger.warning("No processed text available. Run preprocessing first.")
            return

        seen_words = set()
        vocab_sizes = []
        total_words = []
        word_count = 0

        for text in self.raw_data['cleaned_text']:
            words = text.split()
            word_count += len(words)
            seen_words.update(words)
            total_words.append(word_count)
            vocab_sizes.append(len(seen_words))

        plt.figure(figsize=(10, 6))
        plt.loglog(total_words, vocab_sizes, color='#ff7f0e', linewidth=2, marker='o', markersize=3, alpha=0.7)
        plt.title("Heaps' Law: Vocabulary Growth", fontsize=14)
        plt.xlabel('Total Words (log scale)', fontsize=12)
        plt.ylabel('Unique Words (log scale)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7, which='both')

        if len(total_words) > 0:
            beta = 0.5  # typical value
            K = vocab_sizes[0] / (total_words[0] ** beta) if total_words[0] > 0 else 1
            ideal_heaps = K * np.array(total_words) ** beta
            plt.loglog(total_words, ideal_heaps, 'r--', alpha=0.5, label=f"Ideal Heaps' Law (V ∝ N^{beta})")
            plt.legend()

        plot_filename = "plots/heaps_law.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"✅ Heaps' Law plot saved to {plot_filename}")

    def _create_article_network(self):
        if self.raw_data is None or len(self.raw_data) == 0:
            logger.warning("No data available. Run scraping first.")
            return

        logger.info("")
        logger.info("=" * 60)
        logger.info("🕸️ Creating Article Network Visualization...")

        url_to_title = {row['url']: row['title'] for _, row in self.raw_data.iterrows()}

        G = nx.Graph()

        for _, row in self.raw_data.iterrows():
            G.add_node(row['title'], url=row['url'])

        for _, row in self.raw_data.iterrows():
            source_title = row['title']
            links = row.get('links', [])

            for link_url in links:
                if link_url in url_to_title:
                    target_title = url_to_title[link_url]
                    if source_title != target_title:
                        G.add_edge(source_title, target_title)

        logger.info(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        if G.number_of_edges() == 0:
            logger.warning("No connections found. Links may not be saved in the dataset.")
            pos = nx.circular_layout(G)
        else:
            logger.info("Computing network layout...")
            if G.number_of_nodes() > 100:
                pos = nx.spring_layout(G, k=0.8, iterations=40, seed=42)
            else:
                pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

        degrees = dict(G.degree())
        max_degree = max(degrees.values()) if degrees and max(degrees.values()) > 0 else 1

        edge_trace = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace.append(
                go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode='lines',
                    line=dict(width=0.5, color='#888'),
                    hoverinfo='none',
                    showlegend=False
                )
            )

        node_x = []
        node_y = []
        node_text = []
        node_size = []
        node_color = []

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

            degree = degrees[node]
            node_size.append(5 + (degree / max_degree) * 20)
            node_color.append(degree)
            node_text.append(f"{node}<br>Connections: {degree}")

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                size=node_size,
                color=node_color,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Connections", thickness=15, len=0.7),
                line=dict(width=1, color='white')
            ),
            showlegend=False
        )

        fig = go.Figure(data=edge_trace + [node_trace])

        fig.update_layout(
            title=dict(
                text=f"Wikipedia Article Network<br><sub>{G.number_of_nodes()} articles, {G.number_of_edges()} connections</sub>",
                x=0.5,
                xanchor='center'
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=60),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='#f8f9fa',
            width=1200,
            height=800
        )

        plot_filename = "plots/article_network.html"
        fig.write_html(plot_filename)
        logger.info(f"✅ Interactive network saved to {plot_filename}")

    def _scrape_and_process(self, seed_article, links_per_page, limit, progress_scrape, progress_prep):
        self._scrape(seed_article, links_per_page, limit, progress_scrape)
        if self.raw_data is not None and not self.raw_data.empty:
            self._analyze_raw_data()
            self._preprocess(progress_prep)
            self._analyze_processed_data()
            self._plot_zipf_law()
            self._plot_heaps_law()
            self._create_article_network()

    def _load_preprocessed_data(self, db_name):
        logger.info("")
        logger.info("=" * 60)
        db_filename = f"db/{DATABASE_FILENAME_PREFIX}{db_name.lower()}.parquet"
        logger.info(f"📂 Loading preprocessed data from {db_name}...")

        if not os.path.exists(db_filename):
            logger.error(f"Database file not found: {db_filename}")
            return False

        try:
            self.raw_data = pd.read_parquet(db_filename)
            logger.info(f"✅ Loaded {len(self.raw_data)} articles from {db_filename}")
            self._analyze_processed_data()
            self._plot_zipf_law()
            self._plot_heaps_law()
            self._create_article_network()
            return True
        except Exception as e:
            logger.error(f"Failed to load database: {e}")
            return False

    def get_data(self, scrape, progress_scrape, progress_prep,
                 seed_article="Poland",
                 links_per_page=10, limit=200):
        if scrape:
            self._scrape_and_process(seed_article, links_per_page, limit, progress_scrape, progress_prep)
        else:
            self._load_preprocessed_data(seed_article)

    def _plot_tfidf_top_terms(self, top_n=25):
        if not hasattr(self, 'tfidf_matrix') or self.tfidf_matrix is None:
            logger.error("TF-IDF matrix not found. Please run vectorize() first.")
            return

        logger.info("")
        logger.info(f"📊 Plotting Top {top_n} TF-IDF Terms...")

        weights = np.asarray(self.tfidf_matrix.sum(axis=0)).flatten()

        term_weights = pd.DataFrame({
            'term': self.feature_names,
            'weight': weights
        })

        top_terms = term_weights.sort_values(by='weight', ascending=False).head(top_n)

        plt.figure(figsize=(12, 8))

        colors = plt.cm.viridis(np.linspace(0.8, 0.2, top_n))

        plt.barh(top_terms['term'][::-1], top_terms['weight'][::-1], color=colors)

        plt.title(f'Top {top_n} Terms by Aggregate TF-IDF Score', fontsize=15, pad=20)
        plt.xlabel('Aggregate TF-IDF Weight', fontsize=12)
        plt.ylabel('Terms / N-grams', fontsize=12)
        plt.grid(axis='x', linestyle='--', alpha=0.7)

        for i, v in enumerate(top_terms['weight'][::-1]):
            plt.text(v + (max(weights) * 0.01), i, f'{v:.2f}', va='center', fontsize=10)

        os.makedirs("plots", exist_ok=True)
        plot_filename = "plots/top_tfidf_terms.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"✅ Top terms plot saved to {plot_filename}")

        top_5 = top_terms.head(5)['term'].tolist()
        logger.info(f"🔥 Top 5 terms: {', '.join(top_5)}")

    def _log_tfidf_top_terms_of_article(self, article_id=0, n_terms=10):
        logger.info("")
        if self.tfidf_matrix is None:
            logger.warning("Top TF-IDF Terms NOT EVALUATED. No TF-IDF matrix found.")
            return None
        logger.info(f"📝 Top {n_terms} TF-IDF Terms for Article: {self.raw_data.iloc[article_id]['title']}")

        article_vector = self.tfidf_matrix[article_id].toarray().flatten()
        top_indices = article_vector.argsort()[-n_terms:][::-1]

        for rank, idx in enumerate(top_indices, 1):
            term = self.feature_names[idx]
            score = article_vector[idx]
            logger.info(f"   {rank:2d}. {term:20s} (score: {score:.4f})")

        return None

    def _plot_lsa_variance(self, target_variance=0.5):
        if not hasattr(self, 'lsa_model') or self.lsa_model is None:
            logger.info("LSA Variance Plot not created. LSA model not found.")
            return None

        logger.info("")
        logger.info("=" * 60)
        logger.info("📊 Plotting LSA Variance Analysis...")

        cumulative_variance = np.cumsum(self.lsa_model.explained_variance_ratio_)
        n_components_range = range(1, len(cumulative_variance) + 1)

        n_components_target = np.argmax(cumulative_variance >= target_variance) + 1
        actual_variance = cumulative_variance[n_components_target - 1]

        plt.figure(figsize=(10, 6))
        plt.plot(n_components_range, cumulative_variance, color='#1f77b4', linewidth=2)
        plt.title('LSA Cumulative Explained Variance vs. Components', fontsize=14)
        plt.xlabel('Number of Components', fontsize=12)
        plt.ylabel('Cumulative Explained Variance', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.scatter([n_components_target], [actual_variance],
                    color='red', zorder=5)

        plt.axhline(y=target_variance, color='r', linestyle=':',
                    label=f'Target: {target_variance:.0%}')
        plt.axvline(x=n_components_target, color='g', linestyle=':',
                    label=f'Components needed: {n_components_target}')

        plt.legend()
        plt.ylim(0, 1.05)

        plot_filename = "plots/lsa_variance.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"✅ LSA variance plot saved to {plot_filename}")
        logger.info(f"📈 {n_components_target} components needed for {target_variance:.0%} variance")
        logger.info(f"   (Actual variance: {actual_variance:.2%})")

        return n_components_target

    def _log_lsa_top_concepts(self, article_id=None, n_concepts=5, n_terms_per_concept=10):
        logger.info("")
        logger.info("=" * 60)

        if not hasattr(self, 'lsa_model') or self.lsa_model is None:
            logger.warning("LSA concepts NOT EVALUATED. LSA model not found.")
            return None

        if article_id is not None:
            logger.info(f"🧠 Top LSA Concepts for Article: {self.raw_data.iloc[article_id]['title']}...")
            logger.info("")

            article_vector = self.reduced_matrix[article_id]
            positive_indices = np.argsort(article_vector)[-n_concepts:][::-1]
            negative_indices = np.argsort(article_vector)[:n_concepts]

            for label, indices in [("POSITIVE", positive_indices), ("NEGATIVE", negative_indices)]:
                logger.info(f"--- Top {n_concepts} {label} Concepts for this Article ---")

                for i, concept_idx in enumerate(indices, start=1):
                    weight = article_vector[concept_idx]
                    component = self.lsa_model.components_[concept_idx]

                    top_pos_idx = component.argsort()[-n_terms_per_concept:][::-1]
                    top_neg_idx = component.argsort()[:n_terms_per_concept]

                    pos_terms = ", ".join([self.feature_names[idx] for idx in top_pos_idx])
                    neg_terms = ", ".join([self.feature_names[idx] for idx in top_neg_idx])

                    logger.info(f"Concept #{i} (Article weight: {weight:+.4f}):")
                    logger.info(f"   Positive: {pos_terms}")
                    logger.info(f"   Negative: {neg_terms}")
                    logger.info("")
        else:
            logger.info(f"🔍 General LSA Concepts (Top {n_concepts} concepts)...")
            logger.info("")

            top_indices = np.argsort(self.lsa_model.explained_variance_ratio_)[-n_concepts:][::-1]

            for i, concept_idx in enumerate(top_indices, start=1):
                component = self.lsa_model.components_[concept_idx]
                variance = self.lsa_model.explained_variance_ratio_[concept_idx]

                top_pos_idx = component.argsort()[-n_terms_per_concept:][::-1]
                top_neg_idx = component.argsort()[:n_terms_per_concept]

                pos_terms = ", ".join([self.feature_names[idx] for idx in top_pos_idx])
                neg_terms = ", ".join([self.feature_names[idx] for idx in top_neg_idx])

                logger.info(f"Concept #{i} (explains {variance:.2%} variance):")
                logger.info(f"   Positive: {pos_terms}")
                logger.info(f"   Negative: {neg_terms}")
                logger.info("")

        return None

    def _plot_category_similarity_distribution(self, same_cat_sims, diff_cat_sims, ratio, matrix_name):
        plt.figure(figsize=(12, 6))

        plt.hist(diff_cat_sims, bins=50, alpha=0.6, color='#d62728',
                 label=f'Different Categories (μ={np.mean(diff_cat_sims):.3f})', density=True)
        plt.hist(same_cat_sims, bins=50, alpha=0.6, color='#2ca02c',
                 label=f'Shared Categories (μ={np.mean(same_cat_sims):.3f})', density=True)

        plt.axvline(np.mean(diff_cat_sims), color='#d62728', linestyle='--', linewidth=2)
        plt.axvline(np.mean(same_cat_sims), color='#2ca02c', linestyle='--', linewidth=2)

        plt.title(
            f'{matrix_name} Cosine Similarity Distribution by Category Overlap\n(Discrimination Ratio: {ratio:.2f}x)',
            fontsize=14)
        plt.xlabel('Cosine Similarity', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.3)

        plot_filename = f"plots/category_similarity_{matrix_name.lower().replace('-', '_')}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"✅ Category similarity distribution plot saved to {plot_filename}")

    def _evaluate_single_matrix_stats(self, same_cat_sims, diff_cat_sims, matrix_name):
        avg_same = np.mean(same_cat_sims) if same_cat_sims else 0
        avg_diff = np.mean(diff_cat_sims) if diff_cat_sims else 0
        discrimination_ratio = avg_same / avg_diff if avg_diff > 0 else 0

        logger.info(f"   Avg Similarity (Shared Category):    {avg_same:.4f}")
        logger.info(f"   Avg Similarity (Different Category): {avg_diff:.4f}")
        logger.info(f"   Discrimination Ratio:                {discrimination_ratio:.2f}x")
        logger.info(
            f"   Sample pairs analyzed: {len(same_cat_sims):,} same-category, {len(diff_cat_sims):,} different-category")

        self._plot_category_similarity_distribution(same_cat_sims, diff_cat_sims,
                                                    discrimination_ratio, matrix_name)

        return {
            'discrimination_ratio': discrimination_ratio,
            'avg_same': avg_same,
            'avg_diff': avg_diff
        }

    def _evaluate_by_categories(self, sample_size=200):
        logger.info("")
        logger.info("=" * 60)
        logger.info("📊 Evaluating Model by Category Similarity...")

        if self.tfidf_matrix is None:
            logger.warning("No TF-IDF matrix found. Run vectorize() first.")
            return None

        # Pre-calculate Similarity Matrices
        sim_tfidf = cosine_similarity(self.tfidf_matrix)
        sim_lsa = None
        if hasattr(self, 'lsa_model') and self.lsa_model is not None:
            sim_lsa = cosine_similarity(self.reduced_matrix)

        actual_sample_size = min(sample_size, len(self.raw_data))
        indices = np.random.choice(len(self.raw_data), actual_sample_size, replace=False)

        logger.info(f"Sampling {actual_sample_size} articles (comparing against all others)...")

        tfidf_data = {'same': [], 'diff': []}
        lsa_data = {'same': [], 'diff': []}

        for i in indices:
            cats_i = set(self.raw_data.iloc[i]['categories'])

            for j in range(len(self.raw_data)):
                if i == j: continue

                cats_j = set(self.raw_data.iloc[j]['categories'])
                is_same_cat = bool(cats_i.intersection(cats_j))

                # TF-IDF Collection
                sim_val_tfidf = sim_tfidf[i, j]
                if is_same_cat:
                    tfidf_data['same'].append(sim_val_tfidf)
                else:
                    tfidf_data['diff'].append(sim_val_tfidf)

                # LSA Collection
                if sim_lsa is not None:
                    sim_val_lsa = sim_lsa[i, j]
                    if is_same_cat:
                        lsa_data['same'].append(sim_val_lsa)
                    else:
                        lsa_data['diff'].append(sim_val_lsa)

        results = {}

        logger.info("")
        logger.info("🔍 Evaluating TF-IDF Matrix...")
        results['tfidf'] = self._evaluate_single_matrix_stats(tfidf_data['same'], tfidf_data['diff'], "TF-IDF")

        if sim_lsa is not None:
            logger.info("")
            logger.info("🔍 Evaluating LSA-Reduced Matrix...")
            results['lsa'] = self._evaluate_single_matrix_stats(lsa_data['same'], lsa_data['diff'], "LSA")

            logger.info("")
            logger.info("📊 Comparison Summary:")
            improvement = results['lsa']['discrimination_ratio'] - results['tfidf']['discrimination_ratio']
            if improvement > 0:
                logger.info(f"   ✅ LSA improves separation by {improvement:.2f}x")
            else:
                logger.info(f"   ⚠️  LSA decreases separation by {abs(improvement):.2f}x")

        return results

    def vectorize(self, max_features=10000, use_lsa=True, max_df=0.8, min_df=3, ngram_max=3,
                  n_components=100, auto_components=True, target_variance=0.40):
        logger.info("")
        logger.info("=" * 60)
        logger.info("✨ Starting Vectorization...")

        self.tfidf_matrix = None
        self.lsa_model = None
        self.reduced_matrix = None
        gc.collect()

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            sublinear_tf=True,
            ngram_range=(1, ngram_max),
            max_df=max_df,
            min_df=min_df
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.raw_data['cleaned_text'])
        self.feature_names = self.vectorizer.get_feature_names_out()

        logger.info(f"TF-IDF Matrix created: {self.tfidf_matrix.shape[0]} docs, {self.tfidf_matrix.shape[1]} features")
        self._plot_tfidf_top_terms()
        self._log_tfidf_top_terms_of_article()


        if use_lsa:
            # Issue: hardcoded, might not work with very big datasets
            if auto_components:
                n_components = 1000
            limit = min(self.tfidf_matrix.shape[0] - 1, self.tfidf_matrix.shape[1] - 1)
            actual_components = min(n_components, limit)
            if actual_components < n_components:
                logger.info("")
                logger.warning(f"Adjusting n_components from {n_components} to {actual_components} to fit data.")

            self.lsa_model = TruncatedSVD(n_components=actual_components, random_state=42)
            self.reduced_matrix = self.lsa_model.fit_transform(self.tfidf_matrix)
            var_ratio = self.lsa_model.explained_variance_ratio_.sum()

            target_components = self._plot_lsa_variance(target_variance=target_variance)
            if auto_components:
                self.lsa_model = TruncatedSVD(n_components=target_components, random_state=42)
                self.reduced_matrix = self.lsa_model.fit_transform(self.tfidf_matrix)
                var_ratio = self.lsa_model.explained_variance_ratio_.sum()
            logger.info("")
            logger.info(f"LSA completed. Explained Variance: {var_ratio:.2%}")
            logger.info(f"Final Matrix Shape: {self.reduced_matrix.shape}")

            self._log_lsa_top_concepts()
            self._log_lsa_top_concepts(0)

        else:
            logger.info("")
            logger.info("LSA disabled.")
            self.reduced_matrix = self.tfidf_matrix
            logger.info(f"Final Matrix Shape: {self.reduced_matrix.shape}")

        self._evaluate_by_categories()

    def get_recommendations(self, visited_titles, top_n=20):
        indices = []
        weights = []

        for i, title in enumerate(visited_titles):
            idx_match = self.raw_data.index[self.raw_data['title'] == title]
            if not idx_match.empty:
                indices.append(idx_match[0])

                weights.append(math.log(i + 2))

        if not indices:
            return []

        vectors = self.reduced_matrix[indices]

        if hasattr(vectors, "toarray"):
            vectors = vectors.toarray()

        weights = np.array(weights).reshape(-1, 1)

        user_profile = np.sum(vectors * weights, axis=0) / np.sum(weights)
        user_profile = np.asarray(user_profile).reshape(1, -1)

        sims = cosine_similarity(user_profile, self.reduced_matrix).flatten()

        # Exclude already visited items
        sims[indices] = -1.0
        rel_indices = sims.argsort()[::-1][:top_n]

        return [(self.raw_data.iloc[idx]['title'], sims[idx], idx) for idx in rel_indices]

    def calculate_recommendation_coverage(self, visited_titles, recommendations):
        if not visited_titles or not recommendations:
            return {'match_count': 0, 'precision': 0, 'shared_categories': []}

        visited_indices = self.raw_data[self.raw_data['title'].isin(visited_titles)].index
        visited_categories = set()
        for idx in visited_indices:
            visited_categories.update(self.raw_data.loc[idx, 'categories'])

        if not visited_categories:
            logger.warning("No categories found for the visited pages.")
            return {'match_count': 0, 'precision': 0, 'shared_categories': []}

        match_count = 0
        matching_categories = set()

        for title, sim, idx in recommendations:
            rec_categories = set(self.raw_data.loc[idx, 'categories'])

            common = visited_categories.intersection(rec_categories)
            if common:
                match_count += 1
                matching_categories.update(common)

        precision = (match_count / len(recommendations)) * 100 if recommendations else 0

        logger.info(f"✨ Category Coverage: {match_count}/{len(recommendations)} recommendations "
                    f"({precision:.1f}%) share categories with history.")

        return {
            'match_count': match_count,
            'total_recommendations': len(recommendations),
            'precision_percent': precision,
            'shared_categories': list(matching_categories)
        }

    def explain_similarity(self, visited_titles, rec_idx, top_n=5):
        indices = []
        weights = []

        for i, title in enumerate(reversed(visited_titles)):
            matches = self.raw_data.index[self.raw_data['title'] == title]
            if not matches.empty:
                indices.append(matches[0])
                weights.append(1.0 / math.log(i + 3))

        if not indices:
            return []

        user_vectors = self.reduced_matrix[indices]
        rec_vector = self.reduced_matrix[rec_idx]

        if hasattr(user_vectors, "toarray"):
            user_vectors = user_vectors.toarray()
        if hasattr(rec_vector, "toarray"):
            rec_vector = rec_vector.toarray()

        rec_vector = np.asarray(rec_vector).flatten()

        weights = np.array(weights).reshape(-1, 1)
        user_profile = np.sum(user_vectors * weights, axis=0) / np.sum(weights)

        contributions = user_profile * rec_vector

        results = []

        if hasattr(self, 'lsa_model') and self.lsa_model is not None:
            # --- LSA MODE ---
            top_concept_indices = contributions.argsort()[::-1][:top_n]

            for c_idx in top_concept_indices:
                score = contributions[c_idx]
                if score <= 0: continue

                component_weights = self.lsa_model.components_[c_idx]

                pos_indices = component_weights.argsort()[::-1][:3]
                pos_terms = [self.feature_names[i] for i in pos_indices]

                neg_indices = component_weights.argsort()[:3]
                neg_terms = [self.feature_names[i] for i in neg_indices]

                desc = (f"Concept {c_idx}: \n+++ {', '.join(pos_terms)} "
                        f"\n--- {', '.join(neg_terms)}")

                results.append((desc, score))
        else:
            # --- TF-IDF MODE ---
            top_indices = contributions.argsort()[::-1][:top_n]

            for idx in top_indices:
                score = contributions[idx]
                if score > 0:
                    term = self.feature_names[idx]
                    results.append((term, score))

        return results

    def plot_vector_space(self, visited_titles, recommendations, filename="plots/vector_space_map.png"):
        # Import Plotly locally to keep module-level changes minimal
        import plotly.graph_objects as go

        if not hasattr(self, 'embedding_2d') or self.embedding_2d is None:
            logger.info("⚡ Computing 2D projection (t-SNE) for the first time...")
            tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
            self.embedding_2d = tsne.fit_transform(self.reduced_matrix)

        coords = self.embedding_2d

        # Identify Indices for different groups
        visited_indices = []
        for title in visited_titles:
            idx_match = self.raw_data.index[self.raw_data['title'] == title]
            if not idx_match.empty:
                visited_indices.append(idx_match[0])

        rec_indices = [item[2] for item in recommendations]

        # Create a mask for "background" (Knowledge Base) points
        all_indices = set(range(len(self.raw_data)))
        special_indices = set(visited_indices) | set(rec_indices)
        bg_indices = list(all_indices - special_indices)

        # Calculate Centroid
        if visited_indices:
            weights = []
            for i in range(len(visited_indices)):
                weights.append(1.0 / math.log((len(visited_indices) - i) + 3))  # Newest = higher weight

            weights = np.array(weights).reshape(-1, 1)
            visited_coords = coords[visited_indices]
            center_coord = np.sum(visited_coords * weights, axis=0) / np.sum(weights)
        else:
            center_coord = np.mean(coords, axis=0)

        # 1. STATIC MATPLOTLIB PLOT
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(14, 10))

        # A. Background (Knowledge Base)
        ax.scatter(coords[bg_indices, 0], coords[bg_indices, 1],
                   c='#444444', s=10, alpha=0.7, label='Knowledge Base', linewidth=0)

        # B. Reading Path
        if len(visited_indices) > 1:
            path_x = coords[visited_indices, 0]
            path_y = coords[visited_indices, 1]
            ax.plot(path_x, path_y, c='#5da5da', alpha=0.3, linestyle='--', linewidth=1, label='Reading Path')

        # C. Visited Dots (With Fading Alpha)
        if visited_indices:
            n_visited = len(visited_indices)
            for i, idx in enumerate(visited_indices):
                alpha_score = 0.1 + (0.9 * (i / max(n_visited - 1, 1)))

                ax.scatter(coords[idx, 0], coords[idx, 1],
                           c='#5da5da', s=60, alpha=alpha_score,
                           edgecolors='black', linewidth=0.5,
                           label='Visited' if i == n_visited - 1 else "")

        # D. Recommendations (Stars)
        if rec_indices:
            ax.scatter(coords[rec_indices, 0], coords[rec_indices, 1],
                       c='#d4a373', marker='*', s=150, alpha=0.9,
                       edgecolors='black', linewidth=0.5, label='Recommendations')

        # E. Interest Center (X Marker)
        ax.scatter(center_coord[0], center_coord[1],
                   c='#d62728', marker='x', s=100, linewidth=2, label='Interest Center')

        ax.set_title("Vector Space Map: History vs. Recommendations", fontsize=15, pad=20, color='gray')
        ax.axis('off')

        legend = ax.legend(loc='upper right', frameon=True, facecolor='#1e1e1e', edgecolor='#333333')
        for text in legend.get_texts():
            text.set_color('gray')

        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='#1e1e1e')
        plt.close()

        # 2. INTERACTIVE PLOTLY MAP
        fig_px = go.Figure()

        # A. Background
        fig_px.add_trace(go.Scattergl(
            x=coords[bg_indices, 0], y=coords[bg_indices, 1],
            mode='markers',
            name='Knowledge Base',
            text=self.raw_data.iloc[bg_indices]['title'],
            marker=dict(color='#444444', size=5, opacity=0.7)
        ))

        # B. Reading Path
        if len(visited_indices) > 1:
            fig_px.add_trace(go.Scatter(
                x=coords[visited_indices, 0], y=coords[visited_indices, 1],
                mode='lines',
                name='Reading Path',
                line=dict(color='#5da5da', width=1, dash='dash'),
                opacity=0.3, hoverinfo='skip'
            ))

        # C. Visited Dots (Calculated Alphas)
        if visited_indices:
            n_visited = len(visited_indices)
            # Create RGBA colors for fading effect
            visited_colors = [f"rgba(93, 165, 218, {0.1 + (0.9 * (i / max(n_visited - 1, 1))):.2f})"
                              for i in range(n_visited)]

            fig_px.add_trace(go.Scatter(
                x=coords[visited_indices, 0], y=coords[visited_indices, 1],
                mode='markers',
                name='Visited',
                text=self.raw_data.iloc[visited_indices]['title'],
                marker=dict(color=visited_colors, size=10, line=dict(color='black', width=0.5))
            ))

        # D. Recommendations
        if rec_indices:
            fig_px.add_trace(go.Scatter(
                x=coords[rec_indices, 0], y=coords[rec_indices, 1],
                mode='markers',
                name='Recommendations',
                text=self.raw_data.iloc[rec_indices]['title'],
                marker=dict(color='#d4a373', symbol='star', size=15,
                            line=dict(color='black', width=0.5), opacity=0.9)
            ))

        # E. Interest Center
        fig_px.add_trace(go.Scatter(
            x=[center_coord[0]], y=[center_coord[1]],
            mode='markers',
            name='Interest Center',
            text=[' Calculated Interest Center'],
            marker=dict(color='#d62728', symbol='x', size=12, line=dict(width=2))
        ))

        # Layout styling to match Matplotlib dark theme
        fig_px.update_layout(
            title="Vector Space Map: History vs. Recommendations",
            template="plotly_dark",
            paper_bgcolor='#1e1e1e',
            plot_bgcolor='#1e1e1e',
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            showlegend=True
        )

        html_filename = filename.replace(".png", ".html")
        fig_px.write_html(html_filename)

        logger.info(f"✅ Vector maps saved to {filename} and {html_filename}")