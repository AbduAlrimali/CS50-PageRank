import os
import re
import sys
import random

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )
    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """

    internal = list(corpus[page])
    probs = dict()
    for link in corpus:
        if link in internal:
            probs[link] = ((1- damping_factor)/len(corpus)) + (damping_factor/len(internal))
        else:
            probs[link] = (1-damping_factor)/len(corpus)
    return probs


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    current = random.choice(list(corpus.keys()))
    sample = []
    sample.append(current)
    for i in range(n):
        current = sample[i]
        probs = transition_model(corpus, current, damping_factor)
        current = random.choices(list(probs.keys()), weights=list(probs.values()), k=1)[0]
        sample.append(current)

    count = dict()
    for page in corpus.keys():
        count[page] = sample.count(page)/len(sample)
    return count

def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    rank=dict()
    for page in corpus.keys():
        rank.update({page: 1/len(corpus.keys())})
    prev_rank = rank.copy()
    check = dict((key, False) for key in rank.keys())
    loop=True
    while loop:

        for page in corpus.keys():
            prob1 = (1-damping_factor)/len(corpus.keys())
            prob2 = 0
            for prev_page, page_links in corpus.items():
                if len(page_links) == 0:
                    page_links = corpus.keys()
                if page in page_links:
                    prob2 += prev_rank[prev_page]/len(page_links)

            page_rank = prob2*damping_factor + prob1
            rank[page] = page_rank


        for page in prev_rank.keys():
            if abs(prev_rank[page] - rank[page]) < 0.001:
                check[page] = True
            else:
                check[page] = False

        if all(check.values()):
            loop = False
        prev_rank = rank.copy()
    return rank

if __name__ == "__main__":
    main()