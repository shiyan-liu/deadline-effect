"""Review preprocessing retained for numerical compatibility with the analysis."""
import statistics

def extract_paper_features(paper: dict, scale_min: float=None, scale_max: float=None) -> dict | None:
    reviews = paper.get('reviews', [])
    if not reviews:
        return None
    valid_reviews = [r for r in reviews if r.get('cdate') and r.get('rating') is not None]
    if len(valid_reviews) < 2:
        return None
    decision_raw = paper.get('decision', '')
    decision_lower = decision_raw.lower()
    if 'accept' in decision_lower:
        decision = 'Accept'
    else:
        decision = 'Reject'
    cdates_ms = [r['cdate'] for r in valid_reviews]
    latest_ms = max(cdates_ms)
    earliest_ms = min(cdates_ms)
    review_window_ms = latest_ms - earliest_ms
    if review_window_ms < 3600000:
        return None
    reviews_processed = []
    for r in valid_reviews:
        gap_ms = latest_ms - r['cdate']
        relative_pos = gap_ms / review_window_ms
        raw_rating = float(r['rating'])
        if scale_min is not None and scale_max is not None and (scale_max > scale_min):
            rating_norm = (raw_rating - scale_min) / (scale_max - scale_min)
        else:
            rating_norm = raw_rating
        reviews_processed.append({'review_id': r.get('id', ''), 'cdate': r['cdate'], 'rating': raw_rating, 'rating_norm': round(rating_norm, 6), 'confidence': float(r.get('confidence', 3)), 'soundness': float(r.get('soundness', r['rating'])), 'relative_pos': relative_pos, 'gap_hours': gap_ms / 3600000})
    return {'paper_id': paper.get('forum', paper.get('id', '')), 'title': paper.get('title', ''), 'decision': decision, 'decision_binary': 1 if decision == 'Accept' else 0, 'n_reviewers': len(reviews_processed), 'reviews': reviews_processed}

def compute_global_buckets(papers: list[dict]) -> list[dict]:
    all_pos = [r['relative_pos'] for p in papers for r in p['reviews']]
    all_pos_sorted = sorted(all_pos)
    n = len(all_pos_sorted)
    q33 = all_pos_sorted[n // 3]
    q66 = all_pos_sorted[2 * n // 3]
    result = []
    for paper in papers:
        reviews = paper['reviews']
        all_ratings = [r['rating'] for r in reviews]
        all_ratings_norm = [r['rating_norm'] for r in reviews]
        paper_mean_rating = statistics.mean(all_ratings)
        paper_mean_rating_norm = statistics.mean(all_ratings_norm)
        reviews_with_bucket = []
        for r in reviews:
            pos = r['relative_pos']
            if pos >= q66:
                bucket = 'Early'
            elif pos >= q33:
                bucket = 'Mid'
            else:
                bucket = 'Late'
            others_ratings = [rr['rating'] for rr in reviews if rr['review_id'] != r['review_id']]
            others_ratings_norm = [rr['rating_norm'] for rr in reviews if rr['review_id'] != r['review_id']]
            others_mean = statistics.mean(others_ratings) if others_ratings else paper_mean_rating
            others_mean_norm = statistics.mean(others_ratings_norm) if others_ratings_norm else paper_mean_rating_norm
            reviews_with_bucket.append({**r, 'bucket': bucket, 'paper_mean_rating': paper_mean_rating, 'paper_mean_rating_norm': paper_mean_rating_norm, 'others_mean_rating': others_mean, 'within_paper_deviation': r['rating'] - others_mean, 'within_paper_deviation_norm': r['rating_norm'] - others_mean_norm})
        result.append({**paper, 'reviews': reviews_with_bucket})
    return (result, q33, q66)
