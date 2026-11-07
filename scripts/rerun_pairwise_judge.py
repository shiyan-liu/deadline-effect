import os
import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.stats import binomtest
from review_features import extract_paper_features, compute_global_buckets
JUDGE_SYSTEM = "You are an expert meta-reviewer comparing the QUALITY of two peer reviews of the SAME paper (not the paper's quality). Use the paper context to judge whether each review's points are accurate and relevant. Apply the rubric for each axis literally -- base your decision on the concrete criteria given, not a vague overall impression."
JUDGE_PROMPT_TMPL = 'The paper being reviewed:\n\nTitle: {paper_title}\nAbstract: {paper_abstract}\n\nBelow are two independent peer reviews (Review A and Review B) of this SAME\npaper, from two different reviewers. Compare them using the rubric below for\neach axis.\n\n--- REVIEW A START ---\nSummary: {summary_a}\nStrengths: {strengths_a}\nWeaknesses: {weaknesses_a}\nQuestions: {questions_a}\n--- REVIEW A END ---\n\n--- REVIEW B START ---\nSummary: {summary_b}\nStrengths: {strengths_b}\nWeaknesses: {weaknesses_b}\nQuestions: {questions_b}\n--- REVIEW B END ---\n\nRubric (apply literally, per axis, in this fixed order):\n1. THOROUGHNESS: does the review engage with SPECIFIC parts of the paper\n  (particular sections, equations, tables, experiments, assumptions) rather\n  than generic statements that could apply to almost any paper? The review\n  citing more specific paper content wins.\n2. SPECIFICITY: does the critique make concrete, checkable claims (e.g.\n  "missing baseline X", "Eq. 3 assumes Y which doesn\'t hold because Z")\n  rather than generic comments ("needs more experiments", "clarity could\n  improve")? The review with more concrete, checkable claims wins.\n3. CONSTRUCTIVENESS: does the review propose concrete, actionable fixes or\n  alternative approaches, not just flag problems? The review with more\n  actionable suggestions wins.\n4. OVERALL: holistically, which review would be more useful to the authors\n  and an area chair in making a decision?\n\nFor each of the 4 criteria above, decide which review (A or B) better\nsatisfies it, or TIE if genuinely indistinguishable. Use TIE sparingly --\nonly when truly indistinguishable.\n\nRespond with EXACTLY one line, no other text: four values separated by\ncommas, each A, B, or TIE, in the fixed order\nTHOROUGHNESS,SPECIFICITY,CONSTRUCTIVENESS,OVERALL.\nExample valid response: A,B,A,TIE\n'
AXES = ['thoroughness', 'specificity', 'constructiveness', 'overall']

def call_llm(prompt, model, api_key):
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url='https://openrouter.ai/api/v1', timeout=45.0, max_retries=2)
    resp = client.chat.completions.create(model=model, messages=[{'role': 'system', 'content': JUDGE_SYSTEM}, {'role': 'user', 'content': prompt}], temperature=0.0, max_tokens=120, timeout=45.0)
    content = resp.choices[0].message.content
    if not content:
        raise ValueError('Empty response')
    return content

def parse_pairwise(text):
    line = text.strip().splitlines()[0] if text.strip() else ''
    parts = [p.strip().upper() for p in line.split(',')]
    if len(parts) == 4 and all((p in ('A', 'B', 'TIE') for p in parts)):
        return dict(zip(AXES, parts))
    r = {}
    for ax in AXES:
        m = re.search(f'{ax}:\\s*(A|B|TIE)', text, re.IGNORECASE)
        r[ax] = m.group(1).upper() if m else None
    return r

def build_pairs(raw, papers_with_buckets, rng):
    text_by_id = {}
    paper_meta = {}
    for paper in raw:
        fid = paper.get('forum', paper.get('id', ''))
        paper_meta[fid] = {'title': paper.get('title', ''), 'abstract': paper.get('abstract', '')}
        for r in paper.get('reviews', []):
            rid = r.get('id', '')
            if rid:
                text_by_id[rid] = {'summary': (r.get('summary') or '').strip(), 'strengths': (r.get('strengths') or '').strip(), 'weaknesses': (r.get('weaknesses') or '').strip(), 'questions': (r.get('questions') or '').strip()}
    pairs = []
    for paper in papers_with_buckets:
        early = [r for r in paper['reviews'] if r['bucket'] == 'Early']
        late = [r for r in paper['reviews'] if r['bucket'] == 'Late']
        if not early or not late:
            continue
        e = rng.choice(early)
        l = rng.choice(late)
        te = text_by_id.get(e['review_id'])
        tl = text_by_id.get(l['review_id'])
        if not te or not tl:
            continue
        if sum((len(v) for v in te.values())) < 40 or sum((len(v) for v in tl.values())) < 40:
            continue
        meta = paper_meta.get(paper['paper_id'], {'title': '', 'abstract': ''})
        if not meta['abstract']:
            continue
        pairs.append({'paper_id': paper['paper_id'], 'early_text': te, 'late_text': tl, **meta})
    return pairs

def judge_with_position(pair, early_is_a, model, api_key):
    a, b = (pair['early_text'], pair['late_text']) if early_is_a else (pair['late_text'], pair['early_text'])
    prompt = JUDGE_PROMPT_TMPL.format(paper_title=pair['title'][:300], paper_abstract=pair['abstract'][:1500], summary_a=a['summary'][:1200] or '(none provided)', strengths_a=a['strengths'][:1200] or '(none provided)', weaknesses_a=a['weaknesses'][:1200] or '(none provided)', questions_a=a['questions'][:1200] or '(none provided)', summary_b=b['summary'][:1200] or '(none provided)', strengths_b=b['strengths'][:1200] or '(none provided)', weaknesses_b=b['weaknesses'][:1200] or '(none provided)', questions_b=b['questions'][:1200] or '(none provided)')
    try:
        raw_resp = call_llm(prompt, model, api_key)
        picks = parse_pairwise(raw_resp)
    except Exception as e:
        return {'error': str(e)}
    out = {}
    for ax in AXES:
        pick = picks.get(ax)
        if pick is None or pick == 'TIE':
            out[ax + '_winner'] = pick
        else:
            picked_early = (pick == 'A') == early_is_a
            out[ax + '_winner'] = 'Early' if picked_early else 'Late'
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--year', required=True)
    ap.add_argument('--n_pairs', type=int, default=200)
    ap.add_argument('--model', default='openai/gpt-4o-mini')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', required=True)
    ap.add_argument('--execute-paid-api',action='store_true')
    args = ap.parse_args()
    if not args.execute_paid_api:
        ap.error('External API calls incur charges; pass --execute-paid-api to run.')
    api_key = os.environ['OPENROUTER_API_KEY']
    raw = json.load(open(args.data, encoding='utf-8'))
    all_ratings = [float(r['rating']) for p in raw for r in p.get('reviews', []) if r.get('rating') is not None]
    scale_min, scale_max = (min(all_ratings), max(all_ratings))
    papers = [extract_paper_features(p, scale_min, scale_max) for p in raw]
    papers = [p for p in papers if p is not None]
    papers, q33, q66 = compute_global_buckets(papers)
    rng = random.Random(args.seed)
    all_pairs = build_pairs(raw, papers, rng)
    print(f'Papers with both an Early and a Late reviewer + abstract available: {len(all_pairs)}')
    rng.shuffle(all_pairs)
    sample = all_pairs[:args.n_pairs] if args.n_pairs > 0 else all_pairs
    print(f'Sampled {len(sample)} pairs, model={args.model} -- judging BOTH position orderings each ({2 * len(sample)} total LLM calls)')
    for p in sample:
        p['_early_is_a'] = rng.random() < 0.5

    def process_pair(pair):
        fwd = judge_with_position(pair, pair['_early_is_a'], args.model, api_key)
        rev = judge_with_position(pair, not pair['_early_is_a'], args.model, api_key)
        return {'paper_id': pair['paper_id'], 'fwd': fwd, 'rev': rev}
    results = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_pair, p): p for p in sample}
        for i, fut in enumerate(as_completed(futs)):
            results.append(fut.result())
            if (i + 1) % 50 == 0 or i + 1 == len(sample):
                print(f'  {i + 1}/{len(sample)} done  ({time.time() - t0:.0f}s elapsed)')
    print(f'\nDone. {len(results)} pairs x2 orderings, {time.time() - t0:.0f}s total')
    print('\n=== Position-debiased win rates (Early vs Late) -- only counts pairs where BOTH orderings agree; disagreement/tie/error = inconclusive, excluded ===')
    summary = {}
    for ax in AXES:
        key = ax + '_winner'
        early_wins = late_wins = tie_agree = inconsistent = error_or_missing = 0
        for r in results:
            fwd_v = r['fwd'].get(key) if 'error' not in r['fwd'] else None
            rev_v = r['rev'].get(key) if 'error' not in r['rev'] else None
            if 'error' in r['fwd'] or 'error' in r['rev'] or fwd_v is None or (rev_v is None):
                error_or_missing += 1
            elif fwd_v == rev_v == 'TIE':
                tie_agree += 1
            elif fwd_v == rev_v == 'Early':
                early_wins += 1
            elif fwd_v == rev_v == 'Late':
                late_wins += 1
            else:
                inconsistent += 1
        decided = early_wins + late_wins
        if decided > 0:
            bt = binomtest(early_wins, decided, 0.5)
            win_rate = early_wins / decided
            print(f'  {ax:<18s} Early={early_wins} Late={late_wins} decisive_win_rate={win_rate:.3f} p={bt.pvalue:.4g}  |  tie(both)={tie_agree}  inconsistent(position-flip)={inconsistent} ({100 * inconsistent / len(results):.1f}%)  error/missing={error_or_missing}')
            summary[ax] = {'early_wins': early_wins, 'late_wins': late_wins, 'decisive_win_rate': win_rate, 'binom_p': bt.pvalue, 'tie_agree': tie_agree, 'inconsistent': inconsistent, 'inconsistent_rate': inconsistent / len(results), 'error_or_missing': error_or_missing}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({'year': args.year, 'model': args.model, 'n_pairs_target': args.n_pairs, 'summary': summary, 'results': results}, indent=2))
    print(f'\nSaved to {args.out}')
if __name__ == '__main__':
    main()
