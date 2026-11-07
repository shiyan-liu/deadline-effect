import os
import argparse
import json
import re
import sys
import time
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from scipy.stats import pearsonr, spearmanr
from review_features import extract_paper_features, compute_global_buckets
from legacy_helpers import _ols_optimal_timing_weights
JUDGE_SYSTEM = "You are an impartial reader analyzing an Area Chair's written meta-review for an ML conference. Judge only the strength of the reasoning given."
JUDGE_PROMPT_TMPL = "Below is an Area Chair's meta-review for a paper submission.\n\n--- METAREVIEW TEXT START ---\n{metareview}\n--- METAREVIEW TEXT END ---\n\nIgnore whatever explicit verdict word (accept/reject) appears. Instead, rate\nhow STRONG and POSITIVE the substance of the AC's stated reasoning is about\nthe paper's merit -- i.e. read past the verdict to how enthusiastic vs\nlukewarm-to-negative the actual argument is.\n\nRespond in EXACTLY this format, nothing else:\nENTHUSIASM: <1-5, 1=case is clearly negative, 3=mixed/lukewarm, 5=case is clearly strongly positive>\n"

def call_llm(prompt, model, api_key):
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url='https://openrouter.ai/api/v1', timeout=30.0, max_retries=2)
    resp = client.chat.completions.create(model=model, messages=[{'role': 'system', 'content': JUDGE_SYSTEM}, {'role': 'user', 'content': prompt}], temperature=0.0, max_tokens=20, timeout=30.0)
    content = resp.choices[0].message.content
    if not content:
        raise ValueError('Empty response')
    return content

def parse_enthusiasm(text):
    m = re.search('ENTHUSIASM:\\s*(\\d)', text)
    return int(m.group(1)) if m else None

def load_year(path):
    raw = json.load(open(path, encoding='utf-8'))
    all_ratings = [float(r['rating']) for p in raw for r in p.get('reviews', []) if r.get('rating') is not None]
    scale_min, scale_max = (min(all_ratings), max(all_ratings))
    papers = [extract_paper_features(p, scale_min, scale_max) for p in raw]
    papers = [p for p in papers if p is not None]
    papers, q33, q66 = compute_global_buckets(papers)
    meta_by_id = {p.get('forum', p.get('id', '')): (p.get('meta_review') or {}).get('metareview', '') for p in raw}
    return (papers, meta_by_id)

def otca_score(paper, a_star, b_star=1.0):
    rps = [r['relative_pos'] for r in paper['reviews']]
    rats = [r['rating'] for r in paper['reviews']]
    ws = [a_star * rp + b_star for rp in rps]
    if any((w <= 0 for w in ws)):
        return sum(rats) / len(rats)
    total_w = sum(ws)
    return sum((r * w for r, w in zip(rats, ws))) / total_w

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_papers', type=int, default=20, help='cap per year; 0 = use all papers with non-empty metareview')
    ap.add_argument('--model', default='openai/gpt-4o-mini')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', required=True)
    ap.add_argument('--raw_dir',type=Path,required=True)
    ap.add_argument('--execute-paid-api',action='store_true')
    args = ap.parse_args()
    if not args.execute_paid_api:
        ap.error('External API calls incur charges; pass --execute-paid-api to run.')
    api_key = os.environ['OPENROUTER_API_KEY']
    print('Loading 2024 and 2025 (only years with non-empty metareview text)...')
    papers_2024, meta_2024 = load_year(str(args.raw_dir / 'iclr2024_papers.json'))
    papers_2025, meta_2025 = load_year(str(args.raw_dir / 'iclr2025_papers.json'))
    print("Fitting OTCA weight function out-of-sample (2024 papers scored using 'a' fit on 2025, and vice versa)...")
    otca_2025 = _ols_optimal_timing_weights(papers_2025)
    otca_2024 = _ols_optimal_timing_weights(papers_2024)
    print(f"  a_star fit on 2024 data: {otca_2024['a_star']}  (mse_reduction={otca_2024['mse_reduction_pct']}%)")
    print(f"  a_star fit on 2025 data: {otca_2025['a_star']}  (mse_reduction={otca_2025['mse_reduction_pct']}%)")
    rng = random.Random(args.seed)

    def gather(papers, meta_by_id, a_star_from_other_year, year_label):
        items = []
        for p in papers:
            mr = meta_by_id.get(p['paper_id'], '')
            if mr and len(mr.strip()) >= 60:
                items.append({'paper_id': p['paper_id'], 'year': year_label, 'metareview': mr.strip(), 'uniform_avg': sum((r['rating'] for r in p['reviews'])) / len(p['reviews']), 'otca_weighted': otca_score(p, a_star_from_other_year), 'decision_binary': p['decision_binary']})
        rng.shuffle(items)
        if args.n_papers > 0:
            items = items[:args.n_papers]
        return items
    sample_2024 = gather(papers_2024, meta_2024, otca_2025['a_star'], '2024')
    sample_2025 = gather(papers_2025, meta_2025, otca_2024['a_star'], '2025')
    sample = sample_2024 + sample_2025
    print(f'Papers with usable metareview text: 2024={len(sample_2024)}  2025={len(sample_2025)}  total={len(sample)}')

    def judge_one(item):
        prompt = JUDGE_PROMPT_TMPL.format(metareview=item['metareview'][:3000])
        try:
            raw_resp = call_llm(prompt, args.model, api_key)
            enth = parse_enthusiasm(raw_resp)
        except Exception as e:
            enth = None
            item['error'] = str(e)
        item['llm_enthusiasm'] = enth
        return item
    results = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(judge_one, item): item for item in sample}
        for i, fut in enumerate(as_completed(futs)):
            res = fut.result()
            results.append(res)
            if (i + 1) % 50 == 0 or i + 1 == len(sample):
                print(f'  {i + 1}/{len(sample)} done  ({time.time() - t0:.0f}s elapsed)')
    valid = [r for r in results if r.get('llm_enthusiasm') is not None]
    n_errors = len(results) - len(valid)
    print(f'\nDone. {len(valid)} valid judgments, {n_errors} errors, {time.time() - t0:.0f}s total')
    if len(valid) >= 10:
        enth = np.array([r['llm_enthusiasm'] for r in valid], dtype=float)
        uni = np.array([r['uniform_avg'] for r in valid], dtype=float)
        otca = np.array([r['otca_weighted'] for r in valid], dtype=float)
        r_uni_p, p_uni_p = pearsonr(uni, enth)
        r_otca_p, p_otca_p = pearsonr(otca, enth)
        r_uni_s, p_uni_s = spearmanr(uni, enth)
        r_otca_s, p_otca_s = spearmanr(otca, enth)
        print('\n=== Correlation with LLM-derived AC enthusiasm score (non-circular target) ===')
        print(f'  uniform_avg   Pearson r={r_uni_p:.4f} (p={p_uni_p:.4g})  Spearman r={r_uni_s:.4f} (p={p_uni_s:.4g})')
        print(f'  otca_weighted Pearson r={r_otca_p:.4f} (p={p_otca_p:.4g})  Spearman r={r_otca_s:.4f} (p={p_otca_s:.4g})')
        print(f'  Pearson r improvement (otca - uniform): {r_otca_p - r_uni_p:+.4f}')
        summary = {'n_valid': len(valid), 'n_errors': n_errors, 'otca_a_star_fit_on_2024_applied_to_2025': otca_2024['a_star'], 'otca_a_star_fit_on_2025_applied_to_2024': otca_2025['a_star'], 'corr_uniform_avg': {'pearson_r': r_uni_p, 'pearson_p': p_uni_p, 'spearman_r': r_uni_s, 'spearman_p': p_uni_s}, 'corr_otca_weighted': {'pearson_r': r_otca_p, 'pearson_p': p_otca_p, 'spearman_r': r_otca_s, 'spearman_p': p_otca_s}}
    else:
        summary = {'n_valid': len(valid), 'n_errors': n_errors, 'note': 'too few valid judgments for correlation'}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({'summary': summary, 'results': results}, indent=2))
    print(f'Saved to {args.out}')
if __name__ == '__main__':
    main()
