"""Offline, paired run-cluster bootstrap and exploratory role diagnostics."""
import argparse
import csv
import json
import random
import re
from pathlib import Path

import lab

THRESHOLDS = {'A_share_below': 20, 'C_gini_above': 0.4}
BOOTSTRAP = 10000
SEED = 2026

REASON_PATTERNS = {
    'mentions_40': r'(?<!\d)40(?!\d)',
    'mentions_40_bound': (r'(?:(?:меньше|больше|ниже|выше|менее|более|минимум|максимум)\s+'
                          r'(?:чем\s+)?40(?!\d)|[<>≤≥]\s*40(?!\d)|'
                          r'(?<!\d)40\s*[<>≤≥]|порог\w*[^.!?\n]{0,20}\b40(?!\d))'),
    'mentions_B_share': (r'(?:дол[яиюе]\s*(?:агента\s*)?[bбв]\b|'
                         r'\b[bбв]\s*(?:и\s*[acас]\s*)?(?:получ\w*|[—–:=]|\(\s*\d|\d)|'
                         r'\b(?:для|у)\s+[bбв]\b)'),
}


def reason_flags(reason):
    flags = {key: bool(re.search(pattern, reason, flags=re.IGNORECASE))
             for key, pattern in REASON_PATTERNS.items()}
    flags['either'] = flags['mentions_40_bound'] or flags['mentions_B_share']
    return flags


def reason_search(groups):
    counts, records = {}, []
    for condition, group in groups.items():
        counts[condition] = {}
        for agent in ('A', 'C'):
            votes = [v for run in group.values() for v in extract(run)
                     if v['agent'] == agent and not v['automatic']]
            counts[condition][agent] = dict(votes=len(votes),
                with_reason=sum(bool(v['reason']) for v in votes),
                **{key: sum(reason_flags(v['reason'])[key] for v in votes)
                   for key in (*REASON_PATTERNS, 'either')})
            for vote in votes:
                records.append(dict(vote, **reason_flags(vote['reason'])))
    return dict(patterns=REASON_PATTERNS, counts=counts,
                scope='Final nonautomatic A/C votes; not attempts; denominator includes reason-less fallback rejects',
                caveat='Lexical proxies, not a classifier of justification or threshold transfer'), records

# Manual reading of saved explanations, not an automated semantic classifier.
REASON_NOTES = {
    (5, 5, 'B'): 'Автоматический голос; reason отвергает собственное предложение и приписывает C порог 40.',
    (5, 5, 'C'): 'Описывает прежнее 33/33/34 вместо текущего 60/40/0.',
    (5, 7, 'C'): 'Описывает 33/33/34 вместо 95/2/3.',
    (5, 8, 'A'): 'Принимает 0, ссылаясь на выгоду B и C; у B ровно 40, не больше 40.',
    (5, 8, 'C'): 'Описывает прежнее 33/33/34 вместо текущего 0/40/60.',
    (6, 2, 'A'): 'Обосновывает принятие собственной нулевой доли порогом B.',
    (6, 2, 'C'): 'Числа B/C верны, но объяснение не рассматривает нулевую долю A; формальный порог Джини не превышен.',
    (7, 2, 'A'): 'Сравнение с прошлой собственной долей неверно: A ранее предлагал себе 99, теперь получает 40.',
    (7, 3, 'A'): 'Обосновывает голос долей C, а не собственной; сами числа верны.',
    (7, 6, 'A'): 'Обосновывает голос долей C, а не собственной; сами числа верны.',
    (7, 7, 'C'): 'Утверждает C=34, хотя сейчас C=1; смешивает текущее предложение с историей.',
    (7, 8, 'A'): 'Описывает прежнее 99/0/1 вместо текущего 0/100/0.',
    (7, 8, 'C'): 'Утверждает C=34 и справедливость, хотя сейчас C=0 и Джини=0,667.',
    (8, 2, 'A'): 'Принимает собственный 0, обосновывая голос выгодой B/C.',
    (8, 2, 'C'): 'Называет распределение справедливым по равенству B/C, не учитывая A=0; Джини ниже порога 0,4.',
    (9, 2, 'A'): 'Принимает собственный 0, обосновывая голос порогом B/C.',
    (9, 3, 'A'): 'Утверждает B=C=50 из прошлого предложения; сейчас 34/33/33.',
    (9, 5, 'A'): 'Арифметическая ошибка: 40 не больше 40; собственная доля A=40 допустима по разведочному критерию.',
    (9, 5, 'C'): 'Описывает 33/33/34 вместо текущего 40/40/20.',
    (11, 2, 'A'): 'Принимает собственный 0, обосновывая голос выгодой B/C.',
    (11, 2, 'C'): 'Называет распределение справедливым по равенству B/C, не учитывая A=0; Джини ниже порога 0,4.',
    (16, 2, 'A'): 'Принимает собственный 0, обосновывая голос выгодой B/C.',
    (16, 3, 'A'): 'Утверждает B=C=50 из прошлого предложения; сейчас 33/33/34.',
    (16, 4, 'C'): 'Описывает прежнее 33/33/34 вместо текущего 51/49/0.',
}


def conflict(agent, shares):
    return ((agent == 'A' and shares['A'] < THRESHOLDS['A_share_below']) or
            (agent == 'C' and lab.gini(shares) > THRESHOLDS['C_gini_above']))


def extract(run):
    records = []
    for turn in run['history']:
        if turn['proposal'] is None:
            continue
        shares = turn['proposal']['allocation']
        for agent, vote in turn['votes'].items():
            records.append(dict(index=run['index'], seed=run['seed'],
                condition=run['condition'], round=turn['round'], agent=agent,
                proposer=turn['proposer'], automatic=bool(vote.get('automatic')),
                vote=vote['vote'], invalid=bool(vote.get('invalid')),
                share=shares[agent], gini=lab.gini(shares),
                allocation=shares, threshold_flag=conflict(agent, shares),
                reason=(turn['proposal']['reason'] if vote.get('automatic')
                        else vote.get('reason', ''))))
    return records


def cluster(run):
    votes = [v for v in extract(run) if not v['automatic']]
    accepted = [v for v in votes if v['vote'] == 'accept']
    eligible = [v for v in votes if v['agent'] in ('A', 'C')]
    eligible_accepted = [v for v in accepted if v['agent'] in ('A', 'C')]
    flags = sum(v['threshold_flag'] for v in eligible_accepted)
    counts = dict(accept_rate=(len(accepted), len(votes)),
                  conflict_per_AC_accept=(flags, len(eligible_accepted)),
                  conflict_per_AC_vote=(flags, len(eligible)),
                  conflict_per_all_accept=(flags, len(accepted)))
    for agent in lab.AGENTS:
        own = [v for v in accepted if v['agent'] == agent]
        counts[f'{agent}_accepted_share'] = (sum(v['share'] for v in own), len(own))
        if agent != 'B':
            counts[f'{agent}_conflict_per_accept'] = (sum(v['threshold_flag'] for v in own), len(own))
    return counts


def aggregate(clusters):
    result = {}
    for key in clusters[0]:
        numerator = sum(c[key][0] for c in clusters)
        denominator = sum(c[key][1] for c in clusters)
        result[key] = dict(numerator=numerator, denominator=denominator,
                           value=numerator / denominator if denominator else None)
    return result


def percentile(values, q):
    position = (len(values) - 1) * q
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


def paired_bootstrap(groups, repeats=BOOTSTRAP):
    ids = sorted(groups['no_memory'])
    if not ids or set(ids) != set(groups['memory']):
        raise ValueError('Need nonempty, fully paired run indices')
    clusters = {condition: [cluster(groups[condition][i]) for i in ids]
                for condition in groups}
    points = {c: aggregate(cs) for c, cs in clusters.items()}
    distributions = {key: {c: [] for c in (*groups, 'difference')}
                     for key in points['memory']}
    rng = random.Random(SEED)
    for _ in range(repeats):
        draw = rng.choices(range(len(ids)), k=len(ids))
        sample = {c: aggregate([cs[i] for i in draw]) for c, cs in clusters.items()}
        for key, values in distributions.items():
            for condition in groups:
                value = sample[condition][key]['value']
                if value is not None:
                    values[condition].append(value)
            a, b = (sample[c][key]['value'] for c in ('no_memory', 'memory'))
            if a is not None and b is not None:
                values['difference'].append(b - a)
    report = {}
    for key, samples in distributions.items():
        report[key] = {c: dict(points[c][key]) for c in groups}
        a, b = (points[c][key]['value'] for c in ('no_memory', 'memory'))
        report[key]['difference'] = dict(value=b-a if a is not None and b is not None else None)
        for condition, values in samples.items():
            values.sort()
            report[key][condition].update(ci95=([percentile(values, .025), percentile(values, .975)]
                                               if values else None),
                                          valid_bootstrap_samples=len(values))
    return dict(n_pairs=len(ids), bootstrap=repeats, bootstrap_seed=SEED,
                resampling='Paired run clusters; pooled ratio of sums; memory minus no_memory',
                exploratory_thresholds=THRESHOLDS, metrics=report)


def agreement_report(groups):
    lines = ['# Все соглашения: траектории и объяснения accept', '',
             'Источник: full.jsonl. Номера прогонов с нуля. Числа — A/B/C.',
             'Пометки ниже — ручная интерпретация сохранённых reason, не доказательство внутреннего рассуждения модели.',
             'Формальный разведочный флаг: A<20 или C принимает Джини>0,4; B исключён. Автоматические голоса не являются решениями.', '']
    for condition, group in groups.items():
        for index, run in sorted(group.items()):
            if run['status'] != 'agreed':
                continue
            lines += [f'## {condition}, прогон {index}, seed {run["seed"]}', '',
                      '| Раунд | Автор | A/B/C | Против | Джини |', '|---|---|---|---|---|']
            for turn in run['history']:
                proposal = turn['proposal']
                shares = proposal['allocation'] if proposal else None
                split = '/'.join(str(shares[a]) for a in lab.AGENTS) if shares else 'пропущено'
                rejects = ', '.join(a for a, v in turn['votes'].items() if v['vote'] == 'reject') or '—'
                inequality = f'{lab.gini(shares):.3f}' if shares else '—'
                lines.append(f'| {turn["round"]} | {turn["proposer"]} | {split} | {rejects} | {inequality} |')
            lines += ['', 'Все accept по пути к соглашению, включая автоматически заданные:', '']
            for vote in extract(run):
                if vote['vote'] != 'accept':
                    continue
                automatic = 'автоматический; reason предложения' if vote['automatic'] else 'добровольный'
                flag = '; разведочный флаг против роли' if vote['threshold_flag'] and not vote['automatic'] else ''
                lines += [f'- Раунд {vote["round"]}, {vote["agent"]}, доля {vote["share"]}, {automatic}{flag}: «{vote["reason"]}»']
                note = REASON_NOTES.get((index, vote['round'], vote['agent']))
                if note:
                    lines[-1] += f' **Замечание:** {note}'
            lines.append('')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', type=Path, default=Path('results/full.jsonl'))
    parser.add_argument('--output', type=Path, default=Path('results/full-analysis'))
    args = parser.parse_args()
    groups = {c: {} for c in ('no_memory', 'memory')}
    for run in lab.read_logs(args.log):
        if run.get('config') != lab.DEFAULTS or run.get('config_id') != lab.config_id(lab.DEFAULTS):
            continue
        if run.get('status') not in ('failed', 'agreed'):
            continue
        group = groups[run['condition']]
        if run['index'] in group:
            raise ValueError('Duplicate run: check input logs')
        group[run['index']] = run
    for i in set(groups['no_memory']) & set(groups['memory']):
        if groups['no_memory'][i]['seed'] != groups['memory'][i]['seed']:
            raise ValueError('Paired seed mismatch')
    report = paired_bootstrap(groups)
    report['config'] = lab.DEFAULTS
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'paired-bootstrap.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    records = [v for group in groups.values() for run in group.values()
               for v in extract(run) if v['vote'] == 'accept']
    with (args.output / 'accepted-votes.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (args.output / 'agreements.md').write_text(agreement_report(groups))
    search, searched_votes = reason_search(groups)
    (args.output / 'reason-search.json').write_text(json.dumps(search, ensure_ascii=False, indent=2))
    with (args.output / 'reason-search.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(searched_votes[0]))
        writer.writeheader()
        writer.writerows(searched_votes)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
