"""LLM Negotiation Lab. Python 3.10+, HTTP via the standard library."""
import argparse
import copy
import hashlib
import json
import os
import random
import time
import urllib.error
import urllib.request
from pathlib import Path

AGENTS = ('A', 'B', 'C')
DEFAULTS = dict(version=3, model='qwen3:4b-instruct', base_seed=1729,
                temperature=0.4, num_predict=180, num_ctx=8192,
                timeout=180, max_rounds=8, retries=1,
                url='http://localhost:11434/api/chat')
RULES = ('Вы делите 100 единиц ресурса между A, B, C. Предложение содержит '
         'целые неотрицательные доли, сумма ровно 100. Предлагающие идут A, B, C '
         'по кругу. Предлагающий автоматически голосует за. Сделка требует '
         'единогласия; если за 8 раундов соглашение не достигнуто, каждый получает 0. '
         'Отвечай только JSON по заданной схеме. reason пиши на русском языке, не длиннее 15 слов.')
ROLES = {
    'A': 'Ты агент A. Максимизируй свою долю, но будь готов к компромиссу ради сделки.',
    'B': 'Ты агент B. Веди жёсткие переговоры; не принимай предложение, где твоя доля меньше 40.',
    'C': 'Ты агент C. Предпочитай справедливое распределение, близкое к 33/33/34.',
}
SCHEMAS = {
    'proposal': {'type': 'object', 'properties': {
        'allocation': {'type': 'object', 'properties': {
            a: {'type': 'integer', 'minimum': 0, 'maximum': 100} for a in AGENTS},
            'required': list(AGENTS), 'additionalProperties': False},
        'reason': {'type': 'string'}},
        'required': ['allocation', 'reason'], 'additionalProperties': False},
    'vote': {'type': 'object', 'properties': {
        'vote': {'type': 'string', 'enum': ['accept', 'reject']}, 'reason': {'type': 'string'}},
        'required': ['vote', 'reason'], 'additionalProperties': False},
}


class Invalid(ValueError):
    def __init__(self, category, message):
        super().__init__(message)
        self.category = category


def validate(raw, kind, agent, allocation=None):
    try:
        value = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise Invalid('json', str(exc)) from exc
    keys = {'allocation', 'reason'} if kind == 'proposal' else {'vote', 'reason'}
    if not isinstance(value, dict) or set(value) != keys:
        raise Invalid('schema', 'Incorrect object keys')
    reason = value['reason']
    if not isinstance(reason, str):
        raise Invalid('schema', 'reason must be a string')
    if kind == 'proposal':
        shares = value['allocation']
        if (not isinstance(shares, dict) or set(shares) != set(AGENTS)
                or any(type(x) is not int or not 0 <= x <= 100 for x in shares.values())):
            raise Invalid('schema', 'Invalid allocation types or bounds')
        if sum(shares.values()) != 100:
            raise Invalid('game', 'Allocation must sum to 100')
        if agent == 'B' and shares['B'] < 40:
            raise Invalid('game', 'B cannot automatically accept less than 40')
    else:
        if value['vote'] not in ('accept', 'reject'):
            raise Invalid('schema', 'vote must be accept or reject')
        if agent == 'B' and value['vote'] == 'accept' and allocation['B'] < 40:
            raise Invalid('game', 'B cannot accept less than 40')
    return value


def messages(agent, round_number, kind, history, memory, allocation=None):
    context = {'round': round_number, 'rounds_remaining_including_current': 9 - round_number,
               'action': kind}
    if allocation is not None:
        context['ТЕКУЩЕЕ ПРЕДЛОЖЕНИЕ'] = copy.deepcopy(allocation)
        context['задание'] = ('Голосуй только за распределение в блоке «ТЕКУЩЕЕ ПРЕДЛОЖЕНИЕ». '
                              'Не голосуй за предложения из истории.')
    if memory:
        context['ИСТОРИЯ (только для справки)'] = copy.deepcopy(history)
    return [{'role': 'system', 'content': ROLES[agent] + '\n' + RULES},
            {'role': 'user', 'content': json.dumps(context, ensure_ascii=False)}]


def chat(config, seed, msgs, kind):
    payload = dict(model=config['model'], messages=msgs, stream=False,
                   format=SCHEMAS[kind], options={
                       k: config[k] for k in ('temperature', 'num_predict', 'num_ctx')})
    payload['options']['seed'] = seed
    if config['model'] == 'qwen3:1.7b':
        payload['think'] = False
    request = urllib.request.Request(config['url'], json.dumps(payload).encode(),
                                     {'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=config['timeout']) as response:
        return json.load(response)['message']['content']


def action(config, seed, agent, number, kind, history, memory, calls, allocation=None,
           transport=chat):
    # Each agent gets a fresh, separate context containing only its private role.
    msgs = messages(agent, number, kind, history, memory, allocation)
    for attempt in range(config['retries'] + 1):
        entry = dict(agent=agent, round=number, kind=kind, attempt=attempt + 1,
                     messages=copy.deepcopy(msgs))
        start = time.perf_counter()
        try:
            raw = transport(config, seed, msgs, kind)
            entry['raw'] = raw
            result = validate(raw, kind, agent, allocation)
            entry.update(valid=True, error_category=None)
        except Invalid as exc:
            entry.update(valid=False, error_category=exc.category, error=str(exc))
            result = None
        except (OSError, ValueError, KeyError, TypeError) as exc:
            entry.update(valid=False, error_category='transport', error=str(exc))
            result = None
        entry['duration_seconds'] = time.perf_counter() - start
        calls.append(entry)
        if result is not None:
            return result
    return None


def simulate(config, index, condition, transport=chat):
    start = time.perf_counter()
    history, calls, invalid_actions = [], [], 0
    agreement = None
    for number in range(1, config['max_rounds'] + 1):
        proposer = AGENTS[(number - 1) % 3]
        proposal = action(config, config['base_seed'] + index, proposer, number,
                          'proposal', history, condition == 'memory', calls,
                          transport=transport)
        public = dict(round=number, proposer=proposer, proposal=proposal, votes={})
        if proposal is None:
            invalid_actions += 1
            public['skipped'] = True
        else:
            allocation = proposal['allocation']
            public['votes'][proposer] = dict(vote='accept', automatic=True)
            for agent in AGENTS:
                if agent == proposer:
                    continue
                vote = action(config, config['base_seed'] + index, agent, number,
                              'vote', history, condition == 'memory', calls,
                              allocation, transport)
                if vote is None:
                    invalid_actions += 1
                public['votes'][agent] = vote or dict(vote='reject', invalid=True)
            if all(v['vote'] == 'accept' for v in public['votes'].values()):
                agreement = allocation
        history.append(public)
        if agreement is not None:
            break
    return dict(type='run', config=config, config_id=config_id(config), index=index,
                seed=config['base_seed'] + index, condition=condition,
                status='agreed' if agreement is not None else 'failed',
                rounds=len(history), allocation=agreement, history=history, calls=calls,
                invalid_actions=invalid_actions,
                duration_seconds=time.perf_counter() - start)


def log_value(value):
    """Truncate explanations only in the persisted copy, never in agent context."""
    if isinstance(value, list):
        return [log_value(item) for item in value]
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key == 'reason' and isinstance(item, str):
                result[key] = item[:200]
            elif key in ('raw', 'content') and isinstance(item, str):
                try:
                    parsed = json.loads(item)
                except ValueError:
                    result[key] = item
                else:
                    result[key] = json.dumps(log_value(parsed), ensure_ascii=False)
            else:
                result[key] = log_value(item)
        return result
    return value


def config_id(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]


def read_logs(path):
    if not path.exists():
        return []
    rows = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        try:
            rows.append(json.loads(line))
        except ValueError:
            print(f'Пропущена повреждённая строка {number} в {path}')
    return rows


def gini(allocation):
    values = list(allocation.values())
    return sum(abs(x - y) for x in values for y in values) / (2 * 3 * sum(values))


def metrics(rows):
    successful = [r for r in rows if r['status'] == 'agreed']
    mean = lambda values: sum(values) / len(values) if values else None
    calls = [c for r in rows for c in r['calls']]
    first = [c for c in calls if c['attempt'] == 1]
    turns = [t for r in rows for t in r['history'] if t['proposal'] is not None]
    votes = [v for t in turns for v in t['votes'].values() if not v.get('automatic')]
    almost = [sum(v['vote'] == 'reject' for v in t['votes'].values()) == 1 for t in turns]
    return dict(accept_rate=mean([v['vote'] == 'accept' for v in votes]),
                almost_agreement_count=sum(almost),
                agreement_rate=mean([r['status'] == 'agreed' for r in rows]),
                rounds_to_agreement=mean([r['rounds'] for r in successful]),
                **{f'share_{a}': mean([r['allocation'][a] for r in successful]) for a in AGENTS},
                gini=mean([gini(r['allocation']) for r in successful]),
                invalid_response_rate=mean([not c['valid'] for c in calls]),
                initial_invalid_rate=mean([not c['valid'] for c in first]),
                final_invalid_action_rate=(sum(r['invalid_actions'] for r in rows) / len(first)
                                           if first else None))


def diagnostics(rows):
    """Describe public negotiation behavior, including runs without agreement."""
    mean = lambda xs: sum(xs) / len(xs) if xs else None
    rounds, proposals = {}, []
    blockers = {a: dict(rejects=0, votes=0, sole_rejects=0) for a in AGENTS}
    trajectories = {a: [] for a in AGENTS}
    for run in rows:
        own = {a: [] for a in AGENTS}
        for turn in run['history']:
            bucket = rounds.setdefault(turn['round'], dict(
                reached=0, allocations=[], own_shares=[], accepts=0, votes=0,
                automatic_accepts=0, invalid_rejects=0))
            bucket['reached'] += 1
            if turn['proposal'] is None:
                continue
            allocation = turn['proposal']['allocation']
            proposer = turn['proposer']
            rejects = []
            for agent, vote in turn['votes'].items():
                if vote.get('automatic'):
                    bucket['automatic_accepts'] += 1
                    continue
                accepted = vote['vote'] == 'accept'
                bucket['accepts'] += accepted
                bucket['votes'] += 1
                bucket['invalid_rejects'] += bool(vote.get('invalid'))
                blockers[agent]['votes'] += 1
                if not accepted:
                    rejects.append(agent)
                    blockers[agent]['rejects'] += 1
            if len(rejects) == 1:
                blockers[rejects[0]]['sole_rejects'] += 1
            bucket['allocations'].append(allocation)
            bucket['own_shares'].append(allocation[proposer])
            previous = own[proposer][-1]['own_share'] if own[proposer] else None
            point = dict(round=turn['round'], own_share=allocation[proposer],
                         concession_from_previous=(previous - allocation[proposer]
                                                   if previous is not None else None))
            own[proposer].append(point)
            proposals.append(dict(index=run['index'], round=turn['round'], proposer=proposer,
                                  allocation=allocation, rejects=rejects,
                                  sole_reject=len(rejects) == 1,
                                  proposer_share=allocation[proposer], gini=gini(allocation)))
        for agent in AGENTS:
            points = own[agent]
            trajectories[agent].append(dict(index=run['index'], proposals=points,
                net_concession=(points[0]['own_share'] - points[-1]['own_share']
                                if len(points) >= 2 else None)))
    by_round = {}
    for number, bucket in sorted(rounds.items()):
        total = bucket['votes'] + bucket['automatic_accepts']
        by_round[number] = dict(runs_reached=bucket['reached'],
            valid_proposals=len(bucket['allocations']),
            accept_count=bucket['accepts'], vote_count=bucket['votes'],
            accept_rate=bucket['accepts'] / bucket['votes'] if bucket['votes'] else None,
            accept_rate_including_proposer=((bucket['accepts'] + bucket['automatic_accepts']) / total
                                             if total else None),
            invalid_reject_count=bucket['invalid_rejects'],
            proposal_gini=mean([gini(a) for a in bucket['allocations']]),
            proposer_share=mean(bucket['own_shares']))
    for counts in blockers.values():
        counts['reject_rate'] = counts['rejects'] / counts['votes'] if counts['votes'] else None
    concessions = {}
    for agent, runs in trajectories.items():
        deltas = [p['concession_from_previous'] for r in runs for p in r['proposals']
                  if p['concession_from_previous'] is not None]
        nets = [r['net_concession'] for r in runs if r['net_concession'] is not None]
        concessions[agent] = dict(runs=runs, transitions=len(deltas),
            mean_step_concession=mean(deltas), mean_net_concession=mean(nets))
    return dict(by_round=by_round, proposals=proposals, blockers=blockers,
                almost_agreements_by_run=[dict(index=r['index'], count=sum(
                    p['sole_reject'] for p in proposals if p['index'] == r['index']))
                    for r in rows],
                concessions=concessions)


def analyze(path, fingerprint, output, bootstrap=2000):
    rows = [r for r in read_logs(path) if r['config_id'] == fingerprint]
    report = dict(config_id=fingerprint, bootstrap=bootstrap, conditions={})
    rng = random.Random(2026)
    for condition in ('no_memory', 'memory'):
        group = list({r['index']: r for r in rows if r['condition'] == condition}.values())
        estimates = metrics(group)
        samples = [metrics(rng.choices(group, k=len(group))) for _ in range(bootstrap)] if group else []
        intervals = {}
        for key in estimates:
            values = sorted(s[key] for s in samples if s[key] is not None)
            intervals[key] = ([values[int((len(values)-1)*q)] for q in (.025, .975)]
                              if values else None)
        report['conditions'][condition] = dict(n=len(group), metrics=estimates, ci95=intervals,
                                               diagnostics=diagnostics(group))
    output.mkdir(parents=True, exist_ok=True)
    (output / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib отсутствует: summary.json готов; установите matplotlib для графика.')
        return report
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, key, title in zip(axes[0], ('agreement_rate', 'rounds_to_agreement', 'gini'),
                              ('Agreement rate', 'Rounds (successful)', 'Gini (successful)')):
        for x, (condition, data) in enumerate(report['conditions'].items()):
            value, ci = data['metrics'][key], data['ci95'][key]
            if value is not None:
                ax.scatter(x, value)
                if ci:
                    ax.vlines(x, ci[0], ci[1])
        ax.set_xticks([0, 1], ['No memory', 'Memory'])
        ax.set_title(title)
        ax.grid(axis='y', alpha=.25)
    for condition, data in report['conditions'].items():
        by_round = data['diagnostics']['by_round']
        numbers = list(by_round)
        for ax, key, title in zip(axes[1, :2], ('accept_rate', 'proposal_gini'),
                                  ('Accept rate (two voters)', 'Proposal Gini')):
            ax.plot(numbers, [by_round[n][key] for n in numbers], marker='o', label=condition)
            ax.set(xlabel='Round', title=title)
        for agent, stats in data['diagnostics']['concessions'].items():
            points = [p for r in stats['runs'] for p in r['proposals']]
            numbers = sorted({p['round'] for p in points})
            means = [sum(p['own_share'] for p in points if p['round'] == n) /
                     sum(p['round'] == n for p in points) for n in numbers]
            axes[1, 2].plot(numbers, means, marker='o', label=f'{condition}: {agent}')
    axes[1, 2].set(xlabel='Round', title='Own share in own proposals')
    for ax in axes[1]:
        ax.legend(fontsize=7)
        ax.grid(alpha=.25)
    fig.suptitle('Real runs; top: bootstrap 95% CI; bottom: descriptive trajectories')
    fig.tight_layout()
    fig.savefig(output / 'comparison.png', dpi=160)
    plt.close(fig)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['run', 'analyze', 'prompts'])
    parser.add_argument('--model', default=DEFAULTS['model'])
    parser.add_argument('--runs', type=int, default=20)
    parser.add_argument('--log', type=Path, default=Path('results/runs.jsonl'))
    parser.add_argument('--output', type=Path, default=Path('results/analysis'))
    args = parser.parse_args()
    config = dict(DEFAULTS, model=args.model)
    if args.command == 'prompts':
        print(json.dumps(dict(system_prompts={a: ROLES[a] + '\n' + RULES for a in AGENTS},
                              schemas=SCHEMAS), ensure_ascii=False, indent=2))
        return
    if args.runs < 1:
        parser.error('--runs must be positive')
    if args.command == 'analyze':
        print(json.dumps(analyze(args.log, config_id(config), args.output), ensure_ascii=False, indent=2))
        return
    args.log.parent.mkdir(parents=True, exist_ok=True)
    done = {(r['index'], r['condition']) for r in read_logs(args.log)
            if r.get('config_id') == config_id(config) and r.get('config') == config
            and r.get('status') in ('agreed', 'failed')}
    for index in range(args.runs):
        # Alternate order to reduce systematic warm-up/order bias.
        conditions = ('no_memory', 'memory') if index % 2 == 0 else ('memory', 'no_memory')
        for condition in conditions:
            if (index, condition) in done:
                continue
            row = simulate(config, index, condition)
            with args.log.open('a') as stream:
                # A process killed mid-write may leave an incomplete final line.
                # Separate it so the next complete record remains readable.
                if args.log.stat().st_size:
                    with args.log.open('rb') as check:
                        check.seek(-1, 2)
                        if check.read(1) != b'\n':
                            stream.write('\n')
                stream.write(json.dumps(log_value(row), ensure_ascii=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
            print(f"{index=} {condition=} {row['status']} rounds={row['rounds']} seconds={row['duration_seconds']:.2f}", flush=True)


if __name__ == '__main__':
    main()
