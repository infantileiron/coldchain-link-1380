"""货损率预测模型：按班次分组切分、仅用训练集学习缺失填补与标准化。

模型刻意保持简单（两个特征的线性回归），目的是让留出误差和风险原因可解释，
并可以和训练均值基线做同口径比较。
"""
import hashlib
import json
import math
import random

import numpy as np

ALGORITHM_VERSION = 'linear-regression-2f-1.0'
TARGET = 'loss'
FEATURES = ('exposure', 'excursions')
FEATURE_LABELS = {'exposure': '暴露时长', 'excursions': '超温次数'}
FEATURE_UNITS = {'exposure': '分钟', 'excursions': '次'}


def _finite(value):
    if value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _clean_number(value):
    return float(value) if _finite(value) else None


def _mean(values):
    return float(sum(values) / len(values)) if values else 0.0


def _std(values, mean):
    if not values:
        return 0.0
    return float(math.sqrt(sum((v - mean) ** 2 for v in values) / len(values)))


def _mae(predicted, actual):
    return float(sum(abs(float(p) - float(a)) for p, a in zip(predicted, actual)) / len(actual))


def _rmse(predicted, actual):
    return float(math.sqrt(sum((float(p) - float(a)) ** 2 for p, a in zip(predicted, actual)) / len(actual)))


def _canonical_hash(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), default=str)
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


def train(rows, seed=3102, holdout_fraction=0.2):
    """训练并返回可冻结的结果。

    rows 是训练时刻从数据库读出的普通 dict 快照。切分以 shipment_id 为组，
    同一个班次的所有样本只会落入训练集或留出集之一。特征填补均值、标准化
    均值/标准差和目标均值基线都只从训练组计算。
    """
    if not rows:
        raise ValueError('没有样本，无法训练货损模型')
    rows = [row for row in rows if (row.get('shipment_id') is not None)]
    if not 0 < holdout_fraction < 1:
        raise ValueError('留出比例必须介于 0 和 1 之间')

    groups = {}
    for row in rows:
        group = row.get('shipment_id')
        groups.setdefault(int(group), []).append(row)
    group_ids = sorted(groups)
    if len(group_ids) < 2:
        raise ValueError('至少需要两个班次才能按班次切分训练集和留出集')

    shuffled = list(group_ids)
    random.Random(int(seed)).shuffle(shuffled)
    holdout_count = max(1, min(len(group_ids) - 1, int(round(len(group_ids) * float(holdout_fraction)))))
    holdout_groups = set(shuffled[:holdout_count])
    train_groups = set(shuffled[holdout_count:])

    frozen_rows = []
    for row in sorted(rows, key=lambda r: (r.get('sample_id') is None, r.get('sample_id') or 0)):
        group = row.get('shipment_id')
        split = None if group is None else ('holdout' if group in holdout_groups else 'train')
        loss_ok = _finite(row.get('loss'))
        feature_count = sum(1 for feature in FEATURES if _finite(row.get(feature)))
        if group is None:
            included, reason = False, '样本没有关联班次，无法参与按班次切分'
        elif not loss_ok:
            included, reason = False, '货损缺失，不参与训练和留出评估'
        elif feature_count == 0:
            included, reason = False, '暴露时长与超温次数均缺失，无法形成特征'
        else:
            included, reason = True, ''
        frozen_rows.append({
            'sample_id': row.get('sample_id'),
            'sample_code': row.get('sample_code'),
            'shipment_id': group,
            'shipment_code': row.get('shipment_code'),
            'zone': row.get('zone'),
            'exposure': _clean_number(row.get('exposure')),
            'excursions': _clean_number(row.get('excursions')),
            'loss': _clean_number(row.get('loss')),
            'split': split,
            'included': included,
            'exclude_reason': reason,
        })

    train_rows = [r for r in frozen_rows if r['included'] and r['split'] == 'train']
    holdout_rows = [r for r in frozen_rows if r['included'] and r['split'] == 'holdout']
    if not train_rows:
        raise ValueError('训练集没有同时具备特征和货损的样本')
    if not holdout_rows:
        raise ValueError('留出集没有同时具备特征和货损的样本')

    # 缺失填补和标准化参数只从训练集学习；留出集只应用这些参数。
    normalization = {}
    for feature in FEATURES:
        values = [float(r[feature]) for r in train_rows if _finite(r[feature])]
        mean = _mean(values)
        normalization[feature] = {
            'mean': round(mean, 12),
            'std': round(_std(values, mean), 12),
            'impute': round(mean, 12),
            'train_non_missing': len(values),
            'train_missing': len(train_rows) - len(values),
        }

    def design(rows_to_use):
        matrix, target = [], []
        for row in rows_to_use:
            vector = [1.0]
            for feature in FEATURES:
                stats = normalization[feature]
                value = float(row[feature]) if _finite(row[feature]) else stats['impute']
                scale = stats['std'] if stats['std'] > 0 else 1.0
                vector.append((value - stats['mean']) / scale)
            matrix.append(vector)
            target.append(float(row['loss']))
        return np.asarray(matrix, dtype=float), np.asarray(target, dtype=float)

    x_train, y_train = design(train_rows)
    x_holdout, y_holdout = design(holdout_rows)
    coefficients, *_ = np.linalg.lstsq(x_train, y_train, rcond=None)

    predicted_train = x_train @ coefficients
    predicted_holdout = x_holdout @ coefficients
    baseline = float(np.mean(y_train))
    baseline_holdout = np.full(len(y_holdout), baseline)
    baseline_holdout_mae = _mae(baseline_holdout, y_holdout)
    holdout_mae = _mae(predicted_holdout, y_holdout)
    holdout_sst = float(np.sum((y_holdout - np.mean(y_holdout)) ** 2))
    holdout_sse = float(np.sum((y_holdout - predicted_holdout) ** 2))
    r2 = 0.0 if holdout_sst <= 0 else 1.0 - holdout_sse / holdout_sst
    # 要求严格优于基线；等于基线时也拒绝上线。
    eligible = bool(holdout_mae < baseline_holdout_mae - 1e-9)
    improvement = 0.0 if baseline_holdout_mae <= 0 else (baseline_holdout_mae - holdout_mae) / baseline_holdout_mae

    coef = {
        'intercept': round(float(coefficients[0]), 12),
        'exposure': round(float(coefficients[1]), 12),
        'excursions': round(float(coefficients[2]), 12),
    }
    metrics = {
        'holdout_mae': round(holdout_mae, 6),
        'baseline_mae': round(baseline_holdout_mae, 6),
        'train_mae': round(_mae(predicted_train, y_train), 6),
        'holdout_rmse': round(_rmse(predicted_holdout, y_holdout), 6),
        'baseline_rmse': round(_rmse(baseline_holdout, y_holdout), 6),
        'holdout_r2': round(float(r2), 6),
        'improvement': round(float(improvement), 6),
        'baseline_mean': round(baseline, 6),
        'n_train': len(train_rows),
        'n_holdout': len(holdout_rows),
        'n_train_groups': len(train_groups),
        'n_holdout_groups': len(holdout_groups),
        'n_rows': len(rows),
        'n_excluded': len(frozen_rows) - len(train_rows) - len(holdout_rows),
        'eligible': eligible,
        'baseline': '训练集货损率均值；与模型使用同一批留出样本比较',
    }
    split = {
        'method': 'group-by-shipment',
        'seed': int(seed),
        'holdout_fraction': float(holdout_fraction),
        'train_shipments': sorted(train_groups),
        'holdout_shipments': sorted(holdout_groups),
        'train_samples': [r['sample_id'] for r in train_rows],
        'holdout_samples': [r['sample_id'] for r in holdout_rows],
        'groups_disjoint': not bool(train_groups & holdout_groups),
    }
    snapshot = {
        'frozen_at': None,
        'features': list(FEATURES),
        'target': TARGET,
        'seed': int(seed),
        'holdout_fraction': float(holdout_fraction),
        'rows': frozen_rows,
        'train_shipment_ids': sorted(train_groups),
        'holdout_shipment_ids': sorted(holdout_groups),
    }
    snapshot['input_hash'] = _canonical_hash({'rows': frozen_rows, 'features': list(FEATURES), 'seed': int(seed), 'holdout_fraction': float(holdout_fraction)})
    return {
        'snapshot': snapshot,
        'snapshot_hash': snapshot['input_hash'],
        'normalization': normalization,
        'coefficients': coef,
        'metrics': metrics,
        'split': split,
    }


def explain_prediction(normalization, coefficients, exposure, excursions):
    """使用单个已冻结版本的参数生成预测和逐因素风险解释。"""
    values = {'exposure': exposure, 'excursions': excursions}
    contributions = []
    predicted_raw = float(coefficients.get('intercept', 0.0))
    for feature in FEATURES:
        stats = normalization[feature]
        raw_value = values[feature]
        imputed = not _finite(raw_value)
        used_value = float(stats['impute']) if imputed else float(raw_value)
        scale = float(stats['std']) if float(stats['std']) > 0 else 1.0
        z = (used_value - float(stats['mean'])) / scale
        contribution = float(coefficients.get(feature, 0.0)) * z
        predicted_raw += contribution
        label = FEATURE_LABELS[feature]
        if imputed:
            direction = 'unknown'
            text = f'未填写{label}，按训练集均值 {float(stats["impute"]):.2f} 填补，该因素解释力有限'
        elif contribution > 0.05:
            direction = 'up'
            text = f'{label} {used_value:g} {FEATURE_UNITS[feature]}高于训练均值 {float(stats["mean"]):.2f}，预计提高货损率 {contribution:.2f} 个百分点'
        elif contribution < -0.05:
            direction = 'down'
            text = f'{label} {used_value:g} {FEATURE_UNITS[feature]}低于训练均值 {float(stats["mean"]):.2f}，预计降低货损率 {abs(contribution):.2f} 个百分点'
        else:
            direction = 'neutral'
            text = f'{label} {used_value:g} {FEATURE_UNITS[feature]}接近训练均值 {float(stats["mean"]):.2f}，影响较小'
        contributions.append({
            'feature': feature,
            'label': label,
            'value': used_value,
            'input_value': None if imputed else used_value,
            'imputed': imputed,
            'z': round(z, 6),
            'contribution': round(contribution, 6),
            'direction': direction,
            'text': text,
        })

    predicted = min(100.0, max(0.0, predicted_raw))
    ordered = sorted(contributions, key=lambda item: item['contribution'], reverse=True)
    positive = [item['label'] for item in ordered if item['contribution'] > 0.05 and not item['imputed']]
    if positive:
        summary = '主要风险来自' + '、'.join(positive[:2]) + f'；预计货损率 {predicted:.2f}%'
    elif any(item['imputed'] for item in contributions):
        summary = f'部分特征缺失，按训练集均值填补；预计货损率 {predicted:.2f}%'
    else:
        summary = f'未见明显高风险因素；预计货损率 {predicted:.2f}%'
    return {
        'predicted_loss': round(predicted, 4),
        'predicted_raw': round(predicted_raw, 6),
        'clamped': bool(predicted != predicted_raw),
        'risk_reasons': contributions,
        'summary': summary,
    }
