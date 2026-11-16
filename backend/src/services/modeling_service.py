"""训练、对比、上线与回滚的业务编排。

关键约束：
- 训练输入在训练时冻结进 ModelVersion.snapshot，之后不再回看数据库。
- 留出集按班次切分；缺失填补/标准化参数只从训练集学习。
- 留出误差未严格优于训练均值基线的版本标记 rejected，禁止上线。
- 模型文件与上线记录任一失败即回滚，旧版本保持上线。
- 发布使用数据库租约锁 + 版本 CAS，并发上线只接受一个版本。
"""
import logging
import os
import socket
import time
import uuid
from datetime import datetime, timezone

from sqlalchemy import update
from sqlalchemy.exc import IntegrityError

from src.config.database import Session
from src.models.entities import ModelDeployment, ModelLock, ModelState, ModelVersion, Prediction, Sample, Shipment
from src.services import artifacts
from src.services.modeling import FEATURES, TARGET, explain_prediction, train

ALGORITHM_VERSION = 'linear-regression-2f-1.0'


class ModelError(Exception):
    """带 HTTP 状态码的业务错误。"""

    def __init__(self, status, message):
        super().__init__(message)
        self.status = status
        self.message = message


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def row_dict(row):
    if row is None:
        return None
    return {c.name: getattr(row, c.name) for c in row.__table__.columns}


def display_state(version, state):
    """展示状态由唯一指针推导，避免历史版本残留 active 标签。"""
    if state is not None and state.version_id == version.id:
        return 'active'
    if version.state == 'rejected':
        return 'rejected'
    return 'deployed' if version.state == 'retired' else version.state


def version_payload(version, include_input=False, state=None):
    payload = {
        'id': version.id,
        'label': version.label,
        'algorithm': version.algorithm,
        'state': display_state(version, state),
        'eligible': bool(version.eligible),
        'snapshot_hash': version.snapshot_hash,
        'seed': version.seed,
        'holdout_fraction': version.holdout_fraction,
        'metrics': version.metrics or {},
        'split': version.split or {},
        'coefficients': version.coefficients or {},
        'normalization': version.normalization or {},
        'reason': version.reason or '',
        'created': version.created,
        'created_by': version.created_by,
    }
    if include_input:
        payload['frozen_input'] = (version.snapshot or {}).get('rows', [])
        payload['frozen_at'] = (version.snapshot or {}).get('frozen_at')
    return payload


def deployment_payload(row):
    return {
        'id': row.id,
        'version_id': row.version_id,
        'previous_version_id': row.previous_version_id,
        'action': row.action,
        'request_key': row.request_key,
        'actor': row.actor,
        'note': row.note or '',
        'artifact_path': row.artifact_path,
        'created': row.created,
    }


def prediction_payload(row):
    return {
        'id': row.id,
        'sample_id': row.sample_id,
        'shipment_id': row.shipment_id,
        'version_id': row.version_id,
        'version_label': row.version_label,
        'algorithm': row.algorithm,
        'snapshot_hash': row.snapshot_hash,
        'input': row.input,
        'output': row.output,
        'actual_loss': row.actual_loss,
        'actor': row.actor,
        'created': row.created,
    }


def load_training_rows(session):
    """一次性读取训练输入。冷冻后模型不再回看数据库。"""
    rows = []
    query = (
        session.query(Sample, Shipment)
        .outerjoin(Shipment, Sample.shipment_id == Shipment.id)
        .order_by(Sample.id)
    )
    for sample, shipment in query:
        rows.append({
            'sample_id': sample.id,
            'sample_code': sample.code,
            'shipment_id': sample.shipment_id,
            'shipment_code': shipment.code if shipment else None,
            'zone': shipment.zone if shipment else None,
            'exposure': sample.exposure,
            'excursions': sample.excursions,
            'loss': sample.loss,
        })
    return rows


def _next_label(session):
    count = session.query(ModelVersion).count()
    while True:
        count += 1
        label = f'M-{count:04d}'
        if not session.query(ModelVersion).filter_by(label=label).first():
            return label


def create_version(session, actor, seed=3102, holdout_fraction=0.2):
    try:
        seed = int(seed)
        holdout_fraction = float(holdout_fraction)
        trained = train(load_training_rows(session), seed=seed, holdout_fraction=holdout_fraction)
    except ValueError as error:
        raise ModelError(409, str(error))
    trained['snapshot']['frozen_at'] = now_iso()
    metrics = trained['metrics']
    version = ModelVersion(
        label=_next_label(session),
        algorithm=ALGORITHM_VERSION,
        state='candidate' if metrics['eligible'] else 'rejected',
        eligible=bool(metrics['eligible']),
        snapshot_hash=trained['snapshot_hash'],
        seed=seed,
        holdout_fraction=holdout_fraction,
        snapshot=trained['snapshot'],
        normalization=trained['normalization'],
        coefficients=trained['coefficients'],
        metrics=metrics,
        split=trained['split'],
        reason='' if metrics['eligible'] else (
            f'留出集 MAE {metrics["holdout_mae"]:.4f} 未优于均值基线 '
            f'{metrics["baseline_mae"]:.4f}，拒绝上线'
        ),
        created=now_iso(),
        created_by=actor,
    )
    for attempt in range(3):
        try:
            session.add(version)
            session.commit()
            return version
        except IntegrityError:
            session.rollback()
            if attempt == 2:
                raise ModelError(409, '训练版本号冲突，请重试')
            version.label = _next_label(session)
    raise ModelError(409, '训练版本号冲突，请重试')


def list_versions(session):
    return session.query(ModelVersion).order_by(ModelVersion.id.desc()).all()


def reconcile_pointer(session):
    """启动时让 current.json 与数据库指针一致；数据库是唯一事实来源。"""
    try:
        state = current_state(session)
        if not state.version_id:
            if artifacts.current_file().exists():
                artifacts.remove_file(artifacts.current_file())
            return
        version = session.get(ModelVersion, state.version_id)
        if not version:
            return
        payload = {
            'version_id': version.id,
            'deployment_id': state.deployment_id,
            'label': version.label,
            'action': 'reconcile',
            'updated': state.updated,
        }
        current = None
        if artifacts.current_file().exists():
            try:
                current = artifacts.read_json(artifacts.current_file())
            except (OSError, ValueError):
                current = None
        if not current or current.get('version_id') != version.id or current.get('deployment_id') != state.deployment_id:
            artifacts.write_json_atomic(artifacts.current_file(), payload)
    except Exception:
        logging.exception('reconcile model pointer failed')


def current_state(session):
    state = session.get(ModelState, 1)
    if state:
        return state
    state = ModelState(id=1, version_id=None, deployment_id=None, updated='')
    session.add(state)
    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        state = session.get(ModelState, 1)
        if not state:
            raise
    return state


def current_model(session):
    state = current_state(session)
    if not state.version_id:
        return None, state
    return session.get(ModelVersion, state.version_id), state


def compare(session):
    current, state = current_model(session)
    deployments = session.query(ModelDeployment).order_by(ModelDeployment.id.desc()).limit(50).all()
    return {
        'current_version_id': state.version_id,
        'current_deployment_id': state.deployment_id,
        'current_label': current.label if current else None,
        'versions': [version_payload(v, state=state) for v in list_versions(session)],
        'deployments': [deployment_payload(d) for d in deployments],
    }


def _file_payload(version):
    return {
        'version_id': version.id,
        'label': version.label,
        'algorithm': version.algorithm,
        'snapshot_hash': version.snapshot_hash,
        'coefficients': version.coefficients,
        'normalization': version.normalization,
        'metrics': version.metrics,
        'split': version.split,
        'frozen_input_hash': (version.snapshot or {}).get('input_hash'),
        'features': list(FEATURES),
        'target': TARGET,
    }


def _lock(session, owner, ttl=30.0):
    deadline = time.time() + 5.0
    while True:
        now = time.time()
        try:
            existing = session.get(ModelLock, 'deploy', with_for_update=True)
            if existing is None:
                session.add(ModelLock(name='deploy', owner=owner, expires_at=now + ttl))
                session.flush()
                return True
            if not existing.expires_at or float(existing.expires_at) < now:
                changed = session.execute(
                    update(ModelLock)
                    .where(ModelLock.name == 'deploy', ModelLock.expires_at == existing.expires_at)
                    .values(owner=owner, expires_at=now + ttl)
                )
                session.flush()
                if changed.rowcount == 1:
                    return True
        except IntegrityError:
            session.rollback()
            return False
        if time.time() > deadline:
            return False
        time.sleep(0.05)


def _acquire():
    """独立事务取租约并立即提交；逻辑租约（owner + 过期时间）跨请求生效。"""
    owner = f'{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex}'
    lease = Session()
    try:
        if not _lock(lease, owner):
            raise ModelError(409, '另一个上线操作正在进行，请稍后重试')
        lease.commit()
    except Exception:
        lease.rollback()
        raise
    finally:
        lease.close()
    return owner


def _release(owner):
    lease = Session()
    try:
        lease.execute(
            update(ModelLock)
            .where(ModelLock.name == 'deploy', ModelLock.owner == owner)
            .values(owner='', expires_at=0.0)
        )
        lease.commit()
    except Exception:
        lease.rollback()
        logging.exception('release model lock failed')
    finally:
        lease.close()


def _write_pointer(version, deployment_id, action):
    artifacts.write_json_atomic(
        artifacts.current_file(),
        {
            'version_id': version.id,
            'deployment_id': deployment_id,
            'label': version.label,
            'action': action,
            'updated': now_iso(),
        },
    )


def _swap_pointer(session, version, action, build_record):
    """公共发布流程：先落模型文件与指针，再提交上线记录；失败恢复旧指针。"""
    version_file = artifacts.version_file(version.id)
    existed = version_file.exists()
    old_pointer = None
    if artifacts.current_file().exists():
        try:
            old_pointer = artifacts.current_file().read_bytes()
        except OSError:
            old_pointer = None
    state = session.get(ModelState, 1, with_for_update=True) or current_state(session)
    previous = state.version_id
    record = build_record(previous)
    session.add(record)
    try:
        session.flush()
        current_state_row = session.get(ModelState, 1)
        current_state_row.version_id = version.id
        current_state_row.deployment_id = record.id
        current_state_row.updated = now_iso()
        if previous and previous != version.id:
            old = session.get(ModelVersion, previous)
            if old is not None and old.state != 'rejected':
                old.state = 'retired'
        if version.state != 'rejected':
            version.state = 'active'
        session.flush()
        artifacts.write_json_atomic(version_file, _file_payload(version))
        _write_pointer(version, record.id, action)
        session.commit()
        return record
    except Exception:
        session.rollback()
        if not existed:
            referenced = session.query(ModelDeployment).filter_by(version_id=version.id).count()
            if not referenced:
                artifacts.remove_file(version_file)
        if old_pointer is None:
            artifacts.remove_file(artifacts.current_file())
        else:
            try:
                artifacts.write_bytes_atomic(artifacts.current_file(), old_pointer)
            except OSError:
                logging.exception('restore previous pointer failed')
        raise


def deploy(session, version_id, actor, request_key=None, note='', expected_version_id=None):
    request_key = (request_key or '').strip() or f'deploy-{uuid.uuid4().hex}'
    existing = session.query(ModelDeployment).filter_by(request_key=request_key).first()
    if existing:
        return existing, False
    before = current_state(session).version_id
    # 结束当前读事务：MySQL 可重复读会在首条 SELECT 固定快照，拿锁后必须重新开事务。
    session.commit()
    owner = _acquire()
    try:
        session.expire_all()
        existing = session.query(ModelDeployment).filter_by(request_key=request_key).first()
        if existing:
            return existing, False
        current = current_state(session).version_id
        if current != before:
            raise ModelError(409, '并发上线已被其他请求完成，本次未发布；请刷新后重新比较')
        version = session.get(ModelVersion, int(version_id))
        if not version:
            raise ModelError(404, '模型版本不存在')
        if not version.eligible or version.state == 'rejected':
            raise ModelError(409, '该版本留出误差未优于均值基线，禁止上线')
        state = current_state(session)
        if expected_version_id is not None and state.version_id != int(expected_version_id):
            raise ModelError(409, '当前上线版本已被其他操作改变，请刷新后重新比较再上线')
        if state.version_id == version.id:
            raise ModelError(409, '该版本已经是当前上线版本')

        def build(previous):
            return ModelDeployment(
                version_id=version.id,
                previous_version_id=previous,
                action='deploy',
                request_key=request_key,
                actor=actor,
                note=note,
                artifact_path=str(artifacts.version_file(version.id)),
                created=now_iso(),
            )

        try:
            record = _swap_pointer(session, version, 'deploy', build)
        except ModelError:
            raise
        except OSError:
            logging.exception('deploy write failed')
            raise ModelError(503, '模型文件或上线记录写入失败，旧版本保持上线，请重试')
        except IntegrityError:
            session.rollback()
            existing = session.query(ModelDeployment).filter_by(request_key=request_key).first()
            if existing:
                return existing, False
            raise ModelError(409, '同一发布请求已被处理，请刷新后重试')
        return record, True
    finally:
        _release(owner)


def rollback(session, target_version_id, actor, request_key=None, note='', expected_version_id=None):
    request_key = (request_key or '').strip() or f'rollback-{uuid.uuid4().hex}'
    existing = session.query(ModelDeployment).filter_by(request_key=request_key).first()
    if existing:
        return existing, False
    before = current_state(session).version_id
    session.commit()
    owner = _acquire()
    try:
        session.expire_all()
        existing = session.query(ModelDeployment).filter_by(request_key=request_key).first()
        if existing:
            return existing, False
        current = current_state(session).version_id
        if current != before:
            raise ModelError(409, '并发上线已被其他请求完成，本次未发布；请刷新后重试')
        if not target_version_id:
            raise ModelError(400, '缺少恢复目标版本')
        version = session.get(ModelVersion, int(target_version_id))
        if not version:
            raise ModelError(404, '恢复目标版本不存在')
        if not session.query(ModelDeployment).filter_by(version_id=version.id).first():
            raise ModelError(409, '只能回滚到曾经上线过的版本')
        state = current_state(session)
        if expected_version_id is not None and state.version_id != int(expected_version_id):
            raise ModelError(409, '当前上线版本已被其他操作改变，请刷新后重试')
        if state.version_id == version.id:
            raise ModelError(409, '该版本已经是当前上线版本')

        def build(previous):
            return ModelDeployment(
                version_id=version.id,
                previous_version_id=previous,
                action='rollback',
                request_key=request_key,
                actor=actor,
                note=note,
                artifact_path=str(artifacts.version_file(version.id)),
                created=now_iso(),
            )

        try:
            record = _swap_pointer(session, version, 'rollback', build)
        except ModelError:
            raise
        except OSError:
            logging.exception('rollback write failed')
            raise ModelError(503, '回滚记录写入失败，当前版本保持不变，请重试')
        except IntegrityError:
            session.rollback()
            existing = session.query(ModelDeployment).filter_by(request_key=request_key).first()
            if existing:
                return existing, False
            raise ModelError(409, '同一回滚请求已被处理，请刷新后重试')
        return record, True
    finally:
        _release(owner)


def predict(session, actor, input_rows):
    """给新样本预测；未填货损不影响，缺失特征按上线版本冻结参数填补。"""
    version, state = current_model(session)
    if not version or not state.version_id:
        raise ModelError(409, '还没有上线模型，无法预测；请先训练并上线一个通过留出误差检查的版本')
    if not input_rows:
        raise ModelError(400, '至少提供一行样本')
    parameters = {
        'coefficients': version.coefficients,
        'normalization': version.normalization,
        'algorithm': version.algorithm,
        'snapshot_hash': version.snapshot_hash,
    }
    results = []
    for row in input_rows:
        sample_id = row.get('sample_id')
        if sample_id is None and all(row.get(key) is None for key in ('shipment_id', 'exposure', 'excursions')):
            raise ModelError(400, '每笔预测至少给出样本号，或暴露时长/超温次数之一')
        sample = session.get(Sample, int(sample_id)) if sample_id is not None else None
        if sample_id is not None and not sample:
            raise ModelError(404, f'样本 {sample_id} 不存在')
        if sample:
            exposure = row.get('exposure', sample.exposure)
            excursions = row.get('excursions', sample.excursions)
            shipment_id = sample.shipment_id
        else:
            exposure = row.get('exposure')
            excursions = row.get('excursions')
            shipment_id = row.get('shipment_id')
        outcome = explain_prediction(version.normalization, version.coefficients, exposure, excursions)
        record = Prediction(
            sample_id=sample_id,
            shipment_id=shipment_id,
            version_id=version.id,
            version_label=version.label,
            algorithm=version.algorithm,
            snapshot_hash=version.snapshot_hash,
            input={
                'sample_id': sample_id,
                'shipment_id': shipment_id,
                'exposure': exposure,
                'excursions': excursions,
            },
            parameters=parameters,
            output=outcome,
            actual_loss=sample.loss if sample else None,
            actor=actor,
            created=now_iso(),
        )
        session.add(record)
        results.append(record)
    session.commit()
    return version, results


def prediction_history(session, limit=100):
    rows = session.query(Prediction).order_by(Prediction.id.desc()).limit(int(limit)).all()
    return [prediction_payload(r) for r in rows]


def deployment_history(session, limit=100):
    rows = session.query(ModelDeployment).order_by(ModelDeployment.id.desc()).limit(int(limit)).all()
    return [deployment_payload(r) for r in rows]
