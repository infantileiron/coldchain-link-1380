"""模型产物与当前上线指针的原子写入。"""
import json
import logging
import os
import uuid
from pathlib import Path


def artifact_root():
    root = os.environ.get('MODEL_ARTIFACT_DIR')
    if not root:
        root = os.path.join(os.environ.get('ARTIFACT_DIR', '/artifacts'), 'models')
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    return path


def version_file(version_id):
    return artifact_root() / f'v{int(version_id)}.json'


def current_file():
    return artifact_root() / 'current.json'


def write_bytes_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    try:
        with open(tmp, 'wb') as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        _fsync_dir(path.parent)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def write_json_atomic(path, payload):
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, default=str).encode('utf-8')
    write_bytes_atomic(path, raw)


def read_json(path):
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def remove_file(path):
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        logging.exception('remove model artifact failed')


def ensure_artifact_root_writable():
    """启动时报出不可写目录，避免发布时才发现模型文件无法落盘。"""
    root = artifact_root()
    probe = root / f'.write-probe-{uuid.uuid4().hex}'
    try:
        with open(probe, 'wb') as handle:
            handle.write(b'ok')
        probe.unlink()
    except OSError:
        raise RuntimeError(f'模型产物目录不可写：{root}')
    return root


def _fsync_dir(path):
    try:
        descriptor = os.open(path, os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        pass
