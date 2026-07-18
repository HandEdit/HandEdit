from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from handedit_eval.bank import canonicalize_target_name, compact_descriptor, description_for_target, build_instruction, normalize_scope
from handedit_eval.manifest import write_manifest_jsonl

IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}


def index_by_id(root: Path, pattern: re.Pattern[str], recursive: bool) -> Dict[str, str]:
    if not root:
        return {}
    mapping: Dict[str, str] = {}
    if not root.exists():
        return mapping
    iterator = root.rglob('*') if recursive else root.glob('*')
    for path in iterator:
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        match = pattern.search(path.stem)
        if not match:
            continue
        sample_id = match.group(1)
        mapping[sample_id] = str(path.resolve())
    return mapping


def load_json_mapping(path: str, pattern: re.Pattern[str]) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, 'r', encoding='utf-8') as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, list):
        mapping: Dict[str, Any] = {}
        for item in payload:
            if not isinstance(item, dict):
                continue
            if 'id' in item:
                mapping[str(item['id'])] = item
                continue
            file_name = str(item.get('file_name', '') or item.get('path', ''))
            match = pattern.search(Path(file_name).stem)
            if match:
                mapping[match.group(1)] = item
        return mapping
    raise ValueError('JSON mapping must be a dict or a list.')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Build a HandEdit JSONL manifest from image folders.')
    parser.add_argument('--src-root', required=True)
    parser.add_argument('--pred-root', required=True)
    parser.add_argument('--out-manifest', required=True)
    parser.add_argument('--gt-root', default='')
    parser.add_argument('--test-mask-root', default='')
    parser.add_argument('--gt-mask-root', default='')
    parser.add_argument('--human-mask-root', default='')
    parser.add_argument('--robot-mask-root', default='')
    parser.add_argument('--bg-mask-root', default='')
    parser.add_argument('--object-mask-root', default='')
    parser.add_argument('--urdf-refs', '--urdf-ref-paths', dest='urdf_ref_paths', nargs='*', default=None)
    parser.add_argument('--urdf-masks', '--urdf-mask-paths', dest='urdf_mask_paths', nargs='*', default=None)
    parser.add_argument('--instruction-file', default='')
    parser.add_argument('--metadata-file', default='')
    parser.add_argument('--default-instruction', default='')
    parser.add_argument('--dataset', default='HandEdit')
    parser.add_argument('--task', default='handedit')
    parser.add_argument('--replacement-scope', default='hand-only', choices=['hand-only', 'hand-arm'])
    parser.add_argument('--target-name', default='')
    parser.add_argument('--id-pattern', default=r'(\d{6})')
    parser.add_argument('--recursive', action='store_true')
    parser.add_argument('--require-gt', action='store_true')
    parser.add_argument('--require-roi', action='store_true')
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pattern = re.compile(args.id_pattern)

    src_map = index_by_id(Path(args.src_root), pattern, args.recursive)
    pred_map = index_by_id(Path(args.pred_root), pattern, args.recursive)
    gt_map = index_by_id(Path(args.gt_root), pattern, args.recursive) if args.gt_root else {}
    test_mask_map = index_by_id(Path(args.test_mask_root), pattern, args.recursive) if args.test_mask_root else {}
    gt_mask_map = index_by_id(Path(args.gt_mask_root), pattern, args.recursive) if args.gt_mask_root else {}
    human_map = index_by_id(Path(args.human_mask_root), pattern, args.recursive) if args.human_mask_root else {}
    robot_map = index_by_id(Path(args.robot_mask_root), pattern, args.recursive) if args.robot_mask_root else {}
    bg_map = index_by_id(Path(args.bg_mask_root), pattern, args.recursive) if args.bg_mask_root else {}
    object_map = index_by_id(Path(args.object_mask_root), pattern, args.recursive) if args.object_mask_root else {}

    instruction_map = load_json_mapping(args.instruction_file, pattern)
    metadata_map = load_json_mapping(args.metadata_file, pattern)

    rows: List[Dict[str, Any]] = []
    for sample_id, pred_path in sorted(pred_map.items()):
        src_path = src_map.get(sample_id)
        gt_path = gt_map.get(sample_id, '')
        human_path = human_map.get(sample_id, '')
        robot_path = robot_map.get(sample_id, '')
        test_mask_path = test_mask_map.get(sample_id, '')
        gt_mask_path = gt_mask_map.get(sample_id, '')
        if not src_path:
            continue
        if args.require_gt and not gt_path:
            continue
        if args.require_roi and not (human_path or robot_path):
            continue

        metadata = metadata_map.get(sample_id, {})
        if not isinstance(metadata, dict):
            metadata = {'metadata': metadata}

        replacement_scope = normalize_scope(str(metadata.get('replacement_scope', args.replacement_scope) or args.replacement_scope))
        target_name = canonicalize_target_name(str(metadata.get('target_name', args.target_name) or args.target_name))
        target_description = str(metadata.get('target_description', '') or description_for_target(target_name, replacement_scope))
        descriptor = str(metadata.get('compact_descriptor', '') or compact_descriptor(target_description))
        instruction = str(instruction_map.get(sample_id, '') or metadata.get('instruction', '') or args.default_instruction)
        if not instruction and target_name:
            instruction = build_instruction(replacement_scope, target_name, descriptor)

        row = {
            'id': sample_id,
            'dataset': metadata.get('dataset', args.dataset),
            'task': metadata.get('task', args.task),
            'replacement_scope': replacement_scope,
            'target_name': target_name,
            'instruction': instruction,
            'src_path': src_path,
            'pred_path': pred_path,
            'gt_path': gt_path,
            'test_mask_path': test_mask_path,
            'gt_mask_path': gt_mask_path,
            'human_mask_path': human_path,
            'robot_mask_path': robot_path,
            'bg_mask_path': bg_map.get(sample_id, ''),
            'object_mask_path': object_map.get(sample_id, ''),
            'urdf_ref_paths': list(args.urdf_ref_paths or []),
            'urdf_mask_paths': list(args.urdf_mask_paths or []),
        }
        row.update(metadata)
        row.pop('track', None)
        row.pop('embodiment_group', None)
        rows.append(row)

    write_manifest_jsonl(args.out_manifest, rows)
    print(f'[OK] Wrote {len(rows)} records to {args.out_manifest}')


if __name__ == '__main__':
    main()
