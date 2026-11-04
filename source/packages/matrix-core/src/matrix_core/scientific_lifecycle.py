"""KEYMAKER terminal interface for evidence, artifacts, checkpoints and costs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ._scientific_records import fields, read_json
from .scientific_artifacts import ScientificArtifactRegistry
from .scientific_campaign import ScientificCampaign, capture_runtime_snapshot
from .scientific_costs import build_calibrated_plan, calibrate_cost
from .scientific_evidence import evaluate_scientific_plan, owner_validation_record, register_owner_evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description='KEYMAKER scientific result lifecycle; never submits calculations')
    parser.add_argument('action', choices=(
        'register-artifact', 'register-evidence', 'review', 'records', 'runs', 'runtime',
        'prepare', 'start', 'interrupt', 'fail', 'complete', 'reuse', 'calibrate',
        'calibrate-plan', 'remote-runtime', 'remote-handle', 'recover',
    ))
    parser.add_argument('--workspace', type=Path, help='Existing or new KEYMAKER project workspace')
    parser.add_argument('--request', type=Path, help='JSON argument object for the selected action')
    args = parser.parse_args(argv)
    try:
        data = read_json(args.request) if args.request else {}
        if not isinstance(data, dict):
            raise ValueError('request must be a JSON object of keyword arguments')
        if args.action == 'runtime':
            result = capture_runtime_snapshot(**data)
        elif args.action in {'remote-runtime', 'remote-handle', 'recover'}:
            from matrix_gui.keymaker_scientific_recovery import (
                capture_remote_resume_handle, probe_remote_runtime, recover_remote_job,
            )
            operations = {'remote-runtime': probe_remote_runtime, 'remote-handle': capture_remote_resume_handle,
                          'recover': recover_remote_job}
            result = operations[args.action](**data)
        else:
            if args.workspace is None:
                raise ValueError('--workspace is required for scientific state and artifact actions')
            registry = ScientificArtifactRegistry(args.workspace)
            campaign = ScientificCampaign(registry)
            if args.action == 'register-artifact':
                result = registry.register(**data)
            elif args.action == 'register-evidence':
                fields(data, ('plan', 'criterion_id', 'validation', 'metric_units',
                              'source_artifact_ids', 'identity', 'run_id'))
                data['validation'] = owner_validation_record(data['validation'])
                result = register_owner_evidence(registry, **data)
            elif args.action == 'review':
                result = evaluate_scientific_plan(registry=registry, **data)
            elif args.action in {'records', 'runs'}:
                fields(data, ())
                result = registry.records() if args.action == 'records' else campaign.runs()
            elif args.action == 'calibrate':
                result = calibrate_cost(campaign, **data)
            elif args.action == 'calibrate-plan':
                result = build_calibrated_plan(campaign=campaign, **data)
            else:
                operations = dict(prepare=campaign.prepare, start=campaign.start, interrupt=campaign.interrupt,
                                  fail=campaign.fail, complete=campaign.complete, reuse=campaign.reuse_decision)
                result = operations[args.action](**data)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, ImportError, OverflowError) as exc:
        parser.exit(2, f'KEYMAKER scientific lifecycle: {exc}\n')
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
