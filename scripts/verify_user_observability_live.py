"""Opt-in native workspace acceptance against retained synthetic quality data.

Prepare the synthetic account/KB using verify_langfuse_quality_live.py first.
--new-evaluation runs the existing real retrieval benchmark once per work dir.
--judge explicitly invokes the configured model (and incurs model usage).
Credentials are read only from the ignored quality work directory, never logged.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import uuid

from verify_langfuse_quality_live import Verification, check


def verify(args):
    args.phase, args.model_id = 'trace', None
    original_dir = args.work_dir
    args.work_dir = args.quality_dir
    check((args.work_dir/'quality-private.json').exists(), 'Prepare the synthetic quality fixture first')
    fixture = Verification(args)
    args.work_dir = original_dir
    args.work_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.work_dir/'report.json'
    report = json.loads(report_path.read_text(encoding='utf-8')) if report_path.exists() else {}
    def save(): report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    try:
        fixture.login()
        request = fixture.request
        rows = request('GET','/observability/traces',params={'window':'all','page_size':100})
        trace = next((row for row in rows['items'] if row['observability_trace_id'] == fixture.state['trace_id']), None)
        check(trace, 'Saved question/answer execution is absent from native trace list')
        detail = request('GET',f"/observability/traces/{trace['id']}")
        check(detail['answer_message_id'] == fixture.state['message_id'], 'Wrong answer/trace association')
        check(detail['answer'] and detail['events'], 'Native trace did not contain saved answer and events')
        remote = request('GET',f"/observability/traces/{trace['id']}/observations")
        check(remote['state'] == 'available', 'Native observation read was not available: '+remote['state'])
        check(any(row['type']=='RETRIEVER' for row in remote['observations']), 'No actual remote retrieval observations')
        check(any(row['type']=='GENERATION' for row in remote['observations']), 'No actual remote model observations')
        check(fixture.state['password'] not in json.dumps(remote), 'Credential found in native observation output')
        check('sk-synthetic-observability-canary' not in json.dumps(remote), 'Synthetic secret canary was not masked')
        summary = request('GET','/observability/overview',params={'window':'all','conversation_id':detail['conversation_id']})
        check(summary['total'] >= 1 and summary['cost_usd'] is None, 'Overview totals/cost do not match the contract')
        check(any(s['name']=='user-feedback' for s in detail['scores']), 'Native feedback score was not linked')
        report['trace'] = {'id':trace['id'],'remote_id':remote['trace_id'], 'observations':len(remote['observations']),
            'events':len(detail['events']),'feedback':detail['feedback'], 'native_url':f"http://localhost:5173/observability?tab=traces&trace={trace['id']}"}
        # Reuse a separately owned synthetic account to verify real PostgreSQL authorization.
        other_path=Path('tmp/agent-management-live/private.json')
        if other_path.exists():
            other=json.loads(other_path.read_text(encoding='utf-8'))
            login=fixture.api.post('/auth/login',json={'username':other['username'],'password':other['password']})
            login.raise_for_status()
            foreign=fixture.api.get(f"/observability/traces/{trace['id']}/observations",headers={'Authorization':'Bearer '+login.json()['access_token']})
            check(foreign.status_code == 404,'Another user could read private trace observations')
            report['cross_user_trace'] = '404'
        previous = request('GET',f"/evaluation/runs/{report['new_run_id']}") if report.get('new_run_id') else None
        if args.new_evaluation and (not previous or any(item.get('retrieval_error') for item in previous['details'])):
            if previous:
                report.setdefault('failed_run_ids',[]).append(previous['id'])
            run=request('POST',f"/evaluation/benchmarks/{fixture.state['benchmark_id']}/runs",json={'name':'Orion native observability 20261103'})
            report['new_run_id']=run['id'];save()
            check(run['metrics']['execution_started_at'] < run['metrics']['execution_finished_at'], 'Actual evaluation timing not persisted')
        run_id=str(args.run_id) if args.run_id else report.get('new_run_id') or fixture.state['evaluation']['id']
        runs=request('GET','/evaluation/runs')
        check(len(runs)>=2 if args.new_evaluation else bool(runs), 'No comparable historical runs')
        check(all('observability' in run and 'benchmark_version' in run for run in runs), 'Missing experiment summary metadata')
        run=request('GET',f'/evaluation/runs/{run_id}')
        item=request('GET',f'/observability/evaluations/{run_id}/items/0')
        if item['remote']['state']=='pending_index':
            item=fixture.wait('native evaluation observation indexing',lambda: (lambda value:value if value['remote']['state']=='available' else None)(request('GET',f'/observability/evaluations/{run_id}/items/0')))
        check(item['remote']['state']=='available', 'Native evaluation item observations unavailable')
        check(item['remote']['trace_id']==run['details'][0]['observability']['trace_id'], 'Wrong evaluation trace')
        check(not any(case.get('retrieval_error') for case in run['details']), 'Real retrieval failed; restore the embedding service before retrying')
        check(len(item['scores'])>=5, 'Expected five actual retrieval scores per case')
        if args.judge:
            result=request('POST',f"/observability/traces/{trace['id']}/judge",json={'model_id':fixture.state['model_id'],'request_id':str(uuid.uuid4())})
            check(result['state']=='completed','Judge did not persist a result')
        detail=request('GET',f"/observability/traces/{trace['id']}")
        if detail.get('judgement',{}):
            check(detail['judgement']['state']=='completed','Latest judge is not complete')
            expected={'judge.answer_relevance','judge.evidence_support'}
            def indexed_scores():
                scores=fixture.lf.api.scores.get_many(trace_id=remote['trace_id'],fields='details,subject').data
                return scores if expected.issubset({s.name for s in scores}) else None
            scores=fixture.wait('indexed native Judge Scores',indexed_scores)
            report['judge']={'result':detail['judgement']['result'],'remote_scores':[
                {'name':s.name,'value':s.value,'subject_kind':s.subject.kind,'trace_id':s.subject.id} for s in scores if s.name in expected]}
        experiment_id=run['metrics']['observability']['experiment_id']
        start=datetime.now(timezone.utc)-timedelta(days=7)
        def indexed_experiment():
            items=fixture.lf.api.experiments.list_items(from_start_time=start,experiment_id=experiment_id,fields='core,io,scores').data
            return items if len(items)==len(run['details']) and sum(len(i.scores or []) for i in items)>=10 else None
        experiment_items=fixture.wait('indexed native Experiment items and Scores',indexed_experiment)
        report['evaluation']={'run_id':run_id,'runs_available':len(runs),'items':len(experiment_items),
            'scores':sum(len(i.scores or []) for i in experiment_items),'item_observations':len(item['remote']['observations']),
            'recall_at_k':run['recall_at_k'],'mrr_at_k':run['mrr_at_k'],
            'native_url':f'http://localhost:5173/observability?tab=evaluations&run={run_id}&item=0'}
        report['checked_at']=datetime.now(timezone.utc).isoformat();save()
        print('Native user observability acceptance passed. Report: '+str(report_path),flush=True)
    finally:
        fixture.api.close();fixture.lf.shutdown()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live',action='store_true')
    parser.add_argument('--backend',default='http://localhost:8000')
    parser.add_argument('--quality-dir',type=Path,default=Path('tmp/langfuse-quality-live'))
    parser.add_argument('--work-dir',type=Path,default=Path('tmp/user-observability-live'))
    parser.add_argument('--new-evaluation',action='store_true')
    parser.add_argument('--judge',action='store_true')
    parser.add_argument('--run-id',type=uuid.UUID,help='Verify a specific owned run, e.g. one created in the UI')
    args=parser.parse_args()
    if not args.live: parser.error('--live is required for local synthetic acceptance')
    verify(args)


if __name__=='__main__': main()
