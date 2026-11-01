"""Opt-in live agent/provider acceptance using synthetic data and a real model.

Credentials are kept in ignored tmp/agent-management-live/private.json.
The selected built-in model connection is copied into this synthetic owner's
existing custom-model store; secrets are never written to the public report.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import secrets
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def verify(args):
    import httpx
    from app.llm.models import get_chat_model_profile
    folder = args.work_dir
    folder.mkdir(parents=True, exist_ok=True)
    private = folder / "private.json"
    state = json.loads(private.read_text(encoding="utf-8")) if private.exists() else {}
    report = state.setdefault("report", {})
    def save():
        private.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        (folder / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    def check(condition, message):
        if not condition:
            raise AssertionError(message)
    with httpx.Client(base_url=args.backend.rstrip('/') + '/api/v1',timeout=240,trust_env=False) as api:
        def request(method, path, **kwargs):
            response = api.request(method,path,**kwargs)
            if response.is_error:
                (folder / 'last-error.txt').write_text(response.text, encoding='utf-8')
                raise RuntimeError(f'{method} {path}: HTTP {response.status_code}')
            return response.json() if response.content else None
        new = 'username' not in state
        if new:
            state.update(username='agents_live_' + uuid.uuid4().hex[:10],password=secrets.token_urlsafe(24))
            save()
        identity = request('POST','/auth/register' if new else '/auth/login',json={key:state[key] for key in ('username','password')})
        api.headers['Authorization'] = 'Bearer ' + identity['access_token']
        report['user_id'] = identity['user_id']
        profile = get_chat_model_profile(args.model_id)
        body = dict(name='管理页验收模型',provider_name='验收连接',provider_type=profile.provider_type,
                    base_url=profile.base_url,model_name=profile.model,api_key=profile.api_key,
                    requires_api_key=profile.requires_api_key,temperature=0,supports_vision=profile.supports_vision)
        if 'model_id' not in state:
            state['model_id'] = request('POST','/chat/models',json=body)['id']; save()
        updated = request('PUT','/chat/models/' + state['model_id'],json={**body,'api_key':'','temperature':0.1})
        detail = request('GET',f"/chat/models/{state['model_id']}/configuration")
        check(detail['temperature'] == 0.1 and (detail['has_api_key'] or not profile.requires_api_key), 'Model update did not retain credentials')
        check('api_key' not in detail, 'Model detail returned credentials')
        report['model'] = dict(id=state['model_id'],blank_key_preserved=True,temperature=0.1)
        specs = {
            'child':dict(name='计算核验员',slug='verification-child',kind='subagent',model_id=state['model_id'],
                description='用于管理页验收：计算并核验结果。',tool_names=['calculator'],
                system_prompt='你是计算核验员。任何算术任务必须先调用 calculator 工具；最终回答以 AGENT-MANAGER-CHILD 开头，再给出工具的计算结果。'),
            'main':dict(name='验收智能助手',slug='verification-main',model_id=state['model_id'],tool_names=[],
                description='用于管理页验收：独立角色与模型配置。',system_prompt='无论用户问什么，只回答固定短语 AGENT-MANAGER-MAIN，不要添加其他内容。'),
        }
        for key, spec in specs.items():
            if key not in state:
                state[key] = request('POST','/agents',json=spec); save()
        delegation = dict(name='协作计算助手',slug='verification-coordinator',mode='deepagents',model_id=state['model_id'],tool_names=[],
            subagent_ids=[state['child']['id']],description='用于管理页验收：主智能体委派个人子智能体。',
            system_prompt='每个用户任务都必须先通过 task 工具委派给 verification-child。你没有计算权限，不要自行计算。最终原样保留子智能体结果中的 AGENT-MANAGER-CHILD 标记与数值。')
        if 'coordinator' not in state:
            state['coordinator'] = request('POST','/agents',json=delegation); save()
        # Enable/disable round-trip and rejection before a conversation is created.
        main = state['main']
        request('PUT','/agents/' + main['id'],json={**main,'enabled':False})
        denied = api.post('/chat/send',json={'query':'disabled probe','agent_id':main['id']})
        check(denied.status_code == 409,'Disabled agent was still admitted')
        request('PUT','/agents/' + main['id'],json={**main,'enabled':True})
        report['disabled_rejected'] = True
        catalog = request('GET','/agents')['agents']
        check({state[key]['id'] for key in ('main','child','coordinator')} <= {item['id'] for item in catalog}, 'Saved configurations disappeared on reload')
        report['persistent_agents'] = 3
        scenarios = [('main','只返回固定验收短语','AGENT-MANAGER-MAIN'),
                     ('coordinator','请委派 verification-child 使用计算器计算 13+29。','AGENT-MANAGER-CHILD'),
                     ('child','使用计算器计算 21+21。','AGENT-MANAGER-CHILD')]
        for key, question, marker in scenarios:
            print('Verifying actual ' + key + ' execution',flush=True)
            result_key = key + '_execution'
            if result_key not in state:
                state.setdefault(key + '_request_id', str(uuid.uuid4()))
                save()
                state[result_key] = request('POST','/chat/send',json={'query':question,'agent_id':state[key]['id'],
                    'request_id':state[key + '_request_id']})
                save()
            result = state[result_key]
            check(marker in result['answer'],f'{key} instruction did not reach the real model')
            check(result['agent_id'] == state[key]['id'] and result['model_id'] == state['model_id'],f'{key} used a different configuration')
            trace = request('GET','/chat/traces/' + result['trace_id'])
            events = json.dumps(trace['events'],ensure_ascii=False)
            if key != 'main':
                check('42' in result['answer'] and 'calculator' in events,f'{key} did not actually calculate')
            if key == 'coordinator':
                check('verification-child' in events and 'task' in events,'Coordinator did not delegate to the configured child')
            report[key] = {field:result[field] for field in ('agent_id','agent_name','model_id','conversation_id','answer','trace_id')}
            report[key]['events'] = len(trace['events'])
            save()
        print('Live agent/provider acceptance passed; report: ' + str(folder / 'report.json'),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live',action='store_true')
    parser.add_argument('--backend',default='http://localhost:8000')
    parser.add_argument('--model-id',default='deepseek-v4-flash')
    parser.add_argument('--work-dir',type=Path,default=Path('tmp/agent-management-live'))
    args = parser.parse_args()
    if not args.live:
        parser.error('--live is required to create synthetic configurations and call a real model')
    verify(args)
