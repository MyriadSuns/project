# -*- coding: utf-8 -*-
"""推理链模块"""

import re
from .prompt_templates import build_reasoning_prompt

_TRUNC_END = re.compile(r'(因为|。理由|理由[:：])$')


def _looks_truncated(text):
    if not text or not text.strip():
        return True
    stripped = text.strip()
    return len(stripped) < 5 or bool(_TRUNC_END.search(stripped))


def parse_triple_reasoning_output(response):
    result = {
        'core_fact': '',
        'authority_evidence': '',
        'consistency_judgment': '',
        'raw': response or ''
    }
    if not response or not response.strip():
        return result

    text = response.strip()

    patterns_core_fact = [
        r'【新闻核心事实】\s*([\s\S]+?)(?=【权威证据】|【一致性判断】|【综合分析】|【最终结论】|$)',
        r'新闻核心事实[：:]\s*([\s\S]+?)(?=权威证据|一致性判断|综合分析|最终结论|$)',
    ]
    for pattern in patterns_core_fact:
        m = re.search(pattern, text)
        if m:
            result['core_fact'] = m.group(1).strip()
            break

    patterns_evidence = [
        r'【权威证据】\s*([\s\S]+?)(?=【一致性判断】|【综合分析】|【最终结论】|$)',
        r'权威证据[：:]\s*([\s\S]+?)(?=一致性判断|综合分析|最终结论|$)',
    ]
    for pattern in patterns_evidence:
        m = re.search(pattern, text)
        if m:
            result['authority_evidence'] = m.group(1).strip()
            break

    patterns_judgment = [
        r'【一致性判断】\s*([\s\S]+?)(?=【综合分析】|【最终结论】|【新闻核心事实】|【权威证据】|$)',
        r'一致性判断[：:]\s*([\s\S]+?)(?=综合分析|最终结论|【综合分析】|【最终结论】|$)',
        r'一致性判断\n(一致|不一致|真实|虚假|无法确定)[。，]?\n?理由[：:]\s*([\s\S]+?)(?=\n\n|\Z|综合分析|最终结论)',
        r'一致性判断\n(一致|不一致|真实|虚假|无法确定)[。，]?\n?(.+?)(?=\n\n|\Z|综合分析|最终结论)',
        r'判断结果[：:]\s*(一致|不一致|真实|虚假|无法确定)[。，]?\s*主要理由[：:]\s*([\s\S]+?)(?=综合分析|最终结论|【综合分析】|【最终结论】|$)',
    ]
    for pattern in patterns_judgment:
        m = re.search(pattern, text)
        if m:
            if len(m.groups()) == 2:
                result['consistency_judgment'] = f"{m.group(1)}。主要理由：{m.group(2).strip()}"
            else:
                result['consistency_judgment'] = m.group(1).strip()
            break

    return result


def format_triple_display(parsed):
    return (
        f"【新闻核心事实】{parsed.get('core_fact', '')} "
        f"【权威证据】{parsed.get('authority_evidence', '')} "
        f"【一致性判断】{parsed.get('consistency_judgment', '')}"
    )


def parse_reasoning_output(response):
    result = {'reason': '', 'raw': response or ''}
    if not response or not response.strip():
        return result
    text = response.strip()
    
    m = re.search(r'主要理由[：:]\s*(.+?)(?=\n\n|\Z)', text, re.DOTALL)
    if m:
        result['reason'] = m.group(1).strip()
    else:
        parts = [p.strip() for p in text.split('\n') if p.strip()]
        for i, p in enumerate(parts):
            if '理由' in p or (i > 2 and len(p) > 20):
                result['reason'] = p
                break
        if not result['reason'] and parts:
            result['reason'] = parts[-1][:500]
    return result


def reasoning_chain(llm_wrapper, text, authority_summary="", **gen_kwargs):
    prompt = build_reasoning_prompt(text, authority_summary=authority_summary)
    gen_kwargs.setdefault('max_new_tokens', 1600)
    gen_kwargs.setdefault('do_sample', False)

    response = llm_wrapper.chat_single_turn(
        prompt,
        system="你是一个虚假新闻检测专家，请严格按照指定格式输出分析结果。",
        **gen_kwargs
    )

    parsed = parse_triple_reasoning_output(response)

    if _looks_truncated(parsed.get('consistency_judgment', '')):
        jm = re.match(r'(一致|不一致|无法确定)', parsed.get('consistency_judgment', '').strip())
        verdict = jm.group(1) if jm else None
        context = f"【新闻核心事实】{parsed.get('core_fact') or text}\n【权威证据】{parsed.get('authority_evidence') or authority_summary}"
        candidate = ''
        response2 = ''
        for attempt in (0, 1):
            if verdict:
                if attempt:
                    followup_prompt = (
                        f"判定结果为：{verdict}。\n"
                        f"直接写判断理由，一句话即可。只写理由内容，不要写判定词，不要写'主要理由：'，不要换行。\n\n"
                        f"{context}"
                    )
                else:
                    followup_prompt = (
                        f"判定结果为：{verdict}。\n"
                        "请用一句完整的话说明判断理由：权威证据与新闻文本的核心内容是否一致、为何得出该判断。\n"
                        "只输出理由本身，不要重复判定词，也不要写'主要理由：'前缀。\n\n"
                        f"{context}"
                    )
            else:
                followup_prompt = (
                    "基于以下两部分内容，只输出【一致性判断】。\n"
                    "格式：一致/不一致/无法确定。主要理由：<完整的判断依据，至少一句话>\n"
                    "禁止只输出判定词而不写理由。\n\n"
                    f"{context}"
                )
            response2 = (llm_wrapper.chat_single_turn(
                followup_prompt,
                system="你是一个虚假新闻检测专家。",
                max_new_tokens=512,
                do_sample=False,
            ) or '').strip()
            if response2.startswith('【一致性判断】'):
                response2 = response2[len('【一致性判断】'):].lstrip('：: ')
            if verdict:
                reason = response2
                if reason.startswith(verdict):
                    reason = reason[len(verdict):].lstrip('。，！!：: ')
                for pre in ('主要理由：', '主要理由:'):
                    if reason.startswith(pre):
                        reason = reason[len(pre):].lstrip('：: ')
                        break
                candidate = f'{verdict}。主要理由：{reason.strip()}'
            else:
                parsed2 = parse_triple_reasoning_output(f"【一致性判断】{response2}")
                candidate = parsed2.get('consistency_judgment', '')
            if candidate and not _looks_truncated(candidate):
                break
        if candidate and not _looks_truncated(candidate):
            parsed['consistency_judgment'] = candidate
            parsed['raw'] = response + '\n' + response2

    parsed['reason'] = parsed.get('consistency_judgment', '')
    return parsed


def get_prediction_and_reason(parsed):
    return parsed.get('reason', '')
