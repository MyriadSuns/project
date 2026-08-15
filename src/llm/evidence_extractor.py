# -*- coding: utf-8 -*-
"""证据特征提取模块：
该模块实现了从 LLM 推理时最后一层隐藏层输出提取证据特征。"""

import torch
import torch.nn as nn
import re

from .reasoning_chain import reasoning_chain
from .search_engine import generate_search_query, get_search_results, build_search_items
from .filter_mechanism import triple_filter_mechanism

_global_proj_layers = {}

def _get_or_create_proj_layer(llm_hidden_size, evidence_dim, device):
    if isinstance(device, str):
        device = torch.device(device)
    device_key = device.type if hasattr(device, 'type') else str(device)
    key = (llm_hidden_size, evidence_dim, device_key)
    
    if key not in _global_proj_layers:
        proj = nn.Linear(llm_hidden_size, evidence_dim).to(device, dtype=torch.float32)
        gen = torch.Generator()
        gen.manual_seed(42)
        nn.init.xavier_uniform_(proj.weight, generator=gen)
        nn.init.zeros_(proj.bias)
        _global_proj_layers[key] = proj
    
    return _global_proj_layers[key]


class EvidenceFeatureExtractor(nn.Module):
    def __init__(self, llm_hidden_size, evidence_dim=256):
        super().__init__()
        self.llm_hidden_size = llm_hidden_size
        self.evidence_dim = evidence_dim
        self.proj = nn.Linear(llm_hidden_size, evidence_dim)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, hidden_states):
        if hidden_states.dim() == 3:
            h = hidden_states[:, -1, :]
        else:
            h = hidden_states
        evidence = self.proj(h)
        return evidence


def extract_evidence_from_llm(llm_wrapper, prompt,
                              llm_hidden_size=None, evidence_dim=256):
    try:
        llm_model = llm_wrapper.model
        tokenizer = llm_wrapper.tokenizer
        device = llm_wrapper.device
        
        if llm_hidden_size is None:
            llm_hidden_size = getattr(llm_model.config, 'hidden_size', 4096)
        
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        
        with torch.no_grad():
            outputs = llm_model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )
            
            last_hidden = outputs.hidden_states[-1]
            h = last_hidden[:, -1, :]
            h = h.to(torch.float32)
            
            proj = _get_or_create_proj_layer(llm_hidden_size, evidence_dim, device)
            evidence = proj(h)
        
        return evidence
    except Exception:
        device = llm_wrapper.device
        return torch.zeros(1, evidence_dim, device=device, dtype=torch.float32)


def proof_to_evidence_feat(proof, evidence_dim=256, llm_wrapper=None, news_text=None):
    proof = ''

    filter_reason = '未应用过滤'
    filtered_result = None
    
    if not proof and news_text:
        search_query = generate_search_query(llm_wrapper, news_text)
        web_results = get_search_results(search_query)
        search_items = build_search_items(web_results)
        
        if search_items:
            filtered_result, filter_reason = triple_filter_mechanism(
                news_text or "",
                search_items,
            )
        
        if filtered_result:
            title = filtered_result.get('title', '')
            content = filtered_result.get('content', '')
            url = filtered_result.get('url', '')

            if url:
                try:
                    from .search_engine import fetch_page_content
                    full_content = fetch_page_content(url)
                    if full_content and len(full_content) > 100:
                        invalid_keywords = ['安全验证', '网络不给力', '请稍后重试', '返回首页', '问题反馈']
                        if not any(kw in full_content for kw in invalid_keywords):
                            content = full_content
                except Exception:
                    pass
            
            proof = f"【权威信息】{title}\n{content}\n链接：{url}"
        else:
            proof = ""
    
    evidence_item = {
        'url': '',
        'title': proof,
        'content': proof,
        'publish_time': ''
    }
    
    url_pattern = r'https?://\S+'
    url_match = re.search(url_pattern, proof)
    if url_match:
        evidence_item['url'] = url_match.group(0)
    
    llm_retrieval_result = {
        'core_fact': '',
        'authority_evidence': '',
        'consistency_judgment': '',
        'raw': '',
        'filter_reason': filter_reason
    }
    
    if llm_wrapper is not None and proof:
        try:
            from .prompt_templates import build_authority_summary_prompt
            
            proof_cleaned = proof

            authority_summary_prompt = build_authority_summary_prompt(proof_cleaned)
            authority_summary = llm_wrapper.chat_single_turn(
                authority_summary_prompt,
                system="你是一个信息提炼专家。请基于提供的文本内容进行概括。",
                max_new_tokens=400,
                do_sample=False
            )
            
            if not authority_summary or not authority_summary.strip() or '无法访问' in authority_summary or '链接' in authority_summary:
                authority_summary = proof_cleaned.replace('【权威信息】', '').strip()[:500]

            authority_summary = (authority_summary or "").strip()[:600]

            reasoning_result = reasoning_chain(
                llm_wrapper,
                text=news_text or "",
                authority_summary=authority_summary,
                max_new_tokens=1600,
                do_sample=False
            )
            
            from .prompt_templates import build_reasoning_prompt
            prompt = build_reasoning_prompt(
                text=news_text or "",
                authority_summary=authority_summary
            )
            
            evidence_feat = extract_evidence_from_llm(
                llm_wrapper=llm_wrapper,
                prompt=prompt,
                evidence_dim=evidence_dim
            )
            
            llm_retrieval_result.update({
                'core_fact': reasoning_result.get('core_fact', ''),
                'authority_evidence': authority_summary,
                'consistency_judgment': reasoning_result.get('consistency_judgment', ''),
                'raw': reasoning_result.get('raw', '')
            })
            
            return evidence_feat.squeeze(0).to(torch.float32).to(llm_wrapper.device), llm_retrieval_result
        except Exception as e:
            llm_retrieval_result['consistency_judgment'] = '未检索到权威信息'
    
    if not proof:
        llm_retrieval_result['consistency_judgment'] = '未检索到权威信息'
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if llm_wrapper:
            device = llm_wrapper.device
        
        if news_text and llm_wrapper:
            try:
                search_query = generate_search_query(llm_wrapper, news_text)
                tokenizer = llm_wrapper.tokenizer
                inputs = tokenizer(search_query, return_tensors='pt', max_length=128, truncation=True).to(device)
                
                with torch.no_grad():
                    outputs = llm_wrapper.model(**inputs, output_hidden_states=True, return_dict=True)
                    query_hidden = outputs.hidden_states[-1][:, -1, :]
                    query_hidden = query_hidden.to(torch.float32)
                    
                    proj = _get_or_create_proj_layer(query_hidden.size(-1), evidence_dim, device)
                    feat = proj(query_hidden)
                    
                    return feat, llm_retrieval_result
            except Exception:
                pass
        
        feat = torch.zeros(evidence_dim, device=device, dtype=torch.float32)
        return feat, llm_retrieval_result
