# -*- coding: utf-8 -*-
"""三重过滤机制模块"""

import os
import re
import math
import jieba
from urllib.parse import urlparse
from datetime import datetime, timedelta
from dateutil import parser


AUTHORITATIVE_DOMAINS = [
    'xinhuanet.com',
    'people.com.cn',
    'cctv.com',
    'cnr.cn',
    'china.com.cn',
    'gmw.cn',
    'cri.cn',
    'chinadaily.com.cn',
    'ce.cn',
    'stdaily.com',
    'news.cn',
    'xinhua.org',
    'people.cn',
    'sina.com.cn',
    '163.com',
    'qq.com',
    'ifeng.com',
    'sohu.com',
    'toutiao.com',
    'yzmedia.com.cn',
    'thepaper.cn',
    'guancha.cn',
    'jiemian.com',
    'moe.gov.cn',
    'edu.cn',
    'eol.cn',
    'cet.com.cn',
    'pep.com.cn',
    'most.gov.cn',
    'cas.cn',
    'cst.gov.cn',
    'sciencenet.cn',
    'ict.ac.cn',
    'ia.ac.cn',
    'sport.gov.cn',
    'sports.cn',
    'sports.sohu.com',
    'sports.sina.com.cn',
    'sports.qq.com',
    'gov.cn',
    'mfa.gov.cn',
    'ndrc.gov.cn',
    'miit.gov.cn',
    'mohrss.gov.cn',
    'moh.gov.cn',
    'mct.gov.cn',
    'mps.gov.cn',
    'mca.gov.cn',
    'mof.gov.cn',
    'mee.gov.cn',
    'mland.gov.cn',
    'mot.gov.cn',
    'maas.gov.cn',
    'bjnews.com.cn',
    'nbd.com.cn',
    'stcn.com',
    'southern.com',
    'eastday.com',
    'zjol.com.cn',
    'jsnews.com.cn',
    'ahwang.cn',
    'fjsen.com',
    'dzwww.com',
    'sdchina.com',
    'southcn.com',
    'yzwb.net',
    'sznews.com',
    'oeeee.com',
    'nxnews.net',
    'gxnews.com.cn',
    'hinews.cn',
    'cjn.cn',
    'hbnews.net',
    'hnrb.cn',
    'voc.com.cn',
    'dahe.cn',
    'cnhubei.com',
    'northnews.cn',
    'sxrb.com',
    'sjzdaily.com.cn',
    'hebei.com.cn',
    'tjyun.com',
    'northeast.cn',
    'hilizi.com',
    'jlsina.com',
    'ynet.com',
    'gog.cn',
    'scol.com.cn',
    'cqnews.net',
    'lanzhou.cn',
    'xjnews.cn',
    'xizangnews.com.cn',
    'cnstock.com',
    'cs.com.cn',
    'caixin.com',
    'ftchinese.com',
    'wallstreetcn.com',
    'yicai.com',
    'jinrongjie.com',
    'legaldaily.com.cn',
    'jcrb.com',
    'chinacourt.org',
    'moj.gov.cn',
    'procuratorate.gov.cn',
    'court.gov.cn',
    'ac.cn',
    'research.cn',
    'beijing.gov.cn',
    'shanghai.gov.cn',
    'tianjin.gov.cn',
    'chongqing.gov.cn',
    'guangdong.gov.cn',
    'guangxi.gov.cn',
    'hainan.gov.cn',
    'sichuan.gov.cn',
    'guizhou.gov.cn',
    'yunnan.gov.cn',
    'xizang.gov.cn',
    'shaanxi.gov.cn',
    'gansu.gov.cn',
    'qinghai.gov.cn',
    'ningxia.gov.cn',
    'xinjiang.gov.cn',
    'hebei.gov.cn',
    'shanxi.gov.cn',
    'neimenggu.gov.cn',
    'liaoning.gov.cn',
    'jilin.gov.cn',
    'heilongjiang.gov.cn',
    'jiangsu.gov.cn',
    'zhejiang.gov.cn',
    'anhui.gov.cn',
    'fujian.gov.cn',
    'jiangxi.gov.cn',
    'shandong.gov.cn',
    'henan.gov.cn',
    'hubei.gov.cn',
    'hunan.gov.cn',
    'cctv.cn',
    'cntv.cn',
    'cctv.com.cn',
    'cctvnews.cn',
    'cctv.com',
    'yangshipin.cn',
    'people.com.cn',
    'xinhuanet.com',
    'china.com.cn',
    'chinanews.com',
    'gmw.cn',
    'china.com.cn',
    'weibo.com',
    'weibo.cn',
    'weibo.qq.com',
]


def check_source_whitelist(url):
    try:
        if not url or not isinstance(url, str):
            return False
        m = re.search(r'https?://[^\)\'"\s]+', url)
        if m:
            url = m.group(0)

        parsed = urlparse(url)
        domain = (parsed.netloc or '').lower()
        if domain.startswith('www.'):
            domain = domain[4:]
        if domain in AUTHORITATIVE_DOMAINS:
            return True
        for auth_domain in AUTHORITATIVE_DOMAINS:
            if domain.endswith('.' + auth_domain) or domain == auth_domain:
                return True
        return False
    except Exception:
        return False


def filter_by_source(search_results):
    authoritative_results = []
    
    for result in search_results:
        url = result.get('url', '')
        if url and check_source_whitelist(url):
            result['authority_flag'] = 'strict_domain'
            authoritative_results.append(result)
    
    if not authoritative_results and search_results:
        for result in search_results:
            result['authority_flag'] = 'fallback'
            authoritative_results.append(result)
    
    return authoritative_results


def extract_time_from_text(text):
    times = []
    time_patterns = [
        (r'(\d{4})年(\d{1,2})月(\d{1,2})日', lambda m: f"{m.group(1)}-{m.group(2).zfill(2)}-{m.group(3).zfill(2)}"),
        (r'(\d{4})-(\d{1,2})-(\d{1,2})', lambda m: f"{m.group(1)}-{m.group(2).zfill(2)}-{m.group(3).zfill(2)}"),
        (r'(\d{1,2})月(\d{1,2})日', lambda m: f"2024-{m.group(1).zfill(2)}-{m.group(2).zfill(2)}"),
        (r'(\d{4})年(\d{1,2})月', lambda m: f"{m.group(1)}-{m.group(2).zfill(2)}-01"),
        (r'(\d{4})/(\d{1,2})/(\d{1,2})', lambda m: f"{m.group(1)}-{m.group(2).zfill(2)}-{m.group(3).zfill(2)}"),
        (r'(\d{2})/(\d{2})/(\d{4})', lambda m: f"{m.group(3)}-{m.group(1).zfill(2)}-{m.group(2).zfill(2)}"),
    ]
    
    for pattern, formatter in time_patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            try:
                time_str = formatter(match)
                times.append(parser.parse(time_str))
            except Exception:
                continue
    
    return times


def extract_time_from_article(article):
    if 'publish_time' in article:
        try:
            return parser.parse(article['publish_time'])
        except Exception:
            pass
    
    content = article.get('content', '') or article.get('title', '')
    times = extract_time_from_text(content)
    if times:
        return times[0]
    
    return None


def check_time_relevance(news_text, authoritative_article, threshold_months=3):
    news_times = extract_time_from_text(news_text)
    article_time = extract_time_from_article(authoritative_article)
    
    if not news_times or not article_time:
        return True
    
    news_time = min(news_times, key=lambda t: abs((t - article_time).days))
    time_diff = abs((news_time - article_time).days)
    
    if any(keyword in news_text for keyword in ['奥运会', '比赛', '赛事', '突发事件', '事故', '灾难', '紧急', '地震', '火灾', '爆炸']):
        return time_diff < 60
    elif any(keyword in news_text for keyword in ['政策', '通知', '条例', '法规', '规定', '办法', '措施', '方案']):
        return time_diff < 180
    elif any(keyword in news_text for keyword in ['科技', '突破', '研究', '成果', '发现', '发明', '技术', '创新']):
        return time_diff < 365
    elif any(keyword in news_text for keyword in ['文化', '艺术节', '展览', '演出', '活动', ' festival', '音乐会', '戏剧']):
        return time_diff < 180
    else:
        return time_diff < threshold_months * 30


def extract_keywords(text, topK=20, withWeight=False):
    try:
        from jieba import analyse
        keywords = analyse.extract_tags(text, topK=topK, withWeight=withWeight)
        if withWeight:
            keywords = [(kw, w) for kw, w in keywords if len(kw) > 1]
        else:
            keywords = [kw for kw in keywords if len(kw) > 1]
        return keywords
    except Exception:
        words = jieba.cut(text)
        return [word for word in words if len(word) > 1][:topK]


def extract_entities(text):
    entities = {'person': [], 'location': [], 'organization': [], 'other': []}
    
    try:
        import jieba.posseg as pseg
        words = pseg.cut(text)
        
        for word, flag in words:
            word = word.strip()
            if len(word) < 2:
                continue
                
            if flag in ['nr', 'nrt', 'nrfg']:
                entities['person'].append(word)
            elif flag in ['ns', 'nsf']:
                entities['location'].append(word)
            elif flag in ['nt', 'nto', 'nts', 'nth']:
                entities['organization'].append(word)
            elif flag in ['nz', 'nw']:
                entities['other'].append(word)
                
    except Exception:
        pass
    
    return entities


def compute_weighted_keyword_similarity(news_kws_weighted, doc_kws_weighted):
    if not news_kws_weighted or not doc_kws_weighted:
        return 0.0
    
    news_dict = {kw: w for kw, w in news_kws_weighted}
    doc_dict = {kw: w for kw, w in doc_kws_weighted}
    
    common_kws = set(news_dict.keys()) & set(doc_dict.keys())
    if not common_kws:
        return 0.0
    
    numerator = sum(min(news_dict[kw], doc_dict[kw]) for kw in common_kws)
    denominator = sum(news_dict.values())
    
    return numerator / denominator if denominator > 0 else 0.0


def compute_bm25_score(news_text, doc_texts):
    try:
        from rank_bm25 import BM25Okapi

        tokenized_corpus = []
        for text in doc_texts:
            tokens = list(jieba.cut(text))
            if not tokens:
                tokens = ["空文档"]
            tokenized_corpus.append(tokens)

        bm25 = BM25Okapi(tokenized_corpus)

        tokenized_query = list(jieba.cut(news_text))
        if not tokenized_query:
            tokenized_query = ["空查询"]

        scores = bm25.get_scores(tokenized_query)
        scores = [max(0.0, s) for s in scores]
        return scores
    except Exception:
        return [0.0] * len(doc_texts)


def compute_entity_overlap(news_entities, doc_entities):
    total_score = 0.0
    total_weight = 0.0

    weights = {'person': 3.0, 'organization': 2.5, 'location': 2.0, 'other': 1.0}

    for entity_type, weight in weights.items():
        news_set = set(news_entities.get(entity_type, []))
        doc_set = set(doc_entities.get(entity_type, []))

        if news_set:
            overlap = len(news_set & doc_set) / len(news_set)
            total_score += overlap * weight
            total_weight += weight

    return total_score / total_weight if total_weight > 0 else 0.0


def compute_content_score(news_text, article):
    title = article.get('title', '') or ''
    content = article.get('content', '') or ''
    full_text = title + '\n' + content
    
    news_kws_weighted = extract_keywords(news_text, topK=25, withWeight=True)
    doc_kws_weighted = extract_keywords(full_text, topK=30, withWeight=True)
    title_kws_weighted = extract_keywords(title, topK=15, withWeight=True)
    news_entities = extract_entities(news_text)
    doc_entities = extract_entities(full_text)
    
    bm25_scores = compute_bm25_score(news_text, [full_text])
    bm25_score = bm25_scores[0] if bm25_scores else 0.0
    bm25_score_normalized = min(bm25_score / 10.0, 1.0) if bm25_score > 0 else 0.0
    
    kw_sim = compute_weighted_keyword_similarity(news_kws_weighted, doc_kws_weighted)
    title_sim = compute_weighted_keyword_similarity(news_kws_weighted, title_kws_weighted)
    entity_score = compute_entity_overlap(news_entities, doc_entities)
    
    content_score = (
        bm25_score_normalized * 0.35 +
        kw_sim * 0.30 +
        title_sim * 0.25 +
        entity_score * 0.10
    )
    
    content_score = max(0.0, min(1.0, content_score))
    return content_score, {
        'bm25': bm25_score_normalized,
        'keyword': kw_sim,
        'title': title_sim,
        'entity': entity_score
    }


def triple_filter_mechanism(news_text, search_results):
    if not search_results:
        return None, "未找到任何搜索结果"
    candidates = []
    for idx, result in enumerate(search_results):
        content_score, details = compute_content_score(news_text, result)
        candidates.append({
            'result': result,
            'content_score': content_score,
            'details': details,
            'index': idx
        })
    
    if not candidates:
        return search_results[0], "保底返回第一个搜索结果"
    for cand in candidates:
        result = cand['result']
        url = result.get('url', '')
        authority_weight = 1.5 if check_source_whitelist(url) else 1.0
        cand['authority_weight'] = authority_weight
        time_score = 0.7
        article_time = extract_time_from_article(result)
        news_times = extract_time_from_text(news_text)
        if article_time and news_times:
            news_time = min(news_times, key=lambda t: abs((t - article_time).days))
            days_diff = abs((news_time - article_time).days)
            if '科技' in news_text or '突破' in news_text:
                half_life = 365
            elif '政策' in news_text:
                half_life = 180
            else:
                half_life = 90
            time_score = math.exp(-days_diff / half_life)
        cand['time_score'] = time_score
        cand['final_score'] = cand['content_score']*0.7 + cand['authority_weight']*0.2 + cand['time_score']*0.1
    candidates.sort(key=lambda x: x['final_score'], reverse=True)
    best = candidates[0]

    if best['final_score'] <= 0.30:
        return None, "综合得分过低，未找到相关证据"
    
    return best['result'], "找到相关证据"
