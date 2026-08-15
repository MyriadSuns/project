# -*- coding: utf-8 -*-
"""搜索引擎模块"""

import re
import requests
from bs4 import BeautifulSoup
import jieba
from jieba.analyse import extract_tags
from urllib.parse import quote, urlparse
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
import time


def fetch_page_content(url, timeout=10):
    try:
        parsed = urlparse(url)
        if not parsed.scheme:
            url = 'https://' + url
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Connection': 'keep-alive'
        }
        
        response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        for script in soup(['script', 'style']):
            script.decompose()
        
        content = ''
        
        content_selectors = [
            '.content', '.article-content', '.article-body', '.main-content',
            '#content', '#article-content', '#article-body', '#main-content',
            '.post-content', '.news-content', '.news-body',
            'article', 'main', 'div[class*="content"]', 'div[class*="article"]'
        ]
        
        for selector in content_selectors:
            elements = soup.select(selector)
            if elements:
                for elem in elements:
                    content += elem.get_text(separator='\n', strip=True) + '\n'
                if content:
                    break
        
        if not content:
            content = soup.get_text(separator='\n', strip=True)
        
        content = re.sub(r'\n+', '\n', content)
        content = content.strip()
        
        return content[:5000] 
    except Exception as e:
        return ''


def _generate_intelligent_default_query(news_text):
    try:
        keywords = extract_tags(news_text, topK=4, withWeight=False, allowPOS=())
        unique_keywords = []
        seen = set()
        for kw in keywords:
            if kw and len(kw) > 1 and kw not in seen:
                unique_keywords.append(kw)
                seen.add(kw)
                if len(unique_keywords) >= 4:
                    break
        if unique_keywords:
            search_query = " ".join(unique_keywords)
            return search_query
    except Exception:
        pass
    
    search_query = news_text[:50]
    return search_query


def generate_search_query(llm_wrapper, news_text):
    search_query = news_text[:50]
    
    if llm_wrapper:
        try:
            prompt = "\n\n请概括以下新闻原文的核心内容。\n\n"
            prompt += "新闻原文：\n"
            prompt += news_text + "\n\n"
            prompt += "要求：\n"
            prompt += "1. 完全基于当前提供的新闻原文内容\n"
            prompt += "2. 生成一个客观、准确、简洁的陈述句作为回答，不超过20个字符，只有一句话。\n"
            prompt += "3. 陈述句应包含原文本身关键的人物、时间、地点和事件要素，严禁替换\n"
            prompt += "4. 不使用疑问句或感叹句，禁止带有任何感情色彩\n"
            prompt += "5. 不得替换、补充或推断任何未在原文中出现的信息，如'详细描述'、'注意根据实际修改'、'请注意：'等。不要与我对话\n"
            prompt += "6. 避免使用括号、引号、感叹号等特殊字符\n"
            prompt += "7. 使用中文回答\n"

            generated_query = llm_wrapper.chat_single_turn(
                prompt,
                system="""你是一个专业的搜索查询生成助手。请基于提供的新闻原文生成一个简短、聚焦的搜索查询。
                            重要要求：
                            1. 必须严格基于原文内容，不得添加原文不存在的信息
                            2. 只生成客观陈述句，不添加括号、感叹号、引号等标点
                            3. 保持简短，控制在20字以内
                            4. 重点提取核心事实，避免细节描述
                            5. 确保时间、地点、人物、事件等关键信息准确无误
                                
                            例如：
                            原文：教育部发布通知，要求中小学加强体育锻炼
                            正确查询：教育部要求中小学加强体育锻炼
                                
                            原文：5月1日起，北京市将实施新的垃圾分类管理条例
                            正确查询：北京5月1日起实施新垃圾分类管理条例
                        """,
                max_new_tokens=30,
                do_sample=False,
                repetition_penalty=1.2
            )
            
            if generated_query:
                generated_query = generated_query.strip()
                
                sentence_match = re.search(r'[^。.\n]*[。.\n]', generated_query)
                if sentence_match:
                    generated_query = sentence_match.group(0).strip()
            
                if generated_query and len(generated_query) > 5:
                    search_query = generated_query
                else:
                    search_query = _generate_intelligent_default_query(news_text)
        except Exception:
            search_query = _generate_intelligent_default_query(news_text)
    else:
        search_query = _generate_intelligent_default_query(news_text)
        
    return search_query


def baidu_search(query, max_results=50):
    results = []
    url = f"https://www.baidu.com/s?wd={quote(query)}"

    with sync_playwright() as p:
        browser = p.firefox.launch(
            headless=True,
            args=[
                '--disable-blink-features=AutomationControlled',
                '--disable-dev-shm-usage',
                '--no-sandbox',
            ]
        )
        context = browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        )
        page = context.new_page()

        try:
            page.goto(url, timeout=60000)
            page.wait_for_timeout(3000)
            html = page.content()
            
        except PlaywrightTimeoutError:
            browser.close()
            return []
        except Exception:
            browser.close()
            return []
        
        browser.close()

    soup = BeautifulSoup(html, 'html.parser')
    result_items = soup.select('.result.c-container')
    
    if not result_items:
        result_items = soup.select('#content_left .result')
    
    if not result_items:
        return []

    for item in result_items:
        title_elem = item.select_one('h3 a') or item.select_one('.t a')
        if not title_elem:
            continue
        
        title = title_elem.get_text(strip=True)
        url = title_elem.get('href', '')
        
        content = ''
        content_elem = item.select_one('.c-abstract') or item.select_one('.c-span9') or item.select_one('.c-color-text')
        if content_elem:
            content = content_elem.get_text(strip=True)
        
        if not content:
            all_text = item.get_text(separator=' ', strip=True)
            if len(all_text) > len(title):
                content = all_text[len(title):].strip()

        if not title:
            continue

        full_content = ''
        if url and len(results) < 50:  
            full_content = fetch_page_content(url)
            time.sleep(1) 
        
        if full_content:
            content = full_content

        results.append({
            'title': title,
            'content': content,
            'url': url
        })

        if len(results) >= max_results:
            break

    return results[:max_results]


def get_search_results(search_query):
    web_results = baidu_search(search_query, max_results=20)
    return web_results


def build_search_items(web_results):
    search_items = []

    for i, result in enumerate(web_results[:20]):
        if not isinstance(result, dict):
            continue
        title = (result.get('title') or '').strip()
        content = (result.get('body') or result.get('content') or result.get('snippet') or '').strip()
        url = (result.get('href') or result.get('url') or '').strip()

        if not title:
            continue

        search_item = {
            'url': url,
            'title': title,
            'content': content,
            'publish_time': result.get('publish_time', '')
        }
        search_items.append(search_item)
    
    return search_items
