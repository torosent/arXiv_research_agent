"""Activities for the arXiv Research Agent using DurableTask SDK.

Activities are the individual units of work that perform specific tasks
within an orchestration. They are durable and can be retried on failure.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from durabletask import task

from .arxiv_api import search_arxiv
from .llm import call_llm, parse_json_response

logger = logging.getLogger(__name__)

# Maximum number of papers to analyze per query
MAX_PAPERS_TO_ANALYZE = 15


def search_arxiv_activity(ctx: task.ActivityContext, query: str) -> List[Dict[str, Any]]:
    """Activity: Search arXiv for papers about a topic.
    
    Args:
        ctx: Activity context
        query: Search query string
        
    Returns:
        List of paper dictionaries
    """
    logger.info(f"Searching arXiv for: {query}")
    papers = search_arxiv(query, max_results=30)
    logger.info(f"Found {len(papers)} papers")
    return papers


def analyze_papers_activity(ctx: task.ActivityContext, input: Dict[str, Any]) -> Dict[str, Any]:
    """Activity: Analyze arXiv papers and extract academic insights using LLM.
    
    Args:
        ctx: Activity context
        input: Dictionary with topic, query, and papers
        
    Returns:
        Analysis result as dictionary
    """
    topic = input["topic"]
    query = input["query"]
    papers = input["papers"]
    
    logger.info(f"Analyzing papers for topic: {topic}, query: {query}")

    # Create detailed content digest for LLM
    papers_lines = []
    top_papers = []

    for i, paper in enumerate(papers[:MAX_PAPERS_TO_ANALYZE]):
        title = paper.get("title", "No title")
        arxiv_id = paper.get("arxiv_id", "")
        authors_list = paper.get("authors", [])
        authors = ", ".join(authors_list[:3])
        if len(authors_list) > 3:
            authors += " et al."
        summary = paper.get("summary", "")[:500]
        categories = ", ".join(paper.get("categories", [])[:3])
        published = paper.get("published", "")[:10]
        abs_url = paper.get("abs_url", "")
        pdf_url = paper.get("pdf_url", "")

        papers_lines.append(
            f"Paper {i+1}:\n"
            f"  Title: {title}\n"
            f"  arXiv ID: {arxiv_id}\n"
            f"  Authors: {authors}\n"
            f"  Published: {published}\n"
            f"  Categories: {categories}\n"
            f"  Abstract: {summary}...\n"
            f"  URL: {abs_url}\n"
        )

        top_papers.append({
            "arxiv_id": arxiv_id,
            "title": title,
            "authors": authors_list,
            "summary": paper.get("summary", ""),
            "published": paper.get("published", ""),
            "primary_category": paper.get("primary_category", ""),
            "categories": paper.get("categories", []),
            "pdf_url": pdf_url,
            "abs_url": abs_url,
            "comment": paper.get("comment", ""),
            "journal_ref": paper.get("journal_ref", ""),
            "doi": paper.get("doi", ""),
        })

    papers_text = "\n".join(papers_lines)
    
    prompt = f"""
    You are a research agent evaluating arXiv papers for: {topic}

    Query used: {query}

    Papers found:
    {papers_text}

    Provide a DETAILED analysis of these research papers. Focus on:
    - Key research contributions, methodologies, and techniques
    - Specific experimental results, metrics, or benchmarks
    - Novel approaches, architectures, or algorithms proposed
    - Connections between papers and emerging research themes
    - Identified research gaps or open problems
    - Practical applications and potential impact
    - Most influential or highly relevant papers for this topic

    Return JSON with:
    - "insights": String array of specific, technical insights from the papers
    - "relevance_score": Number 1-10 (how relevant are these papers to the research topic)
    - "summary": Brief summary of the research landscape
    - "key_points": Array of most important research findings
    - "research_gaps": Array of identified gaps or future research directions
    """
    
    messages = [
        {
            "role": "system",
            "content": "You are a research evaluation agent. Analyze arXiv papers and provide structured insights in JSON format. Focus on technical depth and research value.",
        },
        {"role": "user", "content": prompt},
    ]
    
    response = call_llm(messages, max_tokens=2000)
    try:
        evaluation_dict = parse_json_response(response)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse evaluation JSON: {e}, using defaults")
        evaluation_dict = {
            "insights": [],
            "relevance_score": 5,
            "summary": "Failed to parse LLM response",
            "key_points": [],
            "research_gaps": [],
        }
    evaluation_dict["query"] = query
    evaluation_dict["top_papers"] = top_papers

    return evaluation_dict


def identify_research_gaps_activity(ctx: task.ActivityContext, input: Dict[str, Any]) -> Optional[str]:
    """Activity: Identify research gaps and generate follow-up queries.
    
    Args:
        ctx: Activity context
        input: Dictionary with topic, current_findings, and iteration
        
    Returns:
        Next query to research, or None if no gaps identified
    """
    topic = input["topic"]
    current_findings = input["current_findings"]
    iteration = input["iteration"]
    
    logger.info(f"Identifying research gaps for iteration {iteration}")

    findings_lines = []
    for finding in current_findings:
        findings_lines.append(
            f"Query: {finding.get('query', 'Unknown')}\n"
            f"Summary: {finding.get('summary', 'No summary')}\n"
            f"Key insights: {finding.get('insights', [])}\n"
            f"Research gaps: {finding.get('research_gaps', [])}"
        )
    findings_summary = "\n\n".join(findings_lines)
    
    prompt = f"""
    You are a research agent investigating: {topic}
    
    This is iteration {iteration} of your research.
    
    Current findings:
    {findings_summary}
    
    Generate 2-4 SHORT KEYWORD-BASED search queries for arXiv that explore DIVERSE aspects of {topic}.
    
    CRITICAL RULES:
    1. Use SHORT keywords (2-5 words max) - NOT long sentences
    2. Focus on DIFFERENT aspects, methodologies, or applications
    3. Use terms that appear in actual arXiv paper titles
    4. Consider exploring identified research gaps
    5. Avoid repeating previous queries
    
    GOOD examples: ["transformer attention mechanisms", "neural network pruning", "federated learning privacy"]
    BAD examples: ["What are the latest advances in transformer-based architectures for natural language processing?"]
    
    Return only a JSON array of SHORT keyword queries: ["query1", "query2", "query3"]
    """
    
    messages = [
        {
            "role": "system",
            "content": "You are a research agent. Generate focused follow-up queries for arXiv search. Return only JSON array.",
        },
        {"role": "user", "content": prompt},
    ]
    
    response = call_llm(messages)
    try:
        queries = parse_json_response(response)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse follow-up queries JSON: {e}")
        queries = []
    return queries[0] if isinstance(queries, list) and queries else None


def decide_continuation_activity(ctx: task.ActivityContext, input: Dict[str, Any]) -> bool:
    """Activity: Decide whether to continue the literature review.
    
    Args:
        ctx: Activity context
        input: Dictionary with topic, all_findings, current_iteration, max_iterations
        
    Returns:
        True if literature review should continue, False otherwise
    """
    topic = input["topic"]
    all_findings = input["all_findings"]
    current_iteration = input["current_iteration"]
    max_iterations = input["max_iterations"]
    
    logger.info(f"Deciding whether to continue (iteration {current_iteration}/{max_iterations})")
    
    if current_iteration >= max_iterations:
        return False

    # Analyze findings completeness
    findings_lines = []
    total_relevance = 0
    for finding in all_findings:
        relevance = finding.get("relevance_score", 5)
        total_relevance += relevance
        findings_lines.append(
            f"Query: {finding.get('query', 'Unknown')}\n"
            f"Summary: {finding.get('summary', 'No summary')}\n"
            f"Relevance: {relevance}/10\n"
            f"Papers found: {len(finding.get('top_papers', []))}"
        )
    findings_summary = "\n\n".join(findings_lines)

    avg_relevance = total_relevance / len(all_findings) if all_findings else 0
    
    prompt = f"""
    You are a research agent investigating: {topic}
    
    Current iteration: {current_iteration}/{max_iterations}
    
    Findings so far:
    {findings_summary}
    
    Average relevance score: {avg_relevance:.1f}/10
    
    Decide whether to continue research or conclude. Continue if:
    1. Current iteration is less than 75% of max_iterations
    2. Average relevance is above 6.0 and there are likely unexplored aspects
    3. Recent queries found significant new papers with valuable insights
    4. There are identified research gaps worth exploring
    
    Only stop early if:
    - Average relevance is below 5.0 for multiple iterations
    - No new meaningful information in the last 2 iterations
    - The topic has been comprehensively covered
    
    Return JSON with:
    - "should_continue": boolean
    """
    
    messages = [
        {
            "role": "system",
            "content": "You are a research decision agent. Evaluate research completeness and decide whether to continue. Return JSON.",
        },
        {"role": "user", "content": prompt},
    ]
    
    raw_response = call_llm(messages)
    try:
        json_response = parse_json_response(raw_response)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse should_continue JSON: {e}, defaulting to False")
        json_response = {}
    return json_response.get("should_continue", False)


def synthesize_research_activity(ctx: task.ActivityContext, input: Dict[str, Any]) -> str:
    """Activity: Synthesize all research findings into a comprehensive report.
    
    Args:
        ctx: Activity context
        input: Dictionary with topic and all_findings
        
    Returns:
        Final research report as a string
    """
    topic = input["topic"]
    all_findings = input["all_findings"]
    
    logger.info(f"Synthesizing findings for topic: {topic}")
    
    findings_text = ""
    paper_citations = {}
    citation_id = 1
    
    for i, finding in enumerate(all_findings, 1):
        findings_text += f"\n=== Finding {i} ===\n"
        findings_text += f"Query: {finding.get('query', 'Unknown')}\n"
        findings_text += f"Summary: {finding.get('summary', 'No summary')}\n"
        findings_text += f"Key Points: {finding.get('key_points', [])}\n"
        findings_text += f"Insights: {finding.get('insights', [])}\n"
        findings_text += f"Research Gaps: {finding.get('research_gaps', [])}\n"
        
        # Extract paper citations
        if finding.get("top_papers"):
            for paper in finding["top_papers"]:
                arxiv_id = paper.get("arxiv_id", "")
                if arxiv_id and arxiv_id not in paper_citations:
                    authors = paper.get("authors", [])
                    author_str = authors[0] if authors else "Unknown"
                    if len(authors) > 1:
                        author_str += " et al."
                    paper_citations[arxiv_id] = {
                        "id": citation_id,
                        "title": paper.get("title", "Unknown"),
                        "authors": author_str,
                        "abs_url": paper.get("abs_url", ""),
                        "pdf_url": paper.get("pdf_url", ""),
                        "published": paper.get("published", "")[:10],
                        "categories": paper.get("categories", []),
                    }
                    citation_id += 1
    
    # Create citation references
    citations_text = "\n".join([
        f"[{cite['id']}] {cite['authors']}: \"{cite['title']}\" ({cite['published']}) - {cite['abs_url']}"
        for cite in paper_citations.values()
    ])
    
    prompt = f"""
    You are a research analyst. Synthesize the following arXiv research findings into a comprehensive,
    detailed report about: {topic}
    
    Research Findings:
    {findings_text}

    Available Paper Citations:
    {citations_text}
    
    Create a comprehensive research report that flows naturally as a single narrative. Include:
    - Overview of the research landscape and current state of the field
    - Key methodologies, techniques, and approaches in the literature
    - Important experimental results, benchmarks, and comparisons
    - Emerging trends and research directions
    - Identified gaps and opportunities for future research
    - Practical implications and applications
    - INLINE LINKS: When referencing papers, include clickable links using [paper title](URL) format
    
    Structure the report with clear sections:
    1. Executive Summary
    2. Research Landscape Overview
    3. Key Findings and Methodologies
    4. Emerging Trends
    5. Research Gaps and Future Directions
    6. Conclusions
    
    Return a JSON object with this exact structure:
    {{
        "report": "A comprehensive research report..."
    }}
    """
    
    messages = [
        {
            "role": "system",
            "content": "You are a research analyst specializing in academic literature review. Provide comprehensive synthesis in JSON format.",
        },
        {"role": "user", "content": prompt},
    ]
    
    raw_response = call_llm(messages, max_tokens=3000)
    try:
        json_response = parse_json_response(raw_response)
        return json_response.get("report", "No report generated")
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse synthesis JSON: {e}, returning raw response")
        # If JSON parsing fails, return the raw response as the report
        # (LLM may have returned plain text instead of JSON)
        return raw_response
