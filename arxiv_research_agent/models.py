"""Data models for the arXiv Research Agent."""

from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict
from enum import Enum


class AgentStatusEnum(str, Enum):
    """Agent status enumeration."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass
class PaperReference:
    """Reference to an arXiv paper."""
    arxiv_id: str
    title: str
    authors: List[str]
    summary: str
    published: str
    primary_category: str
    categories: List[str]
    pdf_url: str
    abs_url: str
    comment: str = ""
    journal_ref: str = ""
    doi: str = ""


@dataclass
class EvaluationResult:
    """Result from evaluating search results."""
    query: str
    insights: List[str]
    relevance_score: int
    summary: str
    key_points: List[str]
    research_gaps: List[str] = field(default_factory=list)
    top_papers: List[PaperReference] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "query": self.query,
            "insights": self.insights,
            "relevance_score": self.relevance_score,
            "summary": self.summary,
            "key_points": self.key_points,
            "research_gaps": self.research_gaps,
            "top_papers": [
                {
                    "arxiv_id": p.arxiv_id,
                    "title": p.title,
                    "authors": p.authors,
                    "summary": p.summary,
                    "published": p.published,
                    "primary_category": p.primary_category,
                    "categories": p.categories,
                    "pdf_url": p.pdf_url,
                    "abs_url": p.abs_url,
                    "comment": p.comment,
                    "journal_ref": p.journal_ref,
                    "doi": p.doi,
                }
                for p in self.top_papers
            ]
        }


@dataclass
class ResearchReport:
    """Final research report."""
    report: str


@dataclass
class AgentStatus:
    """Status of a research agent."""
    created_at: str
    topic: str
    iterations: int
    report: Optional[str] = None
    status: AgentStatusEnum = AgentStatusEnum.PENDING
    agent_id: str = ""


@dataclass
class AgentStartRequest:
    """Request to start a new research agent."""
    topic: str
    max_iterations: int = 3


@dataclass
class ResearchTopicInput:
    """Input for researching a specific topic."""
    main_topic: str
    query: str


@dataclass
class ResearchWorkflowInput:
    """Input for the main research workflow."""
    topic: str
    max_iterations: int = 3
