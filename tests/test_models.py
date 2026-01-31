"""Tests for data models."""

import pytest
from arxiv_research_agent.models import (
    AgentStatusEnum,
    PaperReference,
    EvaluationResult,
    ResearchReport,
    AgentStatus,
    AgentStartRequest,
    ResearchTopicInput,
    ResearchWorkflowInput,
)


class TestAgentStatusEnum:
    """Tests for AgentStatusEnum."""

    def test_enum_values(self):
        """Test that all enum values exist."""
        assert AgentStatusEnum.PENDING == "PENDING"
        assert AgentStatusEnum.RUNNING == "RUNNING"
        assert AgentStatusEnum.COMPLETED == "COMPLETED"
        assert AgentStatusEnum.FAILED == "FAILED"

    def test_enum_is_string(self):
        """Test that enum values are strings."""
        assert isinstance(AgentStatusEnum.PENDING.value, str)


class TestPaperReference:
    """Tests for PaperReference dataclass."""

    def test_create_paper_reference(self):
        """Test creating a PaperReference."""
        paper = PaperReference(
            arxiv_id="2301.12345v1",
            title="Test Paper",
            authors=["John Smith", "Jane Doe"],
            summary="This is a test paper about deep learning.",
            published="2023-01-15T00:00:00Z",
            primary_category="cs.LG",
            categories=["cs.LG", "cs.AI"],
            pdf_url="https://arxiv.org/pdf/2301.12345v1",
            abs_url="https://arxiv.org/abs/2301.12345v1",
            comment="12 pages, 5 figures",
        )
        
        assert paper.arxiv_id == "2301.12345v1"
        assert paper.title == "Test Paper"
        assert len(paper.authors) == 2
        assert paper.primary_category == "cs.LG"
        assert len(paper.categories) == 2


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""

    def test_create_evaluation_result(self):
        """Test creating an EvaluationResult."""
        result = EvaluationResult(
            query="deep learning NLP",
            insights=["Transformers are effective", "Pre-training helps"],
            relevance_score=8,
            summary="Research in NLP has advanced significantly",
            key_points=["Attention mechanisms", "Transfer learning"],
        )
        
        assert result.query == "deep learning NLP"
        assert len(result.insights) == 2
        assert result.relevance_score == 8
        assert result.summary == "Research in NLP has advanced significantly"
        assert len(result.key_points) == 2
        assert result.top_papers == []
        assert result.research_gaps == []

    def test_to_dict(self):
        """Test converting EvaluationResult to dictionary."""
        paper = PaperReference(
            arxiv_id="2301.12345",
            title="Test Paper",
            authors=["John Smith"],
            summary="Test summary",
            published="2023-01-15",
            primary_category="cs.LG",
            categories=["cs.LG"],
            pdf_url="https://arxiv.org/pdf/2301.12345",
            abs_url="https://arxiv.org/abs/2301.12345",
        )
        result = EvaluationResult(
            query="query",
            insights=["insight"],
            relevance_score=7,
            summary="summary",
            key_points=["point"],
            research_gaps=["gap"],
            top_papers=[paper],
        )
        
        d = result.to_dict()
        
        assert d["query"] == "query"
        assert d["insights"] == ["insight"]
        assert d["relevance_score"] == 7
        assert d["summary"] == "summary"
        assert d["key_points"] == ["point"]
        assert d["research_gaps"] == ["gap"]
        assert len(d["top_papers"]) == 1
        assert d["top_papers"][0]["title"] == "Test Paper"


class TestResearchReport:
    """Tests for ResearchReport dataclass."""

    def test_create_research_report(self):
        """Test creating a ResearchReport."""
        report = ResearchReport(report="This is the research report content.")
        assert report.report == "This is the research report content."


class TestAgentStatus:
    """Tests for AgentStatus dataclass."""

    def test_create_agent_status_defaults(self):
        """Test creating AgentStatus with defaults."""
        status = AgentStatus(
            created_at="2024-01-01T00:00:00",
            topic="deep learning",
            iterations=2,
        )
        
        assert status.created_at == "2024-01-01T00:00:00"
        assert status.topic == "deep learning"
        assert status.iterations == 2
        assert status.report is None
        assert status.status == AgentStatusEnum.PENDING
        assert status.agent_id == ""

    def test_create_agent_status_full(self):
        """Test creating AgentStatus with all fields."""
        status = AgentStatus(
            created_at="2024-01-01T00:00:00",
            topic="machine learning",
            iterations=3,
            report="Final research report",
            status=AgentStatusEnum.COMPLETED,
            agent_id="agent-123",
        )
        
        assert status.report == "Final research report"
        assert status.status == AgentStatusEnum.COMPLETED
        assert status.agent_id == "agent-123"


class TestAgentStartRequest:
    """Tests for AgentStartRequest dataclass."""

    def test_create_with_defaults(self):
        """Test creating AgentStartRequest with defaults."""
        request = AgentStartRequest(topic="neural networks")
        
        assert request.topic == "neural networks"
        assert request.max_iterations == 3

    def test_create_with_custom_iterations(self):
        """Test creating AgentStartRequest with custom iterations."""
        request = AgentStartRequest(topic="transformers", max_iterations=5)
        
        assert request.max_iterations == 5


class TestResearchTopicInput:
    """Tests for ResearchTopicInput dataclass."""

    def test_create_research_topic_input(self):
        """Test creating ResearchTopicInput."""
        input_data = ResearchTopicInput(
            main_topic="deep learning",
            query="transformer attention mechanisms",
        )
        
        assert input_data.main_topic == "deep learning"
        assert input_data.query == "transformer attention mechanisms"


class TestResearchWorkflowInput:
    """Tests for ResearchWorkflowInput dataclass."""

    def test_create_with_defaults(self):
        """Test creating ResearchWorkflowInput with defaults."""
        input_data = ResearchWorkflowInput(topic="computer vision")
        
        assert input_data.topic == "computer vision"
        assert input_data.max_iterations == 3

    def test_create_with_custom_iterations(self):
        """Test creating ResearchWorkflowInput with custom iterations."""
        input_data = ResearchWorkflowInput(topic="NLP", max_iterations=5)
        
        assert input_data.max_iterations == 5
