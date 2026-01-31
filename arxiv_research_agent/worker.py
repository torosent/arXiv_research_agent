"""Worker for the arXiv Research Agent using DurableTask SDK.

The worker registers all orchestrators and activities, then processes
tasks from the Durable Task Scheduler.
"""

import asyncio
import logging
import os
import sys

from azure.identity import DefaultAzureCredential, ManagedIdentityCredential
from durabletask.azuremanaged.worker import DurableTaskSchedulerWorker

from .activities import (
    search_arxiv_activity,
    analyze_papers_activity,
    identify_research_gaps_activity,
    decide_continuation_activity,
    synthesize_research_activity,
)
from .orchestrations import (
    paper_research_orchestrator,
    arxiv_research_orchestrator,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_credential():
    """Get Azure credential for authentication.
    
    Returns:
        Credential object or None for local emulator
    """
    endpoint = os.getenv("ENDPOINT", "http://localhost:8080")
    
    if endpoint == "http://localhost:8080":
        return None
    
    try:
        client_id = os.getenv("AZURE_MANAGED_IDENTITY_CLIENT_ID")
        if client_id:
            logger.info(f"Using Managed Identity with client ID: {client_id}")
            credential = ManagedIdentityCredential(client_id=client_id)
            credential.get_token("https://management.azure.com/.default")
            logger.info("Successfully authenticated with Managed Identity")
            return credential
        else:
            logger.info("Using DefaultAzureCredential")
            return DefaultAzureCredential()
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        logger.warning("Continuing without authentication - this may only work with local emulator")
        return None


async def main():
    """Main entry point for the worker process."""
    logger.info("Starting arXiv Research Agent worker...")
    
    # Get environment variables
    taskhub_name = os.getenv("TASKHUB", "default")
    endpoint = os.getenv("ENDPOINT", "http://localhost:8080")
    
    logger.info(f"Using taskhub: {taskhub_name}")
    logger.info(f"Using endpoint: {endpoint}")
    
    credential = get_credential()
    
    # Create worker
    with DurableTaskSchedulerWorker(
        host_address=endpoint,
        secure_channel=endpoint != "http://localhost:8080",
        taskhub=taskhub_name,
        token_credential=credential
    ) as worker:
        # Register activities
        worker.add_activity(search_arxiv_activity)
        worker.add_activity(analyze_papers_activity)
        worker.add_activity(identify_research_gaps_activity)
        worker.add_activity(decide_continuation_activity)
        worker.add_activity(synthesize_research_activity)
        
        # Register orchestrators
        worker.add_orchestrator(paper_research_orchestrator)
        worker.add_orchestrator(arxiv_research_orchestrator)
        
        logger.info("Worker registered all activities and orchestrators")
        logger.info("Starting worker...")
        
        # Start the worker
        worker.start()
        
        try:
            # Keep the worker running
            logger.info("Worker is running. Press Ctrl+C to stop.")
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Worker shutdown initiated")
    
    logger.info("Worker stopped")


if __name__ == "__main__":
    asyncio.run(main())
