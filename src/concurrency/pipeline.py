import asyncio
from typing import Any, Dict, List
from src.services.ai_service import AIService


class BatchProcessingPipeline:
    """Pipeline for processing multiple items concurrently using asyncio and

    non-blocking execution threads.
    """

    def __init__(self, ai_service: AIService):
        self.ai_service = ai_service

    async def _process_single_item(
        self, item: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Processes a single item concurrently (image description and text

        embedding).
        """
        loop = asyncio.get_running_loop()

        # Generate image description if missing
        if not item.get("description") and item.get("image_path"):
            description = await loop.run_in_executor(
                None, self.ai_service.describe_item, item["image_path"]
            )
            item["description"] = description

        # Generate embedding if missing
        if not item.get("embedding") and item.get("description"):
            embedding = await loop.run_in_executor(
                None, self.ai_service.embed, item["description"]
            )
            item["embedding"] = embedding

        return item

    async def process_batch(
        self, items: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Processes a list of items concurrently."""
        tasks = [self._process_single_item(item) for item in items]
        processed_items = await asyncio.gather(*tasks)
        return list(processed_items)