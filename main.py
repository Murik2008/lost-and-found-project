import asyncio

from src.concurrency.pipeline import BatchProcessingPipeline
from src.core.matcher import Matcher
from src.services.ai_service import AIService


async def main():
    ai_service = AIService()
    pipeline = BatchProcessingPipeline(ai_service)
    matcher = Matcher(ai_service)

    items = [
        {"image_path": "data/lost/umbrella_black.png", "user_text": "black umbrella"},
        {"image_path": "data/found/umbrella_black_2.png", "user_text": "black umbrella found"},
    ]

    print("--- Pipeline ---")
    try:
        processed = await pipeline.process_batch(items)
        print(processed)
        matches = matcher.find_matches(processed[0], processed[1:], k=1)
        print("--- Matches ---")
        print(matches)
    except Exception as e:
        print(f"AI demo needs API keys or offline fakes: {e}")


if __name__ == "__main__":
    asyncio.run(main())
