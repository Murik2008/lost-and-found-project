import asyncio
from src.services.ai_service import AIService
from src.core.matcher import Matcher # və ya müvafiq yol: src.core.matcher
from src.concurrency.pipeline import BatchProcessingPipeline

async def main():
    # 1. Servis və obyektləri inisiallaşdırırıq
    ai_service = AIService()
    pipeline = BatchProcessingPipeline(ai_service)
    matcher = Matcher(ai_service)

    # 2. Test məlumatları
    test_items = [
        {"image_path": "path/to/image1.jpg"},
        {"description": "A black leather wallet found near the library"}
    ]

    print("--- Pipeline İşə Düşür ---")
    processed_items = await pipeline.process_batch(test_items)
    print("Emal olunmuş əşyalar:", processed_items)

    # 3. Matcher testi
    target = processed_items[0]
    candidates = processed_items[1:]
    
    matches = matcher.find_matches(target, candidates, k=1)
    print("--- Uyğunluq Nəticələri ---")
    print(matches)

if __name__ == "__main__":
    asyncio.run(main())