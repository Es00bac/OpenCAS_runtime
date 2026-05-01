import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from opencas.bootstrap import BootstrapConfig
from opencas.bootstrap.pipeline import BootstrapPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("test_gemma")

async def test_gemma_fallback():
    config = BootstrapConfig(
        state_dir=Path("/tmp/opencas-public-fixture/.opencas"),
        workspace_root=Path("/tmp/opencas-public-fixture"),
    )
    pipeline = BootstrapPipeline(config)
    bctx = await pipeline.run()

    embeddings = bctx.embeddings

    logger.info("--- Testing FORCED Local Fallback ---")
    # We bypass the primary embed_fn to trigger the fallback_wrapper
    # (By setting model_id to something that will fail resolution if we wanted to be extreme,
    # but here we can just call the wrapper directly for testing)

    text = "Her memory is becoming more robust with local Gemma."

    try:
        # This will try Gemma -> Hash
        record = await embeddings._fallback_embed_wrapper(text)
        logger.info(f"Fallback Result Length: {len(record)}")

        # Check if Gemma was actually initialized
        gemma = await embeddings._get_local_gemma()
        if gemma:
            logger.info(f"Gemma model loaded: {gemma.model_id}")
            logger.info(f"Gemma dimension: {gemma.dimension}")
        else:
            logger.warning("Gemma failed to load, likely still installing or gated.")

    except Exception as e:
        logger.error(f"Test failed: {e}")

    await bctx.embeddings.cache.close()
    await bctx.memory.close()

if __name__ == "__main__":
    asyncio.run(test_gemma_fallback())
