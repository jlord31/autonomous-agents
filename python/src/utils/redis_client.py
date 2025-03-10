import os
import logging
import redis
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

def get_redis_connection() -> Tuple[Optional[redis.Redis], bool]:
    """
    Get a Redis connection if available.
    Returns:
        Tuple containing:
            - Redis client or None if unavailable
            - Boolean indicating if Redis is available
    """
    try:
        client = redis.Redis(
            host=os.environ.get('REDIS_HOST', 'localhost'),
            port=int(os.environ.get('REDIS_PORT', 6379)),
            password=os.environ.get('REDIS_PASSWORD', ''),
            decode_responses=True,
            socket_timeout=5
        )
        # Test connection
        client.ping()
        logger.info("Connected to Redis successfully")
        return client, True
    except (ImportError, redis.RedisError) as e:
        logger.warning(f"Redis not available, using in-memory cache only: {str(e)}")
        return None, False

# Create a singleton instance
redis_client, use_redis = get_redis_connection()