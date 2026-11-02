"""Delete every object version of a (versioned) S3 bucket so CloudFormation can remove it.

``aws s3 rm --recursive`` only adds delete markers on a versioned bucket, and
``delete-stack`` then fails with "bucket not empty".

Usage:
    python -m scripts.empty_bucket <bucket-name>
"""

import logging
import sys

from src.logging_config import setup_logging

logger = logging.getLogger(__name__)


def empty_bucket(bucket_name: str) -> int:
    import boto3

    bucket = boto3.resource("s3").Bucket(bucket_name)
    deleted = 0
    for response in bucket.object_versions.delete():
        deleted += len(response.get("Deleted", []))
    logger.info("Deleted %d object versions from s3://%s", deleted, bucket_name)
    return deleted


if __name__ == "__main__":
    setup_logging()
    if len(sys.argv) != 2 or not sys.argv[1] or sys.argv[1] == "None":
        logger.info("No bucket to empty")
        sys.exit(0)
    empty_bucket(sys.argv[1])
